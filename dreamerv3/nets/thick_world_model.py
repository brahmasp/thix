import re
from functools import partial as bind

import jax
import jax.numpy as jnp

from dreamerv3 import jaxutils
import dreamerv3.nets.encdec as encdec
import dreamerv3.nets.base as basenets
from dreamerv3 import ninjax as nj
from dreamerv3.nets.world_model import WorldModel, report_imagination_losses, make_imagination_video

f32 = jnp.float32
treemap = jax.tree_util.tree_map
sg = lambda x: treemap(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute
sample = lambda dist: {
    k: v.sample(seed=nj.seed()) for k, v in dist.items()}

class ThickWorldModel(WorldModel):
    def __init__(self, obs_space, act_space, config):
        super().__init__(obs_space, act_space, config)
        if config.use_coarse_dec_head:
            dec_space = {
                k: v for k, v in obs_space.items()
                if k not in ('is_first', 'is_last', 'is_terminal', 'reward') and
                not k.startswith('log_') and re.match(config.dec.spaces, k)}
            self.coarse_dec = {
                'simple': bind(encdec.SimpleDecoder, **config.coarse_dec.simple),
            }[config.dec.typ](dec_space, name='coarse_dec')
        if config.use_coarse_rew_head:
            self.coarse_rew = basenets.MLP((), **config.coarse_rewhead, name='coarse_rew')
        if config.use_coarse_con_head:
            self.coarse_con = basenets.MLP((), **config.coarse_conhead, name='coarse_con')

    def loss(self, dyn_outs, wm_outs, data):
        losses, metrics = super().loss(dyn_outs, wm_outs, data)
        coarse_dists = self._coarse_predictions(dyn_outs)
        losses.update({'coarse_' + k: -v.log_prob(f32(data[k])) for k, v in coarse_dists.items() if k != 'cont'})
        if 'cont' in coarse_dists:
            targets = data['cont'] * (1 - 1 / self.config.ac.horizon) if self.config.contdisc else data['cont']
            losses['coarse_cont'] = -coarse_dists['cont'].log_prob(targets)
        metrics = self._coarse_metrics(coarse_dists, data)
        return losses, metrics

    def _coarse_metrics(self, coarse_dists, data):
        metrics = {}
        if 'reward' in coarse_dists:
            stats = jaxutils.balance_stats(coarse_dists['reward'], data['reward'], 0.1)
            metrics.update({f'coarse_rewstats/{k}': v for k, v in stats.items()})
        if 'cont' in coarse_dists:
            stats = jaxutils.balance_stats(coarse_dists['cont'], data['cont'], 0.5)
            metrics.update({f'coarse_constats/{k}': v for k, v in stats.items()})
        return metrics

    def _report_imagination_rollout(self, data, obs_outs, img_outs, openl):
        metrics = super()._report_imagination_rollout(data, obs_outs, img_outs, openl)
        obs_preds = self._coarse_predictions(obs_outs)
        img_preds = self._coarse_predictions(img_outs)

        num_obs = obs_outs['deter'].shape[1]
        img_data = {k: v[:, num_obs:] for k, v in data.items()}
        metrics.update({'coarse_' + k: v for k, v in report_imagination_losses(img_data, img_preds, openl).items()})
        metrics.update({'coarse_' + k: v for k, v in make_imagination_video(data, obs_preds, img_preds, self.coarse_dec.imgkeys, openl).items()})

        return metrics

    def expand_scales(self, scales):
        super().expand_scales(scales)
        if self.config.use_coarse_dec_head:
            cnn = scales.pop('coarse_dec_cnn')
            mlp = scales.pop('coarse_dec_mlp')
            scales.update({'coarse_' + k: cnn for k in self.coarse_dec.imgkeys})
            scales.update({'coarse_' + k: mlp for k in self.coarse_dec.veckeys})
        return scales

    def _coarse_predictions(self, inputs):
        preds = self.coarse_dec(inputs) if self.config.use_coarse_dec_head else {}
        if self.config.use_coarse_rew_head:
            preds['reward'] = self.coarse_rew(inputs)
        if self.config.use_coarse_con_head:
            preds['cont'] = self.coarse_con(inputs)
        return preds

    def policy_logs(self, obs, dyn_outs, wm_outs, carry):
        log_context_change = jnp.any(dyn_outs['coarse_gates'], axis=-1)
        logs = {'log_context_change': log_context_change, 'log_coarse_gates': dyn_outs['coarse_gates']}
        for key in self.dec.imgkeys:
            context_change_signal_image = jnp.where(log_context_change[:, None],
                                                    jnp.array([0, 1, 0], dtype=jnp.float32)[None, :],
                                                    jnp.array([1, 0, 0], dtype=jnp.float32)[None, :])
            context_change_signal_image = jnp.repeat(context_change_signal_image[:, None], 8, 1)
            context_change_signal_image = jnp.repeat(context_change_signal_image[:, None], obs[key].shape[2], 1)
            obs_with_context_change = jnp.concatenate([obs[key], context_change_signal_image], axis=-2)
            logs[f'log_obs_with_context_change/{key}'] = obs_with_context_change
        return logs