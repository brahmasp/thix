import re
from functools import partial as bind

import embodied
import jax
import jax.numpy as jnp
import numpy as np

from dreamerv3 import jaxutils
import dreamerv3.nets.encdec as encdec
import dreamerv3.nets.base as basenets
import dreamerv3.nets.rssm as rssm
import dreamerv3.nets.crssm as crssm
from dreamerv3 import ninjax as nj
from dreamerv3.utils import imagine

f32 = jnp.float32
treemap = jax.tree_util.tree_map
sg = lambda x: treemap(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute
sample = lambda dist: {
    k: v.sample(seed=nj.seed()) for k, v in dist.items()}

def report_imagination_losses(img_data, img_preds, looptype):
    metrics = {}
    losses = {k: -v.log_prob(img_data[k].astype(f32)) for k, v in img_preds.items()}
    metrics.update({f'{looptype}_{k}_loss': v.mean() for k, v in losses.items()})
    stats = jaxutils.balance_stats(img_preds['reward'], img_data['reward'], 0.1)
    metrics.update({f'{looptype}_reward_{k}': v for k, v in stats.items()})
    stats = jaxutils.balance_stats(img_preds['cont'], img_data['cont'], 0.5)
    metrics.update({f'{looptype}_cont_{k}': v for k, v in stats.items()})
    return metrics

def make_imagination_video(data, obs_preds, img_preds, image_keys, looptype):
    metrics = {}
    for key in image_keys:
        true = f32(data[key][:6])
        pred = jnp.concatenate([obs_preds[key].mode()[:6], img_preds[key].mode()[:6]], 1)
        error = (pred - true + 1) / 2
        video = jnp.concatenate([true, pred, error], 2)
        metrics[f'{looptype}/{key}'] = jaxutils.video_grid(video)
    return metrics

def vector_video(obs_preds, img_preds, vec_key, looptype, pixel_size=10, frame=True):
    # TODO assumes vec in [-1, 1]
    vec = jnp.concatenate([obs_preds[vec_key][:6], img_preds[vec_key][:6]], 1).astype(jnp.float32)
    vs = vec.shape
    mask = np.zeros((vs[0], vs[1], int(vs[2] * pixel_size), pixel_size, 3)) # zeros
    if frame:
        mask[:, :, 0, :] = 2.0
        mask[:, :, -1, :] = 2.0
        mask[:, :, :, 0] = 2.0
        mask[:, :, :, -1] = 2.0
    vec = jaxutils.expand_repeat(vec, -1, pixel_size)
    vec = jnp.repeat(vec, pixel_size, axis=2)
    vec = jaxutils.expand_repeat(vec, -1, 3)
    video = jaxutils.video_grid(jnp.clip(vec - mask, -1, 1) * 0.5 + 0.5)
    return {f'{looptype}/{vec_key}': video}

class WorldModel(nj.Module):
    def __init__(self, obs_space, act_space, config):
        self.obs_space = {
            k: v for k, v in obs_space.items() if not k.startswith('log_')}
        self.act_space = {
            k: v for k, v in act_space.items() if k != 'reset'}
        self.config = config
        enc_space = {
            k: v for k, v in obs_space.items()
            if k not in ('is_first', 'is_last', 'is_terminal', 'reward') and
            not k.startswith('log_') and re.match(config.enc.spaces, k)}
        dec_space = {
            k: v for k, v in obs_space.items()
            if k not in ('is_first', 'is_last', 'is_terminal', 'reward') and
            not k.startswith('log_') and re.match(config.dec.spaces, k)}
        embodied.print('Encoder:', {k: v.shape for k, v in enc_space.items()})
        embodied.print('Decoder:', {k: v.shape for k, v in dec_space.items()})

        self.enc = {
            'simple': bind(encdec.SimpleEncoder, **config.enc.simple),
        }[config.enc.typ](enc_space, name='enc')
        self.dec = {
            'simple': bind(encdec.SimpleDecoder, **config.dec.simple),
        }[config.dec.typ](dec_space, name='dec')
        self.dyn = {
            'rssm': bind(rssm.RSSM, **config.dyn.rssm),
            'crssm': bind(crssm.CRSSM, **{**config.dyn.rssm, **config.dyn.crssm}),
        }[config.dyn.typ](name='dyn')
        self.rew = basenets.MLP((), **config.rewhead, name='rew')
        self.con = basenets.MLP((), **config.conhead, name='con')
        self.log_video_modes = config.log_video_modes
        for mode in self.log_video_modes: assert mode in ['openl', 'closedl']
        self.video_latents = config.video_latents

    def observe(self, data, carry):
        prevlat, prevact = carry
        if len(data['is_first'].shape) > 1:
            acts = {
                k: jnp.concatenate([prevact[k][:, None], data[k][:, :-1]], 1)
                for k in self.act_space}
        else:  # There is no sequence dimension, only (batch,)
            assert not any(k in data for k in self.act_space), "Single observations shouldn't contain actions"
            acts = prevact

        acts = jaxutils.onehot_dict(acts, self.act_space)
        embed = self.enc(data)
        newlat, dyn_outs = self.dyn.observe(prevlat, acts, embed, data['is_first'])
        wm_outs = {'embed': embed}
        return newlat, dyn_outs, wm_outs

    def imagine(self, policy, data, replay_outs):
        def imgstep(carry, _):
            lat, act = carry
            lat, out = self.dyn.imagine(lat, act)
            out['stoch'] = sg(out['stoch'])
            act = policy(out)
            return (lat, act), (out, act)

        rew = data['reward']
        con = 1 - f32(data['is_terminal'])
        if self.config.imag_start == 'all':
            B, T = data['is_first'].shape
            startlat = self.dyn.outs_to_carry(treemap(
                lambda x: x.reshape((B * T, 1, *x.shape[2:])), replay_outs))
            startout, startrew, startcon = treemap(
                lambda x: x.reshape((B * T, *x.shape[2:])),
                (replay_outs, rew, con))
        elif self.config.imag_start == 'last':
            startlat = self.dyn.outs_to_carry(replay_outs)
            startout, startrew, startcon = treemap(
                lambda x: x[:, -1], (replay_outs, rew, con))
        else:
            raise ValueError(f"Invalid imag_start: {self.config.imag_start}")
        if self.config.imag_repeat > 1:
            N = self.config.imag_repeat
            startlat, startout, startrew, startcon = treemap(
                lambda x: x.repeat(N, 0), (startlat, startout, startrew, startcon))
        startact = policy(startout)
        start_data, start_outs = (startlat, startact), (startout, startact)
        outs, acts = imagine(imgstep, start_data, self.config.imag_length, self.config.imag_unroll, start_outs)
        rew = jnp.concatenate([startrew[:, None], self.rew(outs).mean()[:, 1:]], 1)
        con = jnp.concatenate([startcon[:, None], self.con(outs).mean()[:, 1:]], 1)
        return outs, acts, rew, con

    def loss(self, dyn_outs, wm_outs, data):
        metrics = {}
        rew_feat = dyn_outs if self.config.reward_grad else sg(dyn_outs)
        dists = dict(
            **self.dec(dyn_outs),
            reward=self.rew(rew_feat, training=True),
            cont=self.con(dyn_outs, training=True))
        losses = {k: -v.log_prob(f32(data[k])) for k, v in dists.items()}
        if self.config.contdisc:
            del losses['cont']
            softlabel = data['cont'] * (1 - 1 / self.config.ac.horizon)
            losses['cont'] = -dists['cont'].log_prob(softlabel)
        dynlosses, mets = self.dyn.loss(dyn_outs)
        losses.update(dynlosses)
        metrics.update(mets)
        metrics = self._metrics(dists, wm_outs['embed'], data)
        return losses, metrics

    def _metrics(self, dists, embed, data):
        metrics = {}
        if 'reward' in dists:
            stats = jaxutils.balance_stats(dists['reward'], data['reward'], 0.1)
            metrics.update({f'rewstats/{k}': v for k, v in stats.items()})
        if 'cont' in dists:
            stats = jaxutils.balance_stats(dists['cont'], data['cont'], 0.5)
            metrics.update({f'constats/{k}': v for k, v in stats.items()})
        metrics['activation/embed'] = jnp.abs(embed).mean()
        return metrics

    def report(self, data, carry):
        # Open loop predictions
        B, T = data['is_first'].shape
        assert self.config.report_openl_context != 0, "report_openl_context must be greater than 0"
        openl_start_obs = min(self.config.report_openl_context, T // 2)
        metrics = {}
        for mode in self.log_video_modes:
            num_obs = openl_start_obs if mode == 'openl' else -2

            img_start, rec_outs, wm_outs = self.observe({k: v[:, :num_obs] for k, v in data.items()}, carry)

            img_acts = {k: data[k][:, num_obs:] for k in self.act_space}
            img_acts = jaxutils.onehot_dict(img_acts, self.act_space)

            img_outs = self.dyn.imagine(img_start, img_acts)[1]
            metrics.update(self._report_imagination_rollout(data, rec_outs, img_outs, mode))

        return metrics

    def _report_imagination_rollout(self, data, obs_outs, img_outs, looptype):
        metrics = {}
        obs_preds = dict(
            **self.dec(obs_outs), reward=self.rew(obs_outs),
            cont=self.con(obs_outs))
        img_preds = dict(
            **self.dec(img_outs), reward=self.rew(img_outs),
            cont=self.con(img_outs))

        num_obs = obs_outs['deter'].shape[1]
        img_data = {k: v[:, num_obs:] for k, v in data.items()}
        metrics.update(report_imagination_losses(img_data, img_preds, looptype))
        metrics.update(make_imagination_video(data, obs_preds, img_preds, self.dec.imgkeys, looptype))
        for latent_key in self.video_latents:
            if latent_key not in obs_outs.keys():
                continue

            metrics.update(vector_video(obs_outs, img_outs, latent_key, looptype))

        return metrics

    def expand_scales(self, scales):
        cnn = scales.pop('dec_cnn')
        mlp = scales.pop('dec_mlp')
        scales.update({k: cnn for k in self.dec.imgkeys})
        scales.update({k: mlp for k in self.dec.veckeys})

    def policy_logs(self, obs, dyn_outs, wm_outs, carry):
        return {}
