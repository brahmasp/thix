import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from dreamerv3 import jaxutils
import dreamerv3.nets.base as basenets
from dreamerv3 import ninjax as nj

f32 = jnp.float32
treemap = jax.tree_util.tree_map
tfd = tfp.distributions
sg = lambda x: treemap(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute
sample = lambda dist: {
    k: v.sample(seed=nj.seed()) for k, v in dist.items()}

class ActorCritic(nj.Module):
    def __init__(self, act_space, config, ac_config):
        self.act_space = act_space
        self.config = config # Full config
        self.ac_config = ac_config # Config specifically for this actor-critic

        kwargs = {}
        kwargs['shape'] = {
            k: (*s.shape, s.classes) if s.discrete else s.shape
            for k, s in self.act_space.items()}
        kwargs['dist'] = {
            k: ac_config.actor_dist_disc if v.discrete else ac_config.actor_dist_cont
            for k, v in self.act_space.items()}
        self.actor = basenets.MLP(**kwargs, **ac_config.actor, name='actor')
        self.retnorm = jaxutils.Moments(**ac_config.retnorm, name='retnorm')
        self.valnorm = jaxutils.Moments(**ac_config.valnorm, name='valnorm')
        self.advnorm = jaxutils.Moments(**ac_config.advnorm, name='advnorm')

        # Critic
        self.critic = basenets.MLP((), name='critic', **self.ac_config.critic)
        self.slowcritic = basenets.MLP(
            (), name='slowcritic', **self.ac_config.critic, dtype='float32')
        
        if config.thick_dreamer:
          # High level critic
          self.hl_critic = basenets.MLP((), name='hl_critic', **self.config.hl_critic.critic)
          self.hl_slowcritic = basenets.MLP(
            (), name='hl_slowcritic', **self.config.hl_critic.critic, dtype='float32')
        
    def policy(self, outs):
        return sample(self.actor(outs, bdims=1))

    def loss(self, data, outs, acts, con, rew, replay_outs, update, hl_out = {}):
        losses = {}
        acts = sg(acts)
        inp = treemap({
            'none': lambda x: sg(x),
            'first': lambda x: jnp.concatenate([x[:, :1], sg(x[:, 1:])], 1),
            'all': lambda x: x,
        }[self.ac_config.grads], outs)
        actor = self.actor(inp)
        critic = self.critic(inp)
        slowcritic = self.slowcritic(inp)
        voffset, vscale = self.valnorm.stats()
        val = critic.mean() * vscale + voffset
        slowval = slowcritic.mean() * vscale + voffset
        tarval = slowval if self.ac_config.slowtar else val
        discount = 1 if self.config.contdisc else 1 - 1 / self.ac_config.horizon
        weight = jnp.cumprod(discount * con, 1) / discount

        # Return
        rets = [tarval[:, -1]]
        disc = con[:, 1:] * discount
        lam = self.ac_config.return_lambda
        interm = rew[:, 1:] + (1 - lam) * disc * tarval[:, 1:]
        for t in reversed(range(disc.shape[1])):
            rets.append(interm[:, t] + disc[:, t] * lam * rets[-1])
        ret = jnp.stack(list(reversed(rets))[:-1], 1)

        # Actor
        roffset, rscale = self.retnorm(ret, update)
        adv = (ret - tarval[:, :-1]) / rscale
        aoffset, ascale = self.advnorm(adv, update)
        adv_normed = (adv - aoffset) / ascale
        logpi = sum([v.log_prob(sg(acts[k]))[:, :-1] for k, v in actor.items()])
        ents = {k: v.entropy()[:, :-1] for k, v in actor.items()}
        actor_loss = sg(weight[:, :-1]) * -(logpi * sg(adv_normed) + self.ac_config.actent * sum(ents.values()))
        losses['actor'] = actor_loss

        # Critic
        v_long = 0
        if self.config.thick_dreamer:
            # if thick_dreamer, then target (ret) needs to be modified
            # to include high-level critic target
            hl_critic = self.hl_critic(inp)
            hl_slowcritic = self.hl_slowcritic(inp)
            # one normalization for both critics since they regress to the same value
            hl_voffset, hl_vscale = self.valnorm.stats()
            hl_val = hl_critic.mean() * hl_vscale + hl_voffset
            hl_slowval = hl_slowcritic.mean() * hl_vscale + hl_voffset
            hl_tarval = hl_slowval if self.config.hl_critic.slowtar else hl_val
            sub_v_long = rew + con * hl_tarval
            v_long = hl_out['reward'] + (discount ** hl_out['t_delta']) * (sub_v_long)

        ret_padded = jnp.concatenate([ret, 0 * ret[:, -1:]], 1)
        final_target = self.config.hl_critic.critic_psi * ret_padded + (1 - self.config.hl_critic.critic_psi) * v_long

        ftarget_offset, ftarget_scale = self.valnorm(final_target, update)
        final_target_normed = (final_target - ftarget_offset) / ftarget_scale

        losses['critic'] = sg(weight)[:, :-1] * -(critic.log_prob(sg(final_target_normed)) +
                                                  self.ac_config.slowreg * critic.log_prob(sg(slowcritic.mean())))[:, :-1]
        if self.config.thick_dreamer:
            losses['hl_critic'] = sg(weight)[:, -1:] * -(hl_critic.log_prob(sg(final_target_normed)) +
                                                  self.config.hl_critic.slowreg * hl_critic.log_prob(sg(hl_slowcritic.mean())))[:, -1:]
        replay_ret = None
        if self.ac_config.replay_critic_loss:
            replay_critic = self.critic(replay_outs if self.ac_config.replay_critic_grad else sg(replay_outs))
            replay_slowcritic = self.slowcritic(replay_outs)
            boot = dict(
                imag=ret[:, 0].reshape(data['reward'].shape),
                critic=replay_critic.mean(),
            )[self.ac_config.replay_critic_bootstrap]
            rets = [boot[:, -1]]
            live = f32(~data['is_terminal'])[:, 1:] * (1 - 1 / self.ac_config.horizon)
            cont = f32(~data['is_last'])[:, 1:] * self.ac_config.return_lambda_replay
            interm = data['reward'][:, 1:] + (1 - cont) * live * boot[:, 1:]
            for t in reversed(range(live.shape[1])):
                rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
            replay_ret = jnp.stack(list(reversed(rets))[:-1], 1)
            voffset, vscale = self.valnorm(replay_ret, update)
            ret_normed = (replay_ret - voffset) / vscale
            ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
            losses['replay_critic'] = sg(f32(~data['is_last']))[:, :-1] * -(replay_critic.log_prob(
                sg(ret_padded)) +self.ac_config.slowreg * replay_critic.log_prob(sg(replay_slowcritic.mean())))[:, :-1]

        metrics = self._metrics(acts, actor, adv, rew, weight, val, ret, ents, roffset, rscale, replay_ret)
        return losses, metrics

    def _metrics(self, acts, actor, adv, rew, weight, val, ret, ents, roffset, rscale, replay_ret):
        metrics = {}
        metrics.update(jaxutils.tensorstats(adv, 'adv'))
        metrics.update(jaxutils.tensorstats(rew, 'rew'))
        metrics.update(jaxutils.tensorstats(weight, 'weight'))
        metrics.update(jaxutils.tensorstats(val, 'val'))
        metrics.update(jaxutils.tensorstats(ret, 'ret'))
        metrics.update(jaxutils.tensorstats(
            (ret - roffset) / rscale, 'ret_normed'))
        if self.ac_config.replay_critic_loss:
            metrics.update(jaxutils.tensorstats(replay_ret, 'replay_ret'))
        metrics['td_error'] = jnp.abs(ret - val[:, :-1]).mean()
        metrics['ret_rate'] = (jnp.abs(ret) > 1.0).mean()

        for k, space in self.act_space.items():
            act = f32(jnp.argmax(acts[k], -1) if space.discrete else acts[k])
            metrics.update(jaxutils.tensorstats(f32(act), f'act/{k}'))
            if hasattr(actor[k], 'minent'):
                lo, hi = actor[k].minent, actor[k].maxent
                rand = ((ents[k] - lo) / (hi - lo)).mean(
                    range(2, len(ents[k].shape)))
                metrics.update(jaxutils.tensorstats(rand, f'rand/{k}'))
            metrics.update(jaxutils.tensorstats(ents[k], f'ent/{k}'))
        return metrics