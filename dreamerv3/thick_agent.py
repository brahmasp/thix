import embodied
import jax
import jax.numpy as jnp
import numpy as np
import optax
import ruamel.yaml as yaml

from . import jaxagent
from . import jaxutils
import dreamerv3.nets.thick_world_model as thick
import dreamerv3.nets.high_level_world_model as hlworldmodel
import dreamerv3.nets.actorcritic as actorcritic
from . import ninjax as nj

f32 = jnp.float32
treemap = jax.tree_util.tree_map
sg = lambda x: treemap(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute
sample = lambda dist: {
    k: v.sample(seed=nj.seed()) for k, v in dist.items()}


@jaxagent.Wrapper
class ThickAgent(nj.Module):

  configs = yaml.YAML(typ='safe').load(
      (embodied.Path(__file__).parent / 'configs.yaml').read())

  def __init__(self, obs_space, act_space, config):
    self.obs_space = {
        k: v for k, v in obs_space.items() if not k.startswith('log_')}
    self.act_space = {
        k: v for k, v in act_space.items() if k != 'reset'}
    self.config = config

    assert config.wm.typ == 'thick', "ThickAgent only supports thick world model"
    self.wm = thick.ThickWorldModel(obs_space, act_space, config, name='wm')
    self.hlwm = hlworldmodel.HierarchicalWorldModel(self.obs_space, self.act_space, config, self.wm, name='hlwm')

    # Actor
    self.ac = actorcritic.ActorCritic(self.act_space, config, config.ac, name='ac')
    self.updater = jaxutils.SlowUpdater(
      self.ac.critic, self.ac.slowcritic,
      self.config.ac.slow_critic_fraction,
      self.config.ac.slow_critic_update,
      name='ac_updater')
  
    if config.thick_dreamer:
      self.hl_updater = jaxutils.SlowUpdater(
        self.ac.hl_critic, self.ac.hl_slowcritic,
        self.config.hl_critic.slow_critic_fraction,
        self.config.hl_critic.slow_critic_update,
        name='hl_critic_updater')

    # Optimizer
    kw = dict(config.opt)
    lr = kw.pop('lr')
    if config.separate_lrs:
      lr = {f'agent/{k}': v for k, v in config.lrs.items()}
    self.opt = jaxutils.Optimizer(lr, **kw, name='opt')
    self.modules = [self.wm, self.ac.actor, self.ac.critic, self.hlwm]
    self.scales = self.config.loss_scales.copy()
    self.wm.expand_scales(self.scales)
    self.hlwm.expand_scales(self.scales)

  @property
  def policy_keys(self):
    return r'(wm/enc|wm/dyn|ac/actor)'

  @property
  def aux_spaces(self):
    spaces = self.wm.dyn.spaces if self.config.replay_context else {}
    spaces['stepid'] = embodied.Space(np.uint8, 20)
    return spaces

  def init_policy(self, batch_size):
    prevact = {
        k: jnp.zeros((batch_size, *v.shape), v.dtype)
        for k, v in self.act_space.items()}
    return self.wm.dyn.initial(batch_size), prevact

  def init_train(self, batch_size):
    prevact = {
        k: jnp.zeros((batch_size, *v.shape), v.dtype)
        for k, v in self.act_space.items()}
    return self.wm.dyn.initial(batch_size), prevact

  def init_report(self, batch_size):
    return self.init_train(batch_size)

  def policy(self, obs, carry, mode='train'):
    self.config.jax.jit and embodied.print(
        'Tracing policy function', color='yellow')
    prevlat, prevact = carry
    obs = self.preprocess(obs)
    lat, dyn_outs, wm_outs = self.wm.observe(obs, carry)
    act = self.ac.policy(dyn_outs)

    outs = {}
    if self.config.replay_context:
      outs.update({k: dyn_outs[k] for k in self.aux_spaces if k != 'stepid'})
      outs['stoch'] = jnp.argmax(outs['stoch'], -1).astype(jnp.int32)

    outs['finite'] = {
        '/'.join(x.key for x in k): (
            jnp.isfinite(v).all(range(1, v.ndim)),
            v.min(range(1, v.ndim)),
            v.max(range(1, v.ndim)))
        for k, v in jax.tree_util.tree_leaves_with_path(dict(
            obs=obs, prevlat=prevlat, prevact=prevact, act=act, dyn_out=dyn_outs, wm_out=wm_outs, lat=lat,
        ))}

    assert all(
        k in outs for k in self.aux_spaces
        if k not in ('stepid', 'finite', 'is_online')), (
              list(outs.keys()), self.aux_spaces)

    outs.update(self.wm.policy_logs(obs, dyn_outs, wm_outs, carry))

    act = {
        k: jnp.nanargmax(act[k], -1).astype(jnp.int32)
        if s.discrete else act[k] for k, s in self.act_space.items()}
    return act, outs, (lat, act)

  def train(self, data, carry):
    self.config.jax.jit and embodied.print(
        'Tracing train function', color='yellow')
    data = self.preprocess(data)
    stepid = data.pop('stepid')

    if self.config.replay_context:
      data, carry = self._load_replay_context(data)
      stepid = stepid[:, self.config.replay_context:]

    if self.config.reset_context:
      keep = jax.random.uniform(
          nj.seed(), data['is_first'][:, :1].shape) > self.config.reset_context
      data['is_first'] = jnp.concatenate([
          data['is_first'][:, :1] & keep, data['is_first'][:, 1:]], 1)

    mets, (out, carry, metrics) = self.opt(
        self.modules, self.loss, data, carry, has_aux=True)
    metrics.update(mets)
    self.updater()
    if self.config.thick_dreamer:
      self.hl_updater()
    outs = {}

    if self.config.replay_context:
      outs['replay'] = {'stepid': stepid}
      outs['replay'].update({
          k: out['replay_outs'][k] for k in self.aux_spaces if k != 'stepid'})
      outs['replay']['stoch'] = jnp.argmax(
          outs['replay']['stoch'], -1).astype(jnp.int32)

    if self.config.replay.fracs.priority > 0:
      bs = data['is_first'].shape
      if self.config.replay.priosignal == 'td':
        priority = out['critic_loss'][:, 0].reshape(bs)
      elif self.config.replay.priosignal == 'model':
        terms = [out[f'{k}_loss'] for k in (
            'rep', 'dyn', *self.wm.dec.veckeys, *self.wm.dec.imgkeys)]
        priority = jnp.stack(terms, 0).sum(0)
      elif self.config.replay.priosignal == 'all':
        terms = [out[f'{k}_loss'] for k in (
            'rep', 'dyn', *self.wm.dec.veckeys, *self.wm.dec.imgkeys)]
        terms.append(out['actor_loss'][:, 0].reshape(bs))
        terms.append(out['critic_loss'][:, 0].reshape(bs))
        priority = jnp.stack(terms, 0).sum(0)
      else:
        raise NotImplementedError(self.config.replay.priosignal)
      assert stepid.shape[:2] == priority.shape == bs
      outs['replay'] = {'stepid': stepid, 'priority': priority}

    return outs, carry, metrics

  def _load_replay_context(self, data):
    K = self.config.replay_context
    data = data.copy()
    context = {
      k: data.pop(k)[:, :K] for k in self.aux_spaces if k != 'stepid'}
    context['stoch'] = f32(jax.nn.one_hot(
      context['stoch'], self.config.dyn.rssm.classes))
    prevlat = self.wm.dyn.outs_to_carry(context)
    prevact = {k: data[k][:, K - 1] for k in self.act_space}
    carry = prevlat, prevact
    data = {k: v[:, K:] for k, v in data.items()}
    return data, carry

  def loss(self, data, carry, update=True):
    metrics = {}
    losses = {}
    newlat, dyn_outs, wm_outs = self.wm.observe(data, carry)
    wmlosses, mets = self.wm.loss(dyn_outs, wm_outs, data)
    losses.update(wmlosses)
    metrics.update(mets)
    replay_outs = dyn_outs

    hlwmlosses, mets = self.hlwm.loss(replay_outs, data)
    losses.update({f'hlwm_{k}': v for k, v in hlwmlosses.items()})
    metrics.update(mets)

    outs, acts, rew, con = self.wm.imagine(lambda out: cast(self.ac.policy(out)), data, replay_outs)

    aclosses, mets = self.ac.loss(data, outs, acts, con, rew, replay_outs, update)
    losses.update(aclosses)
    metrics.update(mets)

    # Metrics
    metrics.update({f'{k}_loss': v.mean() for k, v in losses.items()})
    metrics.update({f'{k}_loss_std': v.std() for k, v in losses.items()})
    metrics['data_rew/max'] = jnp.abs(data['reward']).max()
    metrics['pred_rew/max'] = jnp.abs(rew).max()
    metrics['data_rew/mean'] = data['reward'].mean()
    metrics['pred_rew/mean'] = rew.mean()
    metrics['data_rew/std'] = data['reward'].std()
    metrics['pred_rew/std'] = rew.std()

    # Combine
    losses = {k: v * self.scales[k] for k, v in losses.items()}
    loss = jnp.stack([v.mean() for k, v in losses.items()]).sum()
    newact = {k: data[k][:, -1] for k in self.act_space}
    outs = {'replay_outs': replay_outs}
    outs.update({f'{k}_loss': v for k, v in losses.items()})
    carry = (newlat, newact)
    return loss, (outs, carry, metrics)

  def report(self, data, carry):
    self.config.jax.jit and embodied.print(
        'Tracing report function', color='yellow')
    if not self.config.report:
      return {}, carry

    data = self.preprocess(data)
    data, carry = self._load_replay_context(data)
    metrics = self.wm.report(data, carry)
    hlwm_metrics = self.hlwm.report(data, carry)
    metrics.update(hlwm_metrics)

    # Grad norms per loss term
    if self.config.report_gradnorms:
      for key in self.scales:
        try:
          lossfn = lambda data, carry: self.loss(
              data, carry, update=False)[1][0][f'{key}_loss'].mean()
          grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
          metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
        except KeyError:
          print(f'Skipping gradnorm summary for missing loss: {key}')

    return metrics, carry

  def preprocess(self, obs):
    spaces = {**self.obs_space, **self.act_space, **self.aux_spaces}
    result = {}
    for key, value in obs.items():
      if key.startswith('log_') or key in ('reset', 'key', 'id'):
        continue
      space = spaces[key]
      if len(space.shape) >= 3 and space.dtype == jnp.uint8:
        value = cast(value) / 255.0
      result[key] = value
    result['cont'] = 1.0 - f32(result['is_terminal'])
    return result
