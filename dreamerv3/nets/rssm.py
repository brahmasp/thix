import einops
import jax
import numpy as np
import embodied
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from dreamerv3 import jaxutils
import dreamerv3.ninjax as nj
from dreamerv3.nets.base import Linear, BlockLinear, Norm, get_act

f32 = jnp.float32
tfd = tfp.distributions
sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute


class RSSM(nj.Module):

  deter: int = 4096
  hidden: int = 2048
  stoch: int = 32
  classes: int = 32
  norm: str = 'rms'
  act: str = 'gelu'
  unroll: bool = False
  unimix: float = 0.01
  outscale: float = 1.0
  imglayers: int = 2
  obslayers: int = 1
  dynlayers: int = 1
  absolute: bool = False
  cell: str = 'gru'
  blocks: int = 8
  block_fans: bool = False
  block_norm: bool = False
  free: float = 1.0

  def __init__(self, **kw):
    self.kw = kw

  def initial(self, bsize):
    carry = dict(
        deter=jnp.zeros([bsize, self.deter], f32),
        stoch=jnp.zeros([bsize, self.stoch, self.classes], f32))
    if self.cell == 'stack':
      carry['feat'] = jnp.zeros([bsize, self.hidden], f32)
    return cast(carry)

  def outs_to_carry(self, outs):
    keys = ('deter', 'stoch')
    if self.cell == 'stack':
      keys += ('feat',)
    return {k: outs[k][:, -1] for k in keys}

  @property
  def spaces(self):
    spaces = {}
    latdtype = jaxutils.COMPUTE_DTYPE
    latdtype = np.float32 if latdtype == jnp.bfloat16 else latdtype
    spaces['deter'] = embodied.Space(latdtype, self.deter)
    spaces['stoch'] = embodied.Space(np.int32, self.stoch)
    return spaces

  def observe(self, carry, action, embed, reset):
    kw = dict(**self.kw, norm=self.norm, act=self.act)
    if isinstance(action, dict):
      action = jaxutils.concat_dict(action)
    batch_dims = len(action.shape) - 1
    assert batch_dims in (1, 2)
    carry, action, embed = cast((carry, action, embed))
    if batch_dims == 2:
      return jaxutils.scan(
          lambda carry, inputs: self.observe(carry, *inputs),
          carry, (action, embed, reset), self.unroll, axis=1)
    deter, stoch, action = jaxutils.reset(
        (carry['deter'], carry['stoch'], action), reset)
    deter, feat = self._gru(deter, stoch, action)
    x = embed if self.absolute else jnp.concatenate([feat, embed], -1)
    for i in range(self.obslayers):
      x = self.get(f'obs{i}', Linear, self.hidden, **kw)(x)
    logit = self._logit('obslogit', x)
    stoch = cast(self._dist(logit).sample(seed=nj.seed()))
    carry = dict(deter=deter, stoch=stoch)
    outs = dict(deter=deter, stoch=stoch, logit=logit)
    if self.cell == 'stack':
      carry['feat'] = feat
      outs['feat'] = feat
    return cast(carry), cast(outs)

  def imagine(self, carry, action):
    if isinstance(action, dict):
      action = jaxutils.concat_dict(action)
    batch_dims = len(action.shape) - 1
    assert batch_dims in (1, 2)
    carry, action = cast((carry, action))
    if batch_dims == 2:
      return jaxutils.scan(
          lambda carry, action: self.imagine(carry, action),
          cast(carry), cast(action), self.unroll, axis=1)
    deter, feat = self._gru(carry['deter'], carry['stoch'], action)
    logit = self._prior(feat)
    stoch = cast(self._dist(logit).sample(seed=nj.seed()))
    carry = dict(deter=deter, stoch=stoch)
    outs = dict(deter=deter, stoch=stoch, logit=logit)
    if self.cell == 'stack':
      carry['feat'] = feat
      outs['feat'] = feat
    return cast(carry), cast(outs)

  def loss(self, outs):
    metrics = {}
    prior = self._prior(outs.get('feat', outs['deter']))
    post = outs['logit']
    dyn = self._dist(sg(post)).kl_divergence(self._dist(prior))
    rep = self._dist(post).kl_divergence(self._dist(sg(prior)))
    if self.free:
      dyn = jnp.maximum(dyn, self.free)
      rep = jnp.maximum(rep, self.free)
    metrics.update(jaxutils.tensorstats(
        self._dist(prior).entropy(), 'prior_ent'))
    metrics.update(jaxutils.tensorstats(
        self._dist(post).entropy(), 'post_ent'))
    return {'dyn': dyn, 'rep': rep}, metrics

  def _prior(self, feat):
    kw = dict(**self.kw, norm=self.norm, act=self.act)
    x = feat
    for i in range(self.imglayers):
      x = self.get(f'img{i}', Linear, self.hidden, **kw)(x)
    return self._logit('imglogit', x)

  def _gru(self, deter, stoch, action):
    kw = dict(**self.kw, norm=self.norm, act=self.act)
    inkw = {**self.kw, 'norm': self.norm, 'binit': False}
    stoch = stoch.reshape((stoch.shape[0], -1))
    action /= sg(jnp.maximum(1, jnp.abs(action)))
    if self.cell == 'gru':
      x0 = self.get('dynnorm', Norm, self.norm)(deter)
      x1 = self.get('dynin1', Linear, self.hidden, **inkw)(stoch)
      x2 = self.get('dynin2', Linear, self.hidden, **inkw)(action)
      x = jnp.concatenate([x0, x1, x2], -1)
      for i in range(self.dynlayers):
        x = self.get(f'dyn{i}', Linear, self.hidden, **kw)(x)
      x = self.get('dyncore', Linear, 3 * self.deter, **self.kw)(x)
      reset, cand, update = jnp.split(x, 3, -1)
      reset = jax.nn.sigmoid(reset)
      cand = jnp.tanh(reset * cand)
      update = jax.nn.sigmoid(update - 1)
      deter = update * cand + (1 - update) * deter
      out = deter
    elif self.cell == 'mgu':
      x0 = self.get('dynnorm', Norm, self.norm)(deter)
      x1 = self.get('dynin1', Linear, self.hidden, **inkw)(stoch)
      x2 = self.get('dynin2', Linear, self.hidden, **inkw)(action)
      x = jnp.concatenate([x0, x1, x2], -1)
      for i in range(self.dynlayers):
        x = self.get(f'dyn{i}', Linear, self.hidden, **kw)(x)
      x = self.get('dyncore', Linear, 2 * self.deter, **self.kw)(x)
      cand, update = jnp.split(x, 2, -1)
      update = jax.nn.sigmoid(update - 1)
      cand = jnp.tanh((1 - update) * cand)
      deter = update * cand + (1 - update) * deter
      out = deter
    elif self.cell == 'blockgru':
      g = self.blocks
      flat2group = lambda x: einops.rearrange(x, '... (g h) -> ... g h', g=g)
      group2flat = lambda x: einops.rearrange(x, '... g h -> ... (g h)', g=g)
      x0 = self.get('dynin0', Linear, self.hidden, **kw)(deter)
      x1 = self.get('dynin1', Linear, self.hidden, **kw)(stoch)
      x2 = self.get('dynin2', Linear, self.hidden, **kw)(action)
      x = jnp.concatenate([x0, x1, x2], -1)[..., None, :].repeat(g, -2)
      x = group2flat(jnp.concatenate([flat2group(deter), x], -1))
      for i in range(self.dynlayers):
        x = self.get(
            f'dyn{i}', BlockLinear, self.deter, g, **kw,
            block_norm=self.block_norm, block_fans=self.block_fans)(x)
      x = self.get(
          'dyncore', BlockLinear, 3 * self.deter, g, **self.kw,
          block_fans=self.block_fans)(x)
      gates = jnp.split(flat2group(x), 3, -1)
      reset, cand, update = [group2flat(x) for x in gates]
      reset = jax.nn.sigmoid(reset)
      cand = jnp.tanh(reset * cand)
      update = jax.nn.sigmoid(update - 1)
      deter = update * cand + (1 - update) * deter
      out = deter
    elif self.cell == 'stack':
      result = []
      deters = jnp.split(deter, self.dynlayers, -1)
      x = jnp.concatenate([stoch, action], -1)
      x = self.get('in', Linear, self.hidden, **kw)(x)
      for i in range(self.dynlayers):
        skip = x
        x = get_act(self.act)(jnp.concatenate([
            self.get(f'dyngru{i}norm1', Norm, self.norm)(deters[i]),
            self.get(f'dyngru{i}norm2', Norm, self.norm)(x)], -1))
        x = self.get(
            f'dyngru{i}core', Linear, 3 * deters[i].shape[-1], **self.kw)(x)
        reset, cand, update = jnp.split(x, 3, -1)
        reset = jax.nn.sigmoid(reset)
        cand = jnp.tanh(reset * cand)
        update = jax.nn.sigmoid(update - 1)
        deter = update * cand + (1 - update) * deters[i]
        result.append(deter)
        x = self.get(f'dyngru{i}proj', Linear, self.hidden, **self.kw)(x)
        x += skip
        skip = x
        x = self.get(f'dynmlp{i}norm', Norm, self.norm)(x)
        x = self.get(
            f'dynmlp{i}up', Linear, deters[i].shape[-1], **self.kw)(x)
        x = get_act(self.act)(x)
        x = self.get(f'dynmlp{i}down', Linear, self.hidden, **self.kw)(x)
        x += skip
      out = self.get('outnorm', Norm, self.norm)(x)
      deter = jnp.concatenate(result, -1)
    else:
      raise NotImplementedError(self.cell)
    return deter, out

  def _logit(self, name, x):
    kw = dict(**self.kw, outscale=self.outscale)
    kw['binit'] = False
    x = self.get(name, Linear, self.stoch * self.classes, **kw)(x)
    logit = x.reshape(x.shape[:-1] + (self.stoch, self.classes))
    if self.unimix:
      probs = jax.nn.softmax(logit, -1)
      uniform = jnp.ones_like(probs) / probs.shape[-1]
      probs = (1 - self.unimix) * probs + self.unimix * uniform
      logit = jnp.log(probs)
    return logit

  def _dist(self, logit):
    return tfd.Independent(jaxutils.OneHotDist(logit.astype(f32)), 1)