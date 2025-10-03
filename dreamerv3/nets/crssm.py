from typing import Literal

import einops
import jax
import numpy as np
import embodied
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from dreamerv3 import jaxutils
import dreamerv3.ninjax as nj
from dreamerv3.nets.base import Linear, BlockLinear
from dreamerv3.nets.gatel0rd import Gatel0rdCell, straight_through_heaviside

f32 = jnp.float32
tfd = tfp.distributions
sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute


class CRSSM(nj.Module):
  deter: int = 4096
  hidden: int = 2048
  stoch: int = 32
  classes: int = 32
  context: int = 32
  context_integration: str = 'everywhere'
  context_gate_noise_scale: float = 0.1
  norm: str = 'rms'
  act: str = 'gelu'
  unroll: bool = False
  unimix: float = 0.01
  outscale: float = 1.0
  imglayers: int = 2
  obslayers: int = 1
  coarse_layers: int = 1
  dynlayers: int = 1
  absolute: bool = False
  cell: str = 'gru'
  blocks: int = 8
  block_fans: bool = False
  block_norm: bool = False
  free: float = 1.0
  sparse_free: float = 0.0
  sparse_over_time_only: bool = False

  def __init__(self, **kw):
    self.kw = kw
    assert self.context_integration in ('none', 'posterior', 'everywhere') # ninjax complains about enums and Literals

  def initial(self, bsize):
    carry = dict(
        deter=jnp.zeros([bsize, self.deter], f32),
        stoch=jnp.zeros([bsize, self.stoch, self.classes], f32),
        context=jnp.zeros([bsize, self.context], f32))
    return cast(carry)

  def outs_to_carry(self, outs):
    keys = ('deter', 'stoch', 'context')
    return {k: outs[k][:, -1] for k in keys}

  def outs_to_coarse_carry(self, outs):
    keys = ('stoch', 'context')
    return {k: outs[k][:, -1] for k in keys}

  @property
  def spaces(self):
    spaces = {}
    latdtype = jaxutils.COMPUTE_DTYPE
    latdtype = np.float32 if latdtype == jnp.bfloat16 else latdtype
    spaces['deter'] = embodied.Space(latdtype, self.deter)
    spaces['stoch'] = embodied.Space(np.int32, self.stoch)
    spaces['context'] = embodied.Space(latdtype, self.context)
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
    deter, stoch, context, action = jaxutils.reset(
        (carry['deter'], carry['stoch'], carry['context'], action), reset)
    stoch_proj = self.get('stochproj', Linear, self.hidden, **kw)(stoch.reshape((stoch.shape[0], -1)))
    action /= sg(jnp.maximum(1, jnp.abs(action)))
    action_proj = self.get('actionproj', Linear, self.hidden, **kw)(action)
    coarse_feat, context, coarse_gates = self.coarse(context, stoch_proj, action_proj)
    deter, feat = self._gru(deter, stoch_proj, context, action_proj)
    if self.absolute:
      x = embed
    elif self.context_integration != 'none':
      x = jnp.concatenate([feat, context, embed], -1)
    else:
      x = jnp.concatenate([feat, embed], -1)

    for i in range(self.obslayers):
      x = self.get(f'obs{i}', Linear, self.hidden, **kw)(x)
    logit = self._logit('obslogit', x)
    stoch = cast(self._dist(logit).sample(seed=nj.seed()))
    coarse_logit = self._coarse_prior(coarse_feat)
    coarse_stoch = cast(self._dist(coarse_logit).sample(seed=nj.seed()))
    carry = dict(deter=deter, stoch=stoch, context=context)
    outs = dict(deter=deter, stoch=stoch, context=context, logit=logit, coarse_gates=coarse_gates,
                coarse_logit=coarse_logit, coarse_stoch=coarse_stoch)
    return cast(carry), cast(outs)

  def imagine(self, carry, action):
    kw = dict(**self.kw, norm=self.norm, act=self.act)
    if isinstance(action, dict):
      action = jaxutils.concat_dict(action)
    batch_dims = len(action.shape) - 1
    assert batch_dims in (1, 2)
    carry, action = cast((carry, action))
    if batch_dims == 2:
      return jaxutils.scan(
          lambda carry, action: self.imagine(carry, action),
          cast(carry), cast(action), self.unroll, axis=1)
    stoch = carry['stoch']
    stoch_proj = self.get('stochproj', Linear, self.hidden, **kw)(stoch.reshape((stoch.shape[0], -1)))
    action /= sg(jnp.maximum(1, jnp.abs(action)))
    action_proj = self.get('actionproj', Linear, self.hidden, **kw)(action)
    coarse_feat, context, coarse_gates = self.coarse(carry['context'], stoch_proj, action_proj)
    deter, feat = self._gru(carry['deter'], stoch_proj, context, action_proj)
    logit = self._prior(feat, context)
    stoch = cast(self._dist(logit).sample(seed=nj.seed()))
    coarse_logit = self._coarse_prior(coarse_feat)
    coarse_stoch = cast(self._dist(coarse_logit).sample(seed=nj.seed()))
    carry = dict(deter=deter, stoch=stoch, context=context)
    outs = dict(deter=deter, stoch=stoch, context=context, logit=logit, coarse_gates=coarse_gates,
                coarse_logit=coarse_logit, coarse_stoch=coarse_stoch)
    return cast(carry), cast(outs)

  def coarse_imagine(self, carry, action):
    kw = dict(**self.kw, norm=self.norm, act=self.act)
    if isinstance(action, dict):
      action = jaxutils.concat_dict(action)
    batch_dims = len(action.shape) - 1
    assert batch_dims in (1, 2)
    carry, action = cast((carry, action))
    if batch_dims == 2:
      return jaxutils.scan(
          lambda carry, action: self.coarse_imagine(carry, action),
          cast(carry), cast(action), self.unroll, axis=1)
    old_stoch = carry['stoch']
    stoch_proj = self.get('stochproj', Linear, self.hidden, **kw)(old_stoch.reshape((old_stoch.shape[0], -1)))
    action /= sg(jnp.maximum(1, jnp.abs(action)))
    action_proj = self.get('actionproj', Linear, self.hidden, **kw)(action)
    coarse_feat, context, coarse_gates = self.coarse(carry['context'], stoch_proj, action_proj)
    coarse_logit = self._coarse_prior(coarse_feat)
    coarse_stoch = cast(self._dist(coarse_logit).sample(seed=nj.seed()))
    carry = dict(coarse_stoch=coarse_stoch, context=context)
    outs = dict(context=context, coarse_gates=coarse_gates, coarse_logit=coarse_logit, coarse_stoch=coarse_stoch)
    return cast(carry), cast(outs)

  def loss(self, outs):
    metrics = {}
    prior = self._prior(outs.get('feat', outs['deter']), outs['context'])
    post = outs['logit']
    dyn = self._dist(sg(post)).kl_divergence(self._dist(prior))
    rep = self._dist(post).kl_divergence(self._dist(sg(prior)))
    if self.free:
      dyn = jnp.maximum(dyn, self.free)
      rep = jnp.maximum(rep, self.free)

    coarse_prior = outs['coarse_logit']
    coarse_dyn = self._dist(sg(post)).kl_divergence(self._dist(coarse_prior))
    coarse_rep = self._dist(post).kl_divergence(self._dist(sg(coarse_prior)))
    if self.free:
      coarse_dyn = jnp.maximum(coarse_dyn, self.free)
      coarse_rep = jnp.maximum(coarse_rep, self.free)

    if self.sparse_over_time_only:
      sparse = straight_through_heaviside(jnp.max(outs['coarse_gates'].astype(f32), axis=-1))
    else:
      sparse = jnp.mean(outs['coarse_gates'].astype(f32), axis=-1)
    # First sum so that sparse_free has semantics of "context changes allowed in episode", then divide by seq_len
    # to get reasonable loss scale
    sparse = jnp.sum(sparse, axis=-1)
    if self.sparse_free:
      sparse = jnp.maximum(sparse, self.sparse_free)
    sparse /= post.shape[1]
    metrics.update(jaxutils.tensorstats(
        self._dist(prior).entropy(), 'prior_ent'))
    metrics.update(jaxutils.tensorstats(
        self._dist(post).entropy(), 'post_ent'))
    metrics.update(jaxutils.tensorstats(
        self._dist(coarse_prior).entropy(), 'coarse_prior_ent'))
    return {'dyn': dyn, 'rep': rep, 'coarse_dyn': coarse_dyn, 'coarse_rep': coarse_rep, 'sparse': sparse}, metrics

  def coarse(self, context, stoch_proj, action_proj):
    x = jnp.concatenate([stoch_proj, action_proj], -1)
    out, context, gates = self.get('coarse', Gatel0rdCell, self.context_gate_noise_scale, self.sparse_over_time_only)(x, context, is_training=True)
    return out, context, gates

  def _prior(self, feat, context):
    kw = dict(**self.kw, norm=self.norm, act=self.act)
    x = jnp.concatenate([feat, context], -1) if self.context_integration == 'everywhere' else feat
    for i in range(self.imglayers):
      x = self.get(f'img{i}', Linear, self.hidden, **kw)(x)
    return self._logit('imglogit', x)

  def _coarse_prior(self, coarse_feat):
    kw = dict(**self.kw, norm=self.norm, act=self.act)
    x = coarse_feat
    for i in range(self.coarse_layers):
      x = self.get(f'coarse{i}', Linear, self.hidden, **kw)(x)
    return self._logit('coarselogit', x)

  def _gru(self, deter, stoch_proj, context, action_proj):
    kw = dict(**self.kw, norm=self.norm, act=self.act)
    assert self.cell == 'blockgru', 'Only blockgru is supported'
    g = self.blocks
    flat2group = lambda x: einops.rearrange(x, '... (g h) -> ... g h', g=g)
    group2flat = lambda x: einops.rearrange(x, '... g h -> ... (g h)', g=g)
    x0 = self.get('dynin0', Linear, self.hidden, **kw)(deter)
    if self.context_integration == 'everywhere':
      x3 = self.get('dynin3', Linear, self.hidden, **kw)(context)
      x = jnp.concatenate([x0, stoch_proj, action_proj, x3], -1)[..., None, :].repeat(g, -2)
    else:
      x = jnp.concatenate([x0, stoch_proj, action_proj], -1)[..., None, :].repeat(g, -2)
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
    return deter, deter

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