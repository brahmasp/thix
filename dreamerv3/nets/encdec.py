import einops
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

from dreamerv3 import jaxutils
import dreamerv3.ninjax as nj
from dreamerv3.nets.base import Linear, BlockLinear, Norm, Conv2D, Input, Dist

f32 = jnp.float32
tfd = tfp.distributions
sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute


class SimpleEncoder(nj.Module):

  depth: int = 128
  mults: tuple = (1, 2, 4, 2)
  layers: int = 5
  units: int = 1024
  symlog: bool = True
  norm: str = 'rms'
  act: str = 'gelu'
  kernel: int = 4
  outer: bool = False
  minres: int = 4

  def __init__(self, spaces, **kw):
    assert all(len(s.shape) <= 3 for s in spaces.values()), spaces
    self.spaces = spaces
    self.veckeys = [k for k, s in spaces.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in spaces.items() if len(s.shape) == 3]
    self.vecinp = Input(self.veckeys, featdims=1)
    self.imginp = Input(self.imgkeys, featdims=3)
    self.depths = tuple(self.depth * mult for mult in self.mults)
    self.kw = kw

  def __call__(self, data):
    kw = dict(**self.kw, norm=self.norm, act=self.act)
    outs = []
    batch_dims = len(data['is_first'].shape)
    shape = data['is_first'].shape
    data = {k: data[k] for k in self.spaces}
    data = jaxutils.onehot_dict(data, self.spaces)

    if self.veckeys:
      x = self.vecinp(data, batch_dims, f32)
      x = x.reshape((-1, *x.shape[batch_dims:]))
      x = jaxutils.symlog(x) if self.symlog else x
      x = jaxutils.cast_to_compute(x)
      for i in range(self.layers):
        x = self.get(f'mlp{i}', Linear, self.units, **kw)(x)
      outs.append(x)

    if self.imgkeys:
      x = self.imginp(data, batch_dims, jaxutils.COMPUTE_DTYPE) - 0.5
      x = x.reshape((-1, *x.shape[batch_dims:]))
      for i, depth in enumerate(self.depths):
        stride = 1 if self.outer and i == 0 else 2
        x = self.get(f'conv{i}', Conv2D, depth, self.kernel, stride, **kw)(x)
      assert x.shape[-3] == x.shape[-2] == self.minres, f'shape {x.shape} for minres {self.minres}'
      x = x.reshape((x.shape[0], -1))
      outs.append(x)

    x = jnp.concatenate(outs, -1)
    x = x.reshape((*shape, *x.shape[1:]))
    return x


class SimpleDecoder(nj.Module):

  inputs: tuple = ('deter', 'stoch')
  depth: int = 128
  mults: tuple = (1, 2, 4, 3)
  sigmoid: bool = True
  layers: int = 5
  units: int = 1024
  norm: str = 'rms'
  act: str = 'gelu'
  outscale: float = 1.0
  vecdist: str = 'symlog_mse'
  kernel: int = 4
  outer: bool = False
  block_fans: bool = False
  block_norm: bool = False
  block_space: int = 0
  hidden_stoch: bool = False
  space_hidden: int = 0
  minres: int = 4

  def __init__(self, spaces, **kw):
    assert all(len(s.shape) <= 3 for s in spaces.values()), spaces
    self.inp = Input(self.inputs, featdims=1)
    self.veckeys = [k for k, s in spaces.items() if len(s.shape) <= 2]
    self.imgkeys = [k for k, s in spaces.items() if len(s.shape) == 3]
    self.spaces = spaces
    self.depths = tuple([self.depth * mult for mult in self.mults])
    self.imgdep = sum(self.spaces[k].shape[-1] for k in self.imgkeys)
    self.kw = kw

  def __call__(self, lat):
    kw = dict(**self.kw, norm=self.norm, act=self.act)
    outs = {}
    batch_dims = len(lat['stoch'].shape) - 2

    if self.veckeys:
      inp = self.inp(lat, batch_dims, jaxutils.COMPUTE_DTYPE)
      x = inp.reshape((-1, inp.shape[-1]))
      for i in range(self.layers):
        x = self.get(f'mlp{i}', Linear, self.units, **kw)(x)
      x = x.reshape((*inp.shape[:batch_dims], *x.shape[1:]))
      for k in self.veckeys:
        dist = (
            dict(dist='softmax', bins=self.spaces[k].classes)
            if self.spaces[k].discrete else dict(dist=self.vecdist))
        k = k.replace('/', '_')
        outs[k] = self.get(f'out_{k}', Dist, self.spaces[k].shape, **dist)(x)

    if self.imgkeys:
      inp = self.inp(lat, batch_dims, jaxutils.COMPUTE_DTYPE)
      shape = (self.minres, self.minres, self.depths[-1])
      x = inp.reshape((-1, inp.shape[-1]))

      if self.space_hidden:
        x = self.get('space0', Linear, self.space_hidden * self.units, **kw)(x)
        x = self.get('space1', Linear, shape, **kw)(x)
      elif self.block_space:
        g = self.block_space
        x0 = einops.rearrange(cast(lat['deter']), 'b t ... -> (b t) ...')
        x1 = einops.rearrange(cast(lat['stoch']), 'b t l c -> (b t) (l c)')
        x0 = self.get(
            'space0', BlockLinear, int(np.prod(shape)), g, **self.kw,
            block_fans=self.block_fans, block_norm=self.block_norm)(x0)
        x0 = einops.rearrange(
            x0, '... (g h w c) -> ... h w (g c)',
            h=self.minres, w=self.minres, g=g)
        if self.hidden_stoch:
          x1 = self.get('space1hid', Linear, 2 * self.units, **kw)(x1)
        x1 = self.get('space1', Linear, shape, **self.kw)(x1)
        x = self.get('spacenorm', Norm, self.norm, act=self.act)(x0 + x1)
      else:
        x = self.get('space', Linear, shape, **kw)(x)

      for i, depth in reversed(list(enumerate(self.depths[:-1]))):
        x = self.get(
            f'conv{i}', Conv2D, depth, self.kernel, 2, **kw, transp=True)(x)
      outkw = dict(**self.kw, outscale=self.outscale, transp=True)
      stride = 1 if self.outer else 2
      x = self.get(
          'imgout', Conv2D, self.imgdep, self.kernel, stride, **outkw)(x)
      x = jax.nn.sigmoid(x) if self.sigmoid else x + 0.5
      x = x.reshape((*inp.shape[:batch_dims], *x.shape[1:]))
      split = np.cumsum([self.spaces[k].shape[-1] for k in self.imgkeys][:-1])
      for k, out in zip(self.imgkeys, jnp.split(x, split, -1)):
        outs[k] = jaxutils.MSEDist(f32(out), 3, 'sum')

    return outs