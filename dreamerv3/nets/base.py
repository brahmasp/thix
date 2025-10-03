import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

from dreamerv3 import jaxutils
import dreamerv3.ninjax as nj

f32 = jnp.float32
tfd = tfp.distributions
sg = lambda x: jax.tree_util.tree_map(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute


class MLP(nj.Module):

  layers: int = None
  units: int = None
  block_fans: bool = False
  block_norm: bool = False

  def __init__(self, shape, dist='mse', inputs=['tensor'], **kw):
    shape = (shape,) if isinstance(shape, (int, np.integer)) else shape
    assert isinstance(shape, (tuple, dict, type(None))), shape
    assert isinstance(dist, (str, dict)), dist
    assert isinstance(dist, dict) == isinstance(shape, dict), (dist, shape)
    self.shape = shape
    self.dist = dist
    self.inputs = Input(inputs, featdims=1)
    distonly = ('outscale', 'minstd', 'maxstd', 'unimix', 'bins')
    self.lkw = {k: v for k, v in kw.items() if k not in distonly}
    forbidden = ('binit', 'norm', 'act')
    self.dkw = {k: v for k, v in kw.items() if k not in forbidden}

  def __call__(self, inputs, bdims=2, training=False):
    feat = self.inputs(inputs, bdims, jaxutils.COMPUTE_DTYPE)
    x = feat.reshape([-1, feat.shape[-1]])
    for i in range(self.layers):
      x = self.get(f'h{i}', Linear, self.units, **self.lkw)(x)
    x = x.reshape((*feat.shape[:bdims], -1))
    if self.shape is None:
      return x
    elif isinstance(self.shape, dict):
      return {
          k: self._out(k, v, self.dist[k], x) for k, v in self.shape.items()}
    else:
      return self._out('dist', self.shape, self.dist, x)

  def _out(self, name, shape, dist, x):
    name = name.replace('/', '_').replace('.', '_')
    return self.get(name, Dist, shape, dist, **self.dkw)(x)


class Dist(nj.Module):

  outscale: float = 0.1
  minstd: float = 1.0
  maxstd: float = 1.0
  unimix: float = 0.0
  bins: int = 255

  def __init__(self, shape, dist='mse', **kw):
    assert all(isinstance(dim, (int, np.integer)) for dim in shape), shape
    forbidden = ('binit', 'norm', 'act')
    assert all(k not in kw for k in forbidden), (forbidden, kw)
    self.shape = shape
    self.dist = dist
    self.kw = dict(**kw, outscale=self.outscale)

  def __call__(self, inputs):
    dist = self.inner(inputs)
    assert tuple(dist.batch_shape) == tuple(inputs.shape[:-1]), (
        dist.batch_shape, dist.event_shape, inputs.shape)
    return dist

  def inner(self, inputs):
    shape = self.shape
    padding = 0

    if 'twohot' in self.dist or self.dist == 'softmax':
      padding = int(self.bins % 2)
      shape = (*self.shape, self.bins + padding)

    out = self.get('out', Linear, int(np.prod(shape)), **self.kw)(inputs)
    out = out.reshape(inputs.shape[:-1] + shape).astype(f32)
    out = out[..., :-padding] if padding else out

    if 'normal' in self.dist:
      units = int(np.prod(self.shape))
      std = self.get('std', Linear, units, **self.kw)(inputs)
      std = std.reshape(inputs.shape[:-1] + self.shape).astype(f32)

    if self.dist == 'symlog_mse':
      fwd, bwd = jaxutils.symlog, jaxutils.symexp
      return jaxutils.TransformedMseDist(out, len(self.shape), fwd, bwd)

    if self.dist == 'hyperbolic_mse':
      fwd = lambda x, eps=1e-3: (
          jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1) + eps * x)
      bwd = lambda x, eps=1e-3: jnp.sign(x) * (jnp.square(
          jnp.sqrt(1 + 4 * eps * (eps + 1 + jnp.abs(x))) / 2 / eps -
          1 / 2 / eps) - 1)
      return jaxutils.TransformedMseDist(out, len(self.shape), fwd, bwd)

    if self.dist == 'symlog_and_twohot':
      bins = np.linspace(-20, 20, out.shape[-1])
      return jaxutils.TwoHotDist(
          out, bins, len(self.shape), jaxutils.symlog, jaxutils.symexp)

    if self.dist == 'symexp_twohot':
      if out.shape[-1] % 2 == 1:
        half = jnp.linspace(-20, 0, (out.shape[-1] - 1) // 2 + 1, dtype=f32)
        half = jaxutils.symexp(half)
        bins = jnp.concatenate([half, -half[:-1][::-1]], 0)
      else:
        half = jnp.linspace(-20, 0, out.shape[-1] // 2, dtype=f32)
        half = jaxutils.symexp(half)
        bins = jnp.concatenate([half, -half[::-1]], 0)
      return jaxutils.TwoHotDist(out, bins, len(self.shape))

    if self.dist == 'hyperbolic_twohot':
      eps = 0.001
      f = lambda x: np.sign(x) * (np.square(np.sqrt(
          1 + 4 * eps * (eps + 1 + np.abs(x))) / 2 / eps - 1 / 2 / eps) - 1)
      bins = f(np.linspace(-300, 300, out.shape[-1]))
      return jaxutils.TwoHotDist(out, bins, len(self.shape))

    if self.dist == 'mse':
      return jaxutils.MSEDist(out, len(self.shape), 'sum')

    if self.dist == 'huber':
      return jaxutils.HuberDist(out, len(self.shape), 'sum')

    if self.dist == 'normal':
      lo, hi = self.minstd, self.maxstd
      std = (hi - lo) * jax.nn.sigmoid(std + 2.0) + lo
      dist = tfd.Normal(jnp.tanh(out), std)
      dist = tfd.Independent(dist, len(self.shape))
      dist.minent = np.prod(self.shape) * tfd.Normal(0.0, lo).entropy()
      dist.maxent = np.prod(self.shape) * tfd.Normal(0.0, hi).entropy()
      return dist

    if self.dist == 'trunc_normal':
      lo, hi = self.minstd, self.maxstd
      std = (hi - lo) * jax.nn.sigmoid(std + 2.0) + lo
      dist = tfd.TruncatedNormal(jnp.tanh(out), std, -1, 1)
      dist = tfd.Independent(dist, len(self.shape))
      dist.minent = np.prod(self.shape) * (
          tfd.TruncatedNormal(1.0, lo, -1, 1).entropy())
      dist.maxent = np.prod(self.shape) * (
          tfd.TruncatedNormal(0.0, hi, -1, 1).entropy())
      return dist

    if self.dist == 'binary':
      dist = tfd.Bernoulli(out)
      if self.shape:
        dist = tfd.Independent(dist, len(self.shape))
      return dist

    if self.dist == 'softmax':
      dist = tfd.Categorical(out)
      if len(self.shape) > 1:
        dist = tfd.Independent(dist, len(self.shape) - 1)
      return dist

    if self.dist == 'onehot':
      if self.unimix:
        probs = jax.nn.softmax(out, -1)
        uniform = jnp.ones_like(probs) / probs.shape[-1]
        probs = (1 - self.unimix) * probs + self.unimix * uniform
        out = jnp.log(probs)
      dist = jaxutils.OneHotDist(out)
      if len(self.shape) > 1:
        dist = tfd.Independent(dist, len(self.shape) - 1)
      dist.minent = 0.0
      dist.maxent = np.prod(self.shape[:-1]) * np.log(self.shape[-1])
      return dist

    raise NotImplementedError(self.dist)


class Conv2D(nj.Module):

  groups: int = 1
  transp: bool = False
  act: str = 'none'
  norm: str = 'none'
  pad: str = 'same'
  bias: bool = True
  outscale: float = 1.0
  winit: str = 'normal'
  binit: bool = False
  fan: str = 'in'
  dtype: str = 'default'

  def __init__(self, depth, kernel, stride=1):
    self.depth = depth
    self.kernel = kernel
    self.stride = stride
    self._winit = Initializer(self.winit, self.outscale, self.fan, self.dtype)
    self._binit = Initializer('zeros', 1.0, self.fan, self.dtype)
    self._norm = Norm(self.norm, name='norm')

  def __call__(self, x):
    assert x.dtype == jaxutils.COMPUTE_DTYPE, (x.dtype, x.shape)
    x = self._layer(x)
    x = self._norm(x)
    x = get_act(self.act)(x)
    return x

  def _layer(self, x):
    if self.transp:
      assert self.groups == 1, self.groups
      shape = (self.kernel, self.kernel, x.shape[-1], self.depth)
      kernel = self.get('kernel', self._winit, shape)
      kernel = jaxutils.cast_to_compute(kernel)
      flops = int(np.prod(shape)) * x.shape[-3] * x.shape[-2]
      x = jax.lax.conv_transpose(
          x, kernel, (self.stride, self.stride), self.pad.upper(),
          dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    else:
      G = self.groups
      shape = (self.kernel, self.kernel, x.shape[-1] // G, self.depth)
      kernel = self.get('kernel', self._winit, shape)
      kernel = jaxutils.cast_to_compute(kernel)
      x = jax.lax.conv_general_dilated(
          x, kernel, (self.stride, self.stride), self.pad.upper(),
          feature_group_count=self.groups,
          dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
      flops = int(np.prod(shape)) * x.shape[-3] * x.shape[-2]
    if self.bias:
      if self.binit:
        args = (self._winit, self.depth, shape)
      else:
        args = (self._binit, self.depth)
      x += self.get('bias', *args).astype(x.dtype)
      flops += int(np.prod(x.shape[-3:]))
    assert x.dtype == jaxutils.COMPUTE_DTYPE, (x.dtype, x.shape)
    return x


class Linear(nj.Module):

  act: str = 'none'
  norm: str = 'none'
  bias: bool = True
  outscale: float = 1.0
  winit: str = 'normal'
  binit: bool = False
  fan: str = 'in'
  dtype: str = 'default'
  fanin: int = 0

  def __init__(self, units):
    self.units = (units,) if isinstance(units, int) else tuple(units)
    self._winit = Initializer(
        self.winit, self.outscale, self.fan, self.dtype)
    self._binit = Initializer('zeros', 1.0, self.fan, self.dtype)
    self._norm = Norm(self.norm, name='norm')

  def __call__(self, x):
    assert x.dtype == jaxutils.COMPUTE_DTYPE, (x.dtype, x.shape)
    x = self._layer(x)
    x = self._norm(x)
    x = get_act(self.act)(x)
    return x

  def _layer(self, x):
    shape = (x.shape[-1], int(np.prod(self.units)))
    fan_shape = (self.fanin, shape[1]) if self.fanin else None
    x = x @ self.get('kernel', self._winit, shape, fan_shape).astype(x.dtype)
    flops = int(np.prod(shape))
    if self.bias:
      if self.binit:
        args = (self._winit, np.prod(self.units), shape)
      else:
        args = (self._binit, np.prod(self.units))
      x += self.get('bias', *args).astype(x.dtype)
      flops += int(np.prod(self.units))
    assert x.dtype == jaxutils.COMPUTE_DTYPE, (x.dtype, x.shape)
    if len(self.units) > 1:
      x = x.reshape(x.shape[:-1] + self.units)
    return x


class BlockLinear(nj.Module):

  act: str = 'none'
  norm: str = 'none'
  bias: bool = True
  outscale: float = 1.0
  winit: str = 'normal'
  binit: bool = False
  fan: str = 'in'
  dtype: str = 'default'
  block_fans: bool = False
  block_norm: bool = False

  def __init__(self, units, groups):
    self.units = (units,) if isinstance(units, int) else tuple(units)
    assert groups <= np.prod(units), (groups, units)
    self.groups = groups
    self._winit = Initializer(
        self.winit, self.outscale, self.fan, self.dtype,
        block_fans=self.block_fans)
    self._binit = Initializer('zeros', 1.0, self.fan, self.dtype)
    if self.block_norm:
      self._norm = [
          Norm(self.norm, name=f'norm{i}') for i in range(self.groups)]
    else:
      self._norm = Norm(self.norm, name='norm')

  def __call__(self, x):
    assert x.dtype == jaxutils.COMPUTE_DTYPE, (x.dtype, x.shape)
    x = self._layer(x)
    if self.block_norm and self._norm != 'none':
      x = jnp.concatenate([
          f(y) for f, y in zip(self._norm, jnp.split(x, self.groups, -1))], -1)
    else:
      x = self._norm(x)
    x = get_act(self.act)(x)
    return x

  def _layer(self, x):
    bdims, indim, outdim = x.shape[:-1], x.shape[-1], np.prod(self.units)
    if indim % self.groups != 0:
      pad = int(np.ceil(indim / self.groups)) * self.groups - indim
      x = jnp.concatenate([x, jnp.zeros((*x.shape[:-1], pad), x.dtype)], -1)
      indim = x.shape[-1]
    assert indim % self.groups == outdim % self.groups == 0, (
        indim, outdim, self.groups, self.units)
    shape = (self.groups, indim // self.groups, outdim // self.groups)
    kernel = self.get('kernel', self._winit, shape, shape).astype(x.dtype)
    flops = int(np.prod(shape))
    x = x.reshape((*bdims, self.groups, indim // self.groups))
    x = jnp.einsum('...ki,kio->...ko', x, kernel)
    x = x.reshape((*bdims, outdim))
    if self.bias:
      if self.binit:
        args = (self._winit, np.prod(self.units), shape)
      else:
        args = (self._binit, np.prod(self.units))
      bias = self.get('bias', *args)
      x += bias.astype(x.dtype)
      flops += int(np.prod(self.units))
    if len(self.units) > 1:
      x = x.reshape(x.shape[:-1] + self.units)
    assert x.dtype == jaxutils.COMPUTE_DTYPE, (x.dtype, x.shape)
    return x


class Embed(nj.Module):

  outscale: float = 1.0
  winit: str = 'normal'
  fan: str = 'in'
  dtype: str = 'default'

  def __init__(self, count, units):
    self.count = count
    self.units = units
    self._winit = Initializer(self.winit, self.outscale, self.fan, self.dtype)

  def __call__(self, x):
    assert x.dtype in (jnp.uint32, jnp.int32), x.dtype
    shape = (self.count, self.units)
    fan_shape = (1, self.units)
    w = self.get('embed', self._winit, shape, fan_shape).astype(x.dtype)
    return jnp.take(w, x, axis=0)


class Norm(nj.Module):

  act: str = 'none'

  def __init__(self, impl, eps=1e-4):
    if '1em' in impl:
      impl, exponent = impl.split('1em')
      eps = 10 ** -int(exponent)
    self._impl = impl
    self._eps = eps

  def __call__(self, x):
    x = self._norm(x)
    x = get_act(self.act)(x)
    return x

  def _norm(self, x):
    if self._impl == 'none':
      return x
    elif self._impl == 'layer':
      x = x.astype(f32)
      mean = x.mean(-1)[..., None]
      mean2 = jnp.square(x).mean(-1)[..., None]
      var = jnp.maximum(0, mean2 - jnp.square(mean))
      scale = self.get('scale', jnp.ones, x.shape[-1], f32)
      offset = self.get('offset', jnp.zeros, x.shape[-1], f32)
      mult = scale * jax.lax.rsqrt(var + self._eps)
      x = (x - mean) * mult + offset
      return cast(x)
    elif self._impl == 'rms':
      dtype = x.dtype
      x = f32(x) if x.dtype == jnp.float16 else x
      scale = self.get('scale', jnp.ones, x.shape[-1], f32).astype(x.dtype)
      mult = jax.lax.rsqrt((x * x).mean(-1)[..., None] + self._eps) * scale
      return (x * mult).astype(dtype)
    elif self._impl == 'rms_instance':
      x = x.astype(f32)
      scale = self.get('scale', jnp.ones, x.shape[-1], f32)
      mult = jax.lax.rsqrt((x * x).mean((-3, -2), keepdims=True) + self._eps)
      mult = mult * scale
      return cast(x * mult)
    elif self._impl == 'grn':
      assert len(x.shape) >= 4, x.shape
      x = x.astype(f32)
      norm = jnp.linalg.norm(x, 2, (-3, -2), keepdims=True)
      norm /= (norm.mean(-1, keepdims=True) + self._eps)
      scale = self.get('scale', jnp.ones, x.shape[-1], f32)
      offset = self.get('offset', jnp.zeros, x.shape[-1], f32)
      x = (norm * scale + 1) * x + offset
      return cast(x)
    elif self._impl == 'instance':
      x = x.astype(f32)
      mean = x.mean(axis=(-3, -2), keepdims=True)
      var = x.var(axis=(-3, -2), keepdims=True)
      scale = self.get('scale', jnp.ones, x.shape[-1], f32)
      offset = self.get('offset', jnp.zeros, x.shape[-1], f32)
      x = (scale * jax.lax.rsqrt(var + self._eps)) * (x - mean) + offset
      return cast(x)
    else:
      raise NotImplementedError(self._impl)


class Input:

  def __init__(self, keys=['tensor'], featdims=1):
    self.keys = tuple(keys)
    self.featdims = featdims

  def __call__(self, inputs, bdims=2, dtype=None):
    if not isinstance(inputs, dict):
      inputs = {'tensor': inputs}
    try:
      xs = []
      for key in self.keys:
        x = inputs[key]
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
          x = jnp.concatenate([x.real, x.imag], -1)
        x = x.astype(dtype or inputs[self.keys[0]].dtype)
        x = x.reshape((*x.shape[:bdims + self.featdims - 1], -1))
        msg = f'Invalid input ({nj.SCOPE}, {key}, {x.shape}, {x.dtype}): {{x}}'
        jaxutils.check(jnp.isfinite(x).all(), msg, x=x)
        xs.append(x)
      xs = jnp.concatenate(xs, -1)
    except (KeyError, ValueError, TypeError) as e:
      shapes = {k: v.shape for k, v in inputs.items()}
      raise ValueError(
          f'Error: {e}\n'
          f'Input shapes: {shapes}\n' +
          f'Requested keys: {self.keys}')
    return xs


class Initializer:

  VARIANCE_FACTOR = 1.0

  def __init__(
      self, dist='normal', scale=1.0, fan='in', dtype='default',
      block_fans=False):
    self.dist = dist
    self.scale = scale
    self.fan = fan
    self.dtype = dtype
    self.block_fans = block_fans

  def __call__(self, shape, fan_shape=None):
    shape = (shape,) if isinstance(shape, (int, np.integer)) else tuple(shape)
    assert all(x > 0 for x in shape), shape
    dtype = jaxutils.PARAM_DTYPE if self.dtype == 'default' else self.dtype
    dtype = getattr(jnp, dtype) if isinstance(dtype, str) else dtype
    fanin, fanout = self._fans(fan_shape or shape)
    fan = {'avg': (fanin + fanout) / 2, 'in': fanin, 'out': fanout}[self.fan]
    if self.dist == 'zeros':
      value = jnp.zeros(shape, dtype)
    elif self.dist == 'uniform':
      limit = np.sqrt(self.VARIANCE_FACTOR / fan)
      value = jax.random.uniform(nj.seed(), shape, dtype, -limit, limit)
    elif self.dist == 'normal':
      value = jax.random.truncated_normal(nj.seed(), -2, 2, shape)
      value *= 1.1368 * np.sqrt(self.VARIANCE_FACTOR / fan)
      value = value.astype(dtype)
    elif self.dist == 'normed':
      value = jax.random.uniform(nj.seed(), shape, dtype, -1, 1)
      value /= jnp.linalg.norm(value.reshape((-1, shape[-1])), 2, 0)
    elif self.dist == 'complex':
      assert jnp.issubdtype(dtype, jnp.complexfloating), dtype
      realdt = jnp.finfo(dtype).dtype
      value = jax.random.truncated_normal(
          nj.seed(), -2, 2, (2, *shape), realdt)
      value = value[0] + 1j * value[1]
      value *= jax.lax.convert_element_type(1.137 * np.sqrt(1 / fan), realdt)
    elif self.dist == 'ortho':
      nrows, ncols = shape[-1], np.prod(shape) // shape[-1]
      matshape = (nrows, ncols) if nrows > ncols else (ncols, nrows)
      mat = jax.random.normal(nj.seed(), matshape, dtype)
      qmat, rmat = jnp.linalg.qr(mat)
      qmat *= jnp.sign(jnp.diag(rmat))
      qmat = qmat.T if nrows < ncols else qmat
      qmat = qmat.reshape(nrows, *shape[:-1])
      value = jnp.moveaxis(qmat, 0, -1)
    else:
      raise NotImplementedError(self.dist)
    value *= self.scale
    return value

  def _fans(self, shape):
    if len(shape) == 0:
      return (1, 1)
    elif len(shape) == 1:
      return (1, shape[0])
    elif len(shape) == 2:
      return shape
    elif len(shape) == 3 and self.block_fans:
      return shape[1:]
    else:
      space = int(np.prod(shape[:-2]))
      return (shape[-2] * space, shape[-1] * space)


def get_act(name):
  if callable(name):
    return name
  elif name == 'none':
    return lambda x: x
  elif name == 'cswiglu':
    def fn(x):
      x, y = jnp.split(x, 2, -1)
      y1, y2 = jnp.split(y, 2, -1)
      pad = jnp.ones_like(y1)
      x = jax.nn.swish(jnp.concatenate([x, -x], -1))
      y = jnp.concatenate([y1, pad, y2, pad], -1)
      return x * y
    return fn
  elif name == 'mish':
    return lambda x: x * jnp.tanh(jax.nn.softplus(x))
  elif hasattr(jax.nn, name):
    return getattr(jax.nn, name)
  else:
    raise NotImplementedError(name)
