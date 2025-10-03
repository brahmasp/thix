import jax
import jax.nn as jnn
import jax.numpy as jnp
import dreamerv3.ninjax as nj
from dreamerv3.nets.base import Linear


def straight_through_heaviside(x):
  zero = x - jax.lax.stop_gradient(x)
  return zero + jax.lax.stop_gradient(jnp.where(x > 0, 1, 0))


def retanh(x):
  return jnp.maximum(0, jnp.tanh(x))


class Gatel0rdCell(nj.Module):
  def __init__(self, gate_noise_scale, sparse_over_time_only):
    self._gate_noise_scale = gate_noise_scale
    self._sparse_over_time_only = sparse_over_time_only
    self._kw = {'winit': 'normal', 'fan': 'avg', 'act': 'none', 'bias': True, 'norm': 'none'}

  def __call__(self, x, hidden, is_training=True):
    kw = {**self._kw, 'units': 2 * hidden.shape[-1]}#/ 'outscale': 0.0}
    gu = self.get('update', Linear, **kw)(jnp.concatenate([x, hidden], -1))
    gate, update = jnp.split(gu, 2, -1)
    update = jnp.tanh(update)
    gate_noise_scale = self._gate_noise_scale if is_training else 0.0
    gate += jax.random.normal(nj.seed(), gate.shape, dtype=gate.dtype) * gate_noise_scale
    gate = retanh(gate)
    h_new = hidden + gate * (update - hidden)

    kw = {**self._kw, 'units': 2 * hidden.shape[-1]}
    x = self.get('out', Linear, **kw)(jnp.concatenate([x, h_new], -1))
    output, output_gate = jnp.split(x, 2, -1)
    output = jax.nn.sigmoid(output)
    output_gate = jnp.tanh(output_gate)

    x = output * output_gate
    if self._sparse_over_time_only:
      return x, h_new, gate
    else:
      return x, h_new, straight_through_heaviside(gate)


class Timel0rdCell(nj.Module):
  def __init__(self, gate_noise_scale):
    self._gate_noise_scale = gate_noise_scale
    self._kw = {'winit': 'normal', 'fan': 'avg', 'act': 'none', 'bias': True, 'norm': 'none'}

  def __call__(self, x, hidden, is_training=True):
    kw = {**self._kw, 'units': 2 * hidden.shape[-1] + 1}#/ 'outscale': 0.0}
    gu = self.get('update', Linear, **kw)(jnp.concatenate([x, hidden], -1))
    scale, update, gate = jnp.split(gu, [hidden.shape[-1], 2 * hidden.shape[-1]], -1)
    update = jnp.tanh(update)
    gate_noise_scale = self._gate_noise_scale if is_training else 0.0
    gate += jax.random.normal(nj.seed(), gate.shape, dtype=gate.dtype) * gate_noise_scale
    gate = straight_through_heaviside(jnn.silu(gate))
    scale = jax.nn.sigmoid(scale)
    h_new = hidden + gate * scale * (update - hidden)

    kw = {**self._kw, 'units': 2 * hidden.shape[-1]}
    x = self.get('out', Linear, **kw)(jnp.concatenate([x, h_new], -1))
    output, output_gate = jnp.split(x, 2, -1)
    output = jax.nn.sigmoid(output)
    output_gate = jnp.tanh(output_gate)

    x = output * output_gate
    return x, h_new, gate