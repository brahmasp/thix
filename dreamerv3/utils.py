import jax.numpy as jnp
import jax

from . import jaxutils
from . import ninjax as nj

f32 = jnp.float32
treemap = jax.tree_util.tree_map
sg = lambda x: treemap(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute
sample = lambda dist: {
    k: v.sample(seed=nj.seed()) for k, v in dist.items()}

def imagine(step_function, start_data, horizon, unroll, start_out):
    _, img_outs = jaxutils.scan(
        step_function, sg(start_data),
        jnp.arange(horizon), unroll)
    img_outs = treemap(lambda x: x.swapaxes(0, 1), img_outs)
    img_outs = treemap(
        lambda first, seq: jnp.concatenate([first, seq], 1),
        treemap(lambda x: x[:, None], start_out), img_outs)
    return img_outs

def is_goal_context_reached(context, goal, tolerance):
  return jnp.sum(jnp.square(context - goal), axis=-1) <= tolerance