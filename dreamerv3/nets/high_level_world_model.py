import jax
import jax.numpy as jnp
import jax.nn as jnn
from tensorflow_probability.substrates import jax as tfp

from dreamerv3 import jaxutils
import dreamerv3.nets.base as basenets
from dreamerv3 import ninjax as nj
from dreamerv3.utils import imagine

f32 = jnp.float32
treemap = jax.tree_util.tree_map
tfd = tfp.distributions
sg = lambda x: treemap(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute
sample = lambda dist: {
    k: v.sample(seed=nj.seed()) for k, v in dist.items()}

def generate_high_level_world_model_training_targets(data, act_space, horizon):
    """Generates targets corresponding to the states BEFORE a context change occurs e.g the inputs
    that caused the context change, not the state after"""
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    context, logit, is_first, is_terminal, reward = (
        treemap(swap, (data['context'], data['logit'], data['is_first'], data['is_terminal'], data['reward'])))
    action = {k: swap(data[k]) for k in act_space.keys()}

    change_indices = (jnp.diff(context, append=jnp.array([context[-1]]), axis=0) != 0)
    change_indices = jnp.any(change_indices, axis=-1)

    # roll is_terminal by -1, but make it so the last value is 0
    next_is_terminal = jnp.roll(is_terminal, shift=-1, axis=0)
    next_is_terminal = jnp.concatenate((next_is_terminal[:-1], jnp.zeros_like(jnp.array([next_is_terminal[-1]]))), axis=0)

    # The way change is calculated and since next_is_terminal is rolled, this indicates NEXT state either has change or is terminal
    next_is_change_or_term = jnp.logical_or(change_indices, next_is_terminal)
    next_is_first = jnp.roll(is_first, shift=-1, axis=0)

    seq_len = is_first.shape[0]

    def action_and_logit_scan_fn(carry, x):
        next_first, next_change_or_term, inp = x
        # 0 instead of nan because of gradient issues, will be masked out later
        carry = jnp.where(next_first, 0, jnp.where(next_change_or_term, inp, carry))
        return carry, carry

    def time_scan_fn(carry, x):
        next_first, next_change_or_term = x
        # Use seq_len -1 as pseudo-nan, for easy identification and masking, NaN cannot be used due to gradient issues
        carry = jnp.where(next_first, -seq_len - 1, jnp.where(next_change_or_term, 0, carry + 1))
        return carry, carry


    discount = 1 - 1 / horizon
    # Note how the structure of this scan_fn is a bit different from the others, the inputs are not "next" but the current
    # E.g the reward for a high-level action INCLUDE the reward from the entering the goal-state
    # Basically, it predicts the Q value Q(s,a) up to the next context change instead of episode end
    def reward_scan_fn(carry, x):
        first, change_or_last, inp = x
        new_carry = jnp.where(first, 0, inp + jnp.where(change_or_last, 0, discount * carry))
        return new_carry, carry

    targets = {}

    for key in act_space.keys():
        action_inputs = (
            jaxutils.broadcast_to_match(next_is_first, action[key]), jaxutils.broadcast_to_match(next_is_change_or_term, action[key]), action[key]
        )
        _, targets[key] = jax.lax.scan(action_and_logit_scan_fn, jnp.zeros_like(action[key][-1]), action_inputs, reverse=True)
        targets[key] = jnn.one_hot(targets[key], act_space[key].classes) if act_space[key].discrete else targets[key]

    logit_inputs = (
        jaxutils.broadcast_to_match(next_is_first, logit), jaxutils.broadcast_to_match(next_is_change_or_term, logit), logit
    )
    _, targets['logit'] = jax.lax.scan(action_and_logit_scan_fn, jnp.zeros_like(logit[-1]), logit_inputs, reverse=True)
    _, targets['t_delta'] = jax.lax.scan(time_scan_fn, jnp.full((context.shape[1],), -seq_len - 2), (next_is_first, next_is_change_or_term), reverse=True)
    _, targets['reward'] = jax.lax.scan(reward_scan_fn, jnp.full_like(reward[-1], 0.0),
                                     (is_first, jnp.roll(next_is_change_or_term, shift=1, axis=0),
                                      reward), reverse=True)

    targets = {k: swap(v) for k, v in targets.items()}
    targets['stoch'] = tfd.Independent(jaxutils.OneHotDist(targets['logit']), 1)

    mask = targets['t_delta'] >= 0
    return targets, mask


def get_action_posterior_future_inputs(data):
    """Unlike generate_hlwm_targets, this function returns future contexts and logits AT the timestep where we have a new context, not before"""
    swap = lambda x: x.transpose([1, 0] + list(range(2, len(x.shape))))
    context, logit, is_first, is_terminal = treemap(swap, (
    data['context'], data['logit'], data['is_first'], data['is_terminal']))

    # Opposed to generate hlwm targets function, these are 1 on the timestep where we have the new context
    changed_indices = (jnp.diff(context, prepend=jnp.array([context[0]]), axis=0) != 0)
    changed_indices = jnp.any(changed_indices, axis=-1)
    is_changed_or_term = jnp.logical_or(changed_indices, is_terminal)

    def scan_fn(carry, x):
        first, change_or_term, inp = x
        new_carry = jnp.where(first, 0, jnp.where(change_or_term, inp, carry))
        return new_carry, carry

    post_inputs = {}
    logit_inputs = (
    jaxutils.broadcast_to_match(is_first, logit), jaxutils.broadcast_to_match(is_changed_or_term, logit), logit)
    post_inputs['logit_next'] = jax.lax.scan(scan_fn, jnp.full_like(logit[0], 0), logit_inputs, reverse=True)[1]

    context_inputs = (
    jaxutils.broadcast_to_match(is_first, context), jaxutils.broadcast_to_match(is_changed_or_term, context), context)
    post_inputs['context_next'] = jax.lax.scan(scan_fn, jnp.full_like(context[0], 0), context_inputs, reverse=True)[1]

    post_inputs = {k: swap(v) for k, v in post_inputs.items()}
    return post_inputs


class HierarchicalWorldModel(nj.Module):
  def __init__(self, obs_space, act_space, config, wm):
    self.obs_space = obs_space
    self.act_space = act_space
    self.config = config
    stoch_shape = (config.dyn.rssm.stoch, config.dyn.rssm.classes) if config.dyn.rssm.classes else (config.dyn.rssm.stoch, )
    stoch_dist = 'onehot' if config.dyn.rssm.classes else 'normal'
    action_shape = {
        k: (*s.shape, s.classes) if s.discrete else s.shape
        for k, s in self.act_space.items()}
    action_dist = {
        k: self.config.ac.actor_dist_disc if v.discrete else self.config.ac.actor_dist_cont
        for k, v in self.act_space.items()}
    self.heads = {
        'action': basenets.MLP(shape=action_shape, dist=action_dist, **config.hl_wm.action_head, name='action'),
        't_delta': basenets.MLP((), **config.hl_wm.t_delta_head, name='t_delta'),
        'reward': basenets.MLP((), **config.hl_wm.reward_head, name='reward'),
    }
    self.stoch_head = basenets.MLP(stoch_shape, dist=stoch_dist, **config.hl_wm.stoch_head, name='stoch')
    self.hl_action_prior = basenets.MLP(self.config.hl_wm.hl_actions, **config.hl_wm.hl_action_prior, name='hl_action_prior')
    self.hl_action_post = basenets.MLP(self.config.hl_wm.hl_actions, **config.hl_wm.hl_action_post, name='hl_action_post')

    self.wm = wm

  def expand_scales(self, scales):
      action_scale = scales.pop('hlwm_action')
      scales.update({f'hlwm_{k}': action_scale for k in self.act_space})

  def imagine(self, policy, replay_outs):
    def imgstep(carry, _):
        lat, act = carry
        lat, out = self.step(lat, act)
        out['coarse_stoch'] = sg(out['coarse_stoch'])
        out['stoch'] = out['coarse_stoch']
        lat['stoch'] = lat.pop('coarse_stoch')
        act = policy(out)
        return (lat, act), (out, act)

    if self.config.imag_start == 'all':
        B, T = replay_outs['stoch'].shape[:2]
        startlat = self.wm.dyn.outs_to_coarse_carry(treemap(lambda x: x.reshape((B * T, 1, *x.shape[2:])), replay_outs))
        startout = treemap(lambda x: x.reshape((B * T, *x.shape[2:])), replay_outs)
    elif self.config.imag_start == 'last':
        startlat = self.wm.dyn.outs_to_carry(replay_outs)
        startout = treemap(lambda x: x[:, -1], replay_outs)
    else:
        raise ValueError(f"Invalid imag_start: {self.config.imag_start}")
    if self.config.imag_repeat > 1:
        N = self.config.imag_repeat
        startlat, startout = treemap(lambda x: x.repeat(N, 0), (startlat, startout))
    startout = {k: startout[k] for k in ['context', 'coarse_stoch', 'coarse_logit', 'coarse_gates']}
    startout['stoch'] = startout['coarse_stoch']
    startact = policy(startout)
    start_data, start_outs = (startlat, startact), (startout, startact)
    outs, acts = imagine(imgstep, start_data, self.config.imag_length, self.config.imag_unroll, start_outs)

    rew = self.heads['reward']({**outs, **acts}).mean()[:, :-1]
    rew = jnp.concatenate([jnp.zeros_like(rew[:, 0][:, None]), rew], 1)
    con = self.wm.coarse_con(outs).mean()
    return outs, acts, rew, con

  def step(self, latent, hl_action):
    latent = {**latent, **hl_action}
    bdims = len(latent['context'].shape) - 1

    carry = self.predict(latent)
    stoch = carry['coarse_stoch']
    context = carry['context']
    action = carry['action']

    #action_pred = self.heads['action'](latent, bdims=bdims)
    #action = {}
    #for key, dist in action_pred.items():
    #    action[key] = dist.sample(seed=nj.seed()) if self.config.hl_wm.step_sample_actions else dist.mode()
    #stoch = self.stoch_head(latent, bdims=bdims).sample(seed=nj.seed()) if self.config.hl_wm.step_sample_stoch else self.stoch_head(latent, bdims=bdims).mode()
    #context = latent['context']
    #if bdims > 1:
    #    context = latent['context'].reshape(-1, latent['context'].shape[-1])
    #    stoch = stoch.reshape(-1, *stoch.shape[-2:])
    #    action = {k: v.reshape(-1, v.shape[-1]) for k, v in action.items()}
    carry, outs = self.wm.dyn.coarse_imagine({'context': context, 'stoch': stoch}, action)
    if bdims > 1:
        carry = {k: v.reshape(*latent['context'].shape[:-1], *v.shape[1:]) for k, v in carry.items()}
        outs = {k: v.reshape(*latent['context'].shape[:-1], *v.shape[1:]) for k, v in outs.items()}
    return carry, outs


  def predict(self, latent):
    bdims = len(latent['context'].shape) - 1
    action_pred = self.heads['action'](latent, bdims=bdims)
    action = {}
    for key, dist in action_pred.items():
        action[key] = dist.sample(seed=nj.seed()) if self.config.hl_wm.step_sample_actions else dist.mode()
    stoch = self.stoch_head(latent, bdims=bdims).sample(seed=nj.seed()) if self.config.hl_wm.step_sample_stoch else self.stoch_head(latent, bdims=bdims).mode()
    context = latent['context']
    if bdims > 1:
        context = latent['context'].reshape(-1, latent['context'].shape[-1])
        stoch = stoch.reshape(-1, *stoch.shape[-2:])
        action = {k: v.reshape(-1, v.shape[-1]) for k, v in action.items()}
    return dict(coarse_stoch=stoch, context=context, action=action)

  def loss(self, dyn_outs, data):
    if self.config.hl_wm.stop_input_gradients:
      data = sg(data)
      dyn_outs = sg(dyn_outs)
    targets, mask = sg(generate_high_level_world_model_training_targets({**dyn_outs, **data}, self.act_space, self.config.ac.horizon))
    post_inputs = sg(get_action_posterior_future_inputs({**dyn_outs, **data}))

    action_post = self.hl_action_post({**dyn_outs, **post_inputs})
    action_prior = self.hl_action_prior(dyn_outs)

    dyn_outs = {**dyn_outs, 'hl_action': action_post.sample(seed=nj.seed()) if self.config.hl_wm.sample_actions else action_post.mode()}

    dists = {}
    for name, head in filter(lambda item: item[0] in self.config.hl_wm.heads, self.heads.items()):
      out = head(dyn_outs)
      out = out if isinstance(out, dict) else {name: out}
      dists.update(out)

    losses = {}
    stoch_pred = self.stoch_head(dyn_outs)
    # Important to do KL(targets, pred) instead of KL(pred, targets)
    losses['stoch'] = jnp.where(mask, targets['stoch'].kl_divergence(stoch_pred), 0.0)
    losses['action_prior_to_post'] = jnp.where(mask, sg(action_post).kl_divergence(action_prior), 0.0)

    for key, dist in dists.items():
      loss = -dist.log_prob(f32(targets[key]))
      assert loss.shape == targets['action'].shape[:2], (key, loss.shape)
      losses[key] = jnp.where(jaxutils.broadcast_to_match(mask, loss), loss, 0.0)

    metrics = self._metrics(dists, targets, mask)
    return losses, metrics

  def report(self, data, carry):
      report = {}
      newlat, dyn_outs, wm_outs = self.wm.observe(data, carry)

      hl_actions = self.config.hl_wm.hl_actions
      seq_len = data['is_first'].shape[1]
      hl_action = jnp.repeat(jnp.eye(hl_actions)[:, None], seq_len, 1)

      step_input = {
          'context': jnp.repeat(dyn_outs['context'][0][None], hl_actions, 0),
          'stoch': jnp.repeat(dyn_outs['stoch'][0][None], hl_actions, 0),
      }

      def make_reconstruction(latents, decoder):
          latents['stoch'] = latents['coarse_stoch']
          recon = decoder(latents)['image'].mode()
          recon = recon.transpose(1, 2, 0, 3, 4)
          return recon

      def add_sampled_selection(recon, samp_hl_act):
          selector = jnp.argmax(samp_hl_act, axis=1)
          selected = jax.vmap(lambda x, sel: x[:, sel, :, :])(recon, selector)
          selected = selected[:, :, jnp.newaxis, :, :]
          recon = jnp.concatenate([ll_recon, recon, selected], axis=2)
          recon = recon.reshape(seq_len, recon.shape[1], -1, *recon.shape[4:])
          full_recon = jnp.concatenate([data['image'][0], recon], axis=2)
          return full_recon.reshape(-1, *full_recon.shape[2:])

      # 1-step low-level reconstruction
      ll_recon = self.wm.dec(dyn_outs)['image'].mode()[0][None]
      ll_recon = ll_recon.transpose(1, 2, 0, 3, 4)

      # Predicted state AFTER next context chang (step)
      step_latent, _ = self.step(step_input, {'hl_action': hl_action})
      step_recon = make_reconstruction(step_latent, self.wm.coarse_dec)

      # Sample HL action
      samp_hl_act = self.hl_action_prior(dyn_outs).sample(seed=nj.seed())[0]

      report['hlwm_next_context_recon'] = add_sampled_selection(step_recon, samp_hl_act)

      # Predicted state BEFORE next context change  (predict)
      pred_latent = self.predict({'hl_action': hl_action, **step_input})
      pred_latent.pop('action')
      pred_latent = {
          k: v.reshape(*step_input['context'].shape[:-1], *v.shape[1:])
          for k, v in pred_latent.items()
      }
      pred_recon = make_reconstruction(pred_latent, self.wm.coarse_dec)

      report['hlwm_predict_recon'] = add_sampled_selection(pred_recon, samp_hl_act)

      return report

  # def report(self, data, carry):
  #   report = {}
  #   newlat, dyn_outs, wm_outs = self.wm.observe(data, carry)
  #   # Make an array of all possible hl_actions (one-hot encoded) with the same shape as original hl_action
  #   # plus an additional dimension for the possibilities and assign it to post['hl_action']
  #   hl_actions = self.config.hl_wm.hl_actions
  #   seq_len = data['is_first'].shape[1]
  #   hl_action = jnp.repeat(jnp.eye(hl_actions)[:, None], seq_len, 1)
  #   step_input = {}
  #   step_input['context'] = jnp.repeat(dyn_outs['context'][0][None], hl_actions, 0)
  #   step_input['stoch'] = jnp.repeat(dyn_outs['stoch'][0][None], hl_actions, 0)
  #
  #   ll_recon = self.wm.dec(dyn_outs)['image'].mode()[0][None]
  #   ll_recon = ll_recon.transpose(1, 2, 0, 3, 4)
  #   print("Low level")
  #   print(ll_recon.shape)
  #
  #   # Full step prediction
  #   next_step_lat, _ = self.step(step_input, {'hl_action': hl_action})
  #   next_step_lat['stoch'] = next_step_lat['coarse_stoch']
  #   futures_recon = self.wm.coarse_dec(next_step_lat)['image'].mode()
  #   print("Future rec level")
  #   print(futures_recon.shape)
  #   futures_recon = futures_recon.transpose(1, 2, 0, 3, 4)
  #   print("Future rec level2")
  #   print(futures_recon.shape)
  #   #assert False
  #   # append prediction based on sampled HL action at end
  #   samp_hl_act = self.hl_action_prior(dyn_outs).sample(seed=nj.seed())[0]
  #   selector = jnp.argmax(samp_hl_act, axis=1)
  #   selected_images = jax.vmap(lambda x, sel: x[:, sel, :, :])(futures_recon, selector)
  #   selected_images = selected_images[:, :, jnp.newaxis,  :, :]
  #   futures_recon = jnp.concatenate([ll_recon, futures_recon, selected_images], axis=2)
  #   # to pixel shape
  #   futures_recon = futures_recon.reshape(seq_len, futures_recon.shape[1], -1, *futures_recon.shape[4:])
  #   recon = jnp.concatenate([data['image'][0], futures_recon], axis=2)
  #   recon = recon.reshape(-1, *recon.shape[2:])
  #   report['hlwm_next_context_recon'] = recon
  #
  #   # Just HL prediction
  #   pred_latent = self.predict({'hl_action': hl_action, **step_input})
  #   pred_latent.pop('action')
  #   pred_latent = {k: v.reshape(*step_input['context'].shape[:-1], *v.shape[1:]) for k, v in pred_latent.items()}
  #   pred_latent['stoch'] = pred_latent['coarse_stoch']
  #   skip_recon = self.wm.coarse_dec(pred_latent)['image'].mode()
  #   skip_recon = skip_recon.transpose(1, 2, 0, 3, 4)
  #   # append prediction based on sampled HL action at end
  #   selector2 = jnp.argmax(samp_hl_act, axis=1)
  #   selected_images2 = jax.vmap(lambda x, sel: x[:, sel, :, :])(skip_recon, selector2)
  #   selected_images2 = selected_images2[:, :, jnp.newaxis, :, :]
  #   skip_recon = jnp.concatenate([ll_recon, skip_recon, selected_images2], axis=2)
  #   # to pixel shape
  #   skip_recon = skip_recon.reshape(seq_len, skip_recon.shape[1], -1, *skip_recon.shape[4:])
  #   recon2 = jnp.concatenate([data['image'][0], skip_recon], axis=2)
  #   recon2 = recon2.reshape(-1, *recon2.shape[2:])
  #   report['hlwm_predict_recon'] = recon2
  #
  #   return report

  def _metrics(self, dists, targets, mask):
    metrics = {}
    reward_targets = jnp.where(mask, targets['reward'], 0.0)
    metrics['hlwm_reward_max_targets'] = jnp.abs(reward_targets).max()
    metrics['hlwm_reward_max_pred'] = jnp.abs(dists['reward'].mean()).max()
    if 'reward' in dists:
      stats = jaxutils.balance_stats(dists['reward'], reward_targets, 0.1)
      metrics.update({f'hlwm_reward_{k}': v for k, v in stats.items()})
    return metrics