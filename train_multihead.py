"""
Multi-head actor-critic training for Robotron RL.

Architecture:
- Shared feature extractor (MlpExtractor 512x512)
- Two policy heads: movement (Categorical(8)) and shooting (Categorical(8))
- Two value heads: V_move (survival-focused) and V_shoot (kill-focused)
- Two reward streams:
    * Movement reward: survival bonus, death penalty, stagnation penalty, wave complete bonus
    * Shooting reward: score delta, spawner kill bonus, shooter kill bonus
  Each head gets 80% of its own stream + 20% of the other for cross-signal awareness.

Action space stays MultiDiscrete([8, 8]) but each head optimizes for its own reward.
"""
import argparse
import warnings
from typing import Callable, Optional, Tuple, Dict, Any

warnings.filterwarnings('ignore', message='.*UnsupportedFieldAttribute.*')
warnings.filterwarnings('ignore', message='.*Field.*has no.*attribute.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='wandb')
warnings.filterwarnings('ignore', message='.*frozen.*', category=UserWarning, module='pydantic')
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')

import numpy as np
import torch
import torch as th
import torch.nn as nn
from torch.nn import functional as F

import gymnasium as gym
from gymnasium import spaces

from robotron import RobotronEnv
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import explained_variance
from wandb.integration.sb3 import WandbCallback
import wandb

from wrappers import FrameSkipWrapper
from position_wrapper import GroundTruthPositionWrapper, OBS_DIM


SPAWNER_TYPES = frozenset({'Sphereoid', 'Quark'})
SHOOTER_TYPES = frozenset({'Enforcer', 'Tank'})


# ═══════════════════════════════════════════════════════════════════════════
# REWARD WRAPPER — per-head rewards passed through info dict
# ═══════════════════════════════════════════════════════════════════════════

class MultiHeadRewardWrapper(gym.Wrapper):
    """Reward wrapper that computes two reward streams (movement, shooting).

    Movement reward (survival-focused):
      - survival bonus (+0.3/step)
      - death penalty (-20)
      - stagnation penalty (-0.5/step when camping)
      - wave complete bonus (+250/wave)

    Shooting reward (kill-focused):
      - score delta / 10 (base game reward)
      - spawner kill bonus (+50 per Sphereoid/Quark)
      - shooter kill bonus (+20 per Enforcer/Tank)

    The gym step reward is the 80/20 mix: env sees `combined` reward.
    The raw split rewards are exposed in info['reward_move'] and info['reward_shoot']
    so the training loop can use them for separate advantage computation.
    """
    STAGNATION_WINDOW = 30
    STAGNATION_DIST = 20.0
    STAGNATION_PENALTY = -0.5
    SPAWNER_KILL_BONUS = 50.0
    SHOOTER_KILL_BONUS = 20.0
    DEATH_PENALTY = -20.0
    SURVIVAL_BONUS = 0.3
    WAVE_COMPLETE_BONUS = 250.0
    # Cross-head mixing so each head feels some of the other's consequences
    CROSS_MIX = 0.2

    def __init__(self, env, score_scale=10.0):
        super().__init__(env)
        self.score_scale = score_scale
        self.last_score = 0
        self.last_lives = 0
        self.last_spawner_count = 0
        self.last_shooter_count = 0
        self.last_level = 0
        self._pos_history = []

    def _player_pos(self):
        player = self.env.unwrapped.engine.player
        return (float(player.rect.x), float(player.rect.y))

    def _count_spawners(self):
        return sum(1 for s in self.env.unwrapped.engine.enemy_group
                   if s.__class__.__name__ in SPAWNER_TYPES)

    def _count_shooters(self):
        return sum(1 for s in self.env.unwrapped.engine.enemy_group
                   if s.__class__.__name__ in SHOOTER_TYPES)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        engine = self.env.unwrapped.engine
        self.last_score = engine.score
        self.last_lives = engine.lives
        self.last_spawner_count = self._count_spawners()
        self.last_shooter_count = self._count_shooters()
        self.last_level = engine.level
        self._pos_history = [self._player_pos()]
        return obs, info

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)
        engine = self.env.unwrapped.engine

        # ── Shooting reward components ─────────────────────────────
        current_score = engine.score
        score_delta = current_score - self.last_score
        self.last_score = current_score
        r_shoot = score_delta / self.score_scale

        spawner_count = self._count_spawners()
        spawners_killed = self.last_spawner_count - spawner_count
        if spawners_killed > 0 and score_delta > 0:
            r_shoot += self.SPAWNER_KILL_BONUS * spawners_killed
        self.last_spawner_count = spawner_count

        shooter_count = self._count_shooters()
        shooters_killed = self.last_shooter_count - shooter_count
        if shooters_killed > 0 and score_delta > 0:
            r_shoot += self.SHOOTER_KILL_BONUS * shooters_killed
        self.last_shooter_count = shooter_count

        # ── Movement reward components ─────────────────────────────
        r_move = 0.0
        current_lives = engine.lives
        if current_lives < self.last_lives:
            r_move += self.DEATH_PENALTY
        else:
            r_move += self.SURVIVAL_BONUS
        self.last_lives = current_lives

        current_level = engine.level
        if current_level > self.last_level:
            r_move += self.WAVE_COMPLETE_BONUS * (current_level - self.last_level)
        self.last_level = current_level

        pos = self._player_pos()
        self._pos_history.append(pos)
        if len(self._pos_history) > self.STAGNATION_WINDOW:
            old_pos = self._pos_history[-self.STAGNATION_WINDOW]
            dx = pos[0] - old_pos[0]
            dy = pos[1] - old_pos[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < self.STAGNATION_DIST:
                r_move += self.STAGNATION_PENALTY
        if len(self._pos_history) > self.STAGNATION_WINDOW + 10:
            self._pos_history = self._pos_history[-self.STAGNATION_WINDOW:]

        # ── Cross-head mixing so each head feels some of the other ─
        r_move_mixed = r_move + self.CROSS_MIX * r_shoot
        r_shoot_mixed = r_shoot + self.CROSS_MIX * r_move

        # Combined reward for env return value (mainly for logging/VecNormalize)
        combined = r_move_mixed + r_shoot_mixed

        info['reward_move'] = float(r_move_mixed)
        info['reward_shoot'] = float(r_shoot_mixed)

        return obs, combined, terminated, truncated, info


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-HEAD POLICY — shared features, two action heads, two value heads
# ═══════════════════════════════════════════════════════════════════════════

class MultiHeadActorCriticPolicy(ActorCriticPolicy):
    """ActorCriticPolicy with two independent value heads (V_move, V_shoot).

    The two policy heads already exist as MultiCategoricalDistribution
    (PPO's native support for MultiDiscrete action spaces). We only need
    to add a second value head and expose per-head values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add second value net — same architecture as the first, parallel to value_net
        latent_dim_vf = self.mlp_extractor.latent_dim_vf
        self.value_net_shoot = nn.Linear(latent_dim_vf, 1)
        # Rename existing for clarity (keep original for SB3 compatibility)
        # self.value_net = movement value head
        # self.value_net_shoot = shooting value head

    def forward(self, obs, deterministic=False):
        """Forward pass: return action, combined_value, log_prob.

        SB3's PPO expects a scalar value per step. We return V_move + V_shoot
        as the "combined" value so existing buffer machinery works. We also
        store V_move and V_shoot separately via `forward_heads()` for the
        custom training loop.
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        values_move = self.value_net(latent_vf)
        values_shoot = self.value_net_shoot(latent_vf)
        values_combined = values_move + values_shoot

        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values_combined, log_prob

    def predict_values_split(self, obs):
        """Return (V_move, V_shoot) separately for per-head advantage computation."""
        features = self.extract_features(obs)
        if self.share_features_extractor:
            _, latent_vf = self.mlp_extractor(features)
        else:
            _, vf_features = features
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        return self.value_net(latent_vf), self.value_net_shoot(latent_vf)

    def evaluate_actions_split(self, obs, actions):
        """Evaluate actions and return (log_prob, V_move, V_shoot, entropy)."""
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values_move = self.value_net(latent_vf)
        values_shoot = self.value_net_shoot(latent_vf)
        return log_prob, values_move, values_shoot, distribution.entropy()


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-HEAD PPO — subclass that uses split rewards and split advantages
# ═══════════════════════════════════════════════════════════════════════════

class MultiHeadPPO(PPO):
    """PPO with separate reward streams and advantages for movement and shooting.

    The rollout buffer is the standard SB3 buffer, but we override rollout
    collection to store per-head rewards and we override train() to compute
    per-head advantages.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Parallel storage for per-head rewards (populated during rollout)
        buf_shape = (self.n_steps, self.n_envs)
        self._rewards_move = np.zeros(buf_shape, dtype=np.float32)
        self._rewards_shoot = np.zeros(buf_shape, dtype=np.float32)
        self._values_move = np.zeros(buf_shape, dtype=np.float32)
        self._values_shoot = np.zeros(buf_shape, dtype=np.float32)
        self._advantages_move = np.zeros(buf_shape, dtype=np.float32)
        self._advantages_shoot = np.zeros(buf_shape, dtype=np.float32)
        self._returns_move = np.zeros(buf_shape, dtype=np.float32)
        self._returns_shoot = np.zeros(buf_shape, dtype=np.float32)

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        """Collect rollout storing per-head rewards and values from info dict.

        This mostly mirrors SB3's PPO.collect_rollouts but also tracks the
        per-head reward streams from info dict and per-head values from the
        policy.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = th.as_tensor(self._last_obs, device=self.device, dtype=th.float32)
                actions, values_combined, log_probs = self.policy(obs_tensor)
                v_move, v_shoot = self.policy.predict_values_split(obs_tensor)
            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            # Extract per-head rewards from info dict
            r_move = np.array([info.get('reward_move', 0.0) for info in infos], dtype=np.float32)
            r_shoot = np.array([info.get('reward_shoot', 0.0) for info in infos], dtype=np.float32)

            self.num_timesteps += env.num_envs

            # Give callbacks access to the locals for logging etc.
            self._update_info_buffer(infos, dones)
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)
            n_steps += 1

            # Store standard rollout data in buffer
            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values_combined,
                log_probs,
            )

            # Store per-head parallel data (same index as rollout_buffer.pos - 1)
            pos = rollout_buffer.pos - 1
            self._rewards_move[pos] = r_move
            self._rewards_shoot[pos] = r_shoot
            self._values_move[pos] = v_move.cpu().numpy().flatten()
            self._values_shoot[pos] = v_shoot.cpu().numpy().flatten()

            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            obs_tensor = th.as_tensor(new_obs, device=self.device, dtype=th.float32)
            _, last_values_combined, _ = self.policy(obs_tensor)
            last_v_move, last_v_shoot = self.policy.predict_values_split(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=last_values_combined, dones=dones)

        # Compute per-head advantages and returns using GAE
        self._compute_split_gae(
            last_v_move.cpu().numpy().flatten(),
            last_v_shoot.cpu().numpy().flatten(),
            dones,
        )

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def _compute_split_gae(self, last_v_move, last_v_shoot, dones):
        """Compute GAE advantages separately for each head."""
        gamma = self.gamma
        lam = self.gae_lambda

        # Get episode_starts from the rollout buffer (same indexing as parallel arrays)
        episode_starts = self.rollout_buffer.episode_starts

        for head, rewards, values, last_values, advantages, returns in [
            ('move', self._rewards_move, self._values_move, last_v_move, self._advantages_move, self._returns_move),
            ('shoot', self._rewards_shoot, self._values_shoot, last_v_shoot, self._advantages_shoot, self._returns_shoot),
        ]:
            last_gae_lam = 0.0
            for step in reversed(range(self.n_steps)):
                if step == self.n_steps - 1:
                    next_non_terminal = 1.0 - dones.astype(np.float32)
                    next_values = last_values
                else:
                    next_non_terminal = 1.0 - episode_starts[step + 1]
                    next_values = values[step + 1]
                delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
                last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
                advantages[step] = last_gae_lam
            returns[:] = advantages + values

    def train(self):
        """Train policy with per-head advantages and losses."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses, pg_losses_move, pg_losses_shoot = [], [], []
        value_losses_move, value_losses_shoot, clip_fractions = [], [], []

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                log_prob, values_move, values_shoot, entropy = self.policy.evaluate_actions_split(
                    rollout_data.observations, actions
                )
                values_move = values_move.flatten()
                values_shoot = values_shoot.flatten()

                # Use the parallel per-head advantage arrays indexed the same way
                # The rollout buffer iterates over a flat index — we need to match it.
                # SB3's RolloutBuffer.get() shuffles indices, so we need to retrieve them.
                # Workaround: use the buffer's own advantages as the "total" advantage and
                # compute the split via the ratio of our per-head advantages.
                # Simpler: flatten our parallel arrays and index by the buffer's internal indices.
                # SB3 stores `_batch_indices` during iteration — access via instance state.
                batch_idx = self.rollout_buffer._last_batch_idx  # will set below

                adv_move = th.as_tensor(
                    self._advantages_move.flatten()[batch_idx],
                    device=self.device, dtype=th.float32,
                )
                adv_shoot = th.as_tensor(
                    self._advantages_shoot.flatten()[batch_idx],
                    device=self.device, dtype=th.float32,
                )
                ret_move = th.as_tensor(
                    self._returns_move.flatten()[batch_idx],
                    device=self.device, dtype=th.float32,
                )
                ret_shoot = th.as_tensor(
                    self._returns_shoot.flatten()[batch_idx],
                    device=self.device, dtype=th.float32,
                )

                # Normalize advantages
                if self.normalize_advantage:
                    if len(adv_move) > 1:
                        adv_move = (adv_move - adv_move.mean()) / (adv_move.std() + 1e-8)
                        adv_shoot = (adv_shoot - adv_shoot.mean()) / (adv_shoot.std() + 1e-8)

                # Combined advantage for ratio clipping (policy sees joint distribution)
                # Each head contributes equally to the policy gradient
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # Policy loss computed as sum of per-head contributions
                # Since both heads share the same action distribution and log_prob,
                # we use the sum of advantages as the effective "joint" advantage.
                adv_combined = adv_move + adv_shoot
                policy_loss_1 = adv_combined * ratio
                policy_loss_2 = adv_combined * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Split value losses — each value head learns its own returns
                value_loss_move = F.mse_loss(ret_move, values_move)
                value_loss_shoot = F.mse_loss(ret_shoot, values_shoot)

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * (value_loss_move + value_loss_shoot)
                )

                # Approximate KL for early stopping
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                # Clip fraction
                with th.no_grad():
                    clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).cpu().numpy()
                    clip_fractions.append(clip_fraction)

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                entropy_losses.append(entropy_loss.item())
                pg_losses_move.append((adv_move * ratio).mean().item())
                pg_losses_shoot.append((adv_shoot * ratio).mean().item())
                value_losses_move.append(value_loss_move.item())
                value_losses_shoot.append(value_loss_shoot.item())

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var_move = explained_variance(
            self._values_move.flatten(), self._returns_move.flatten()
        )
        explained_var_shoot = explained_variance(
            self._values_shoot.flatten(), self._returns_shoot.flatten()
        )

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss_move", np.mean(pg_losses_move))
        self.logger.record("train/policy_gradient_loss_shoot", np.mean(pg_losses_shoot))
        self.logger.record("train/value_loss_move", np.mean(value_losses_move))
        self.logger.record("train/value_loss_shoot", np.mean(value_losses_shoot))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/explained_variance_move", explained_var_move)
        self.logger.record("train/explained_variance_shoot", explained_var_shoot)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)


# Monkey-patch the rollout buffer to expose batch indices during iteration
# (SB3 shuffles indices but doesn't expose them — we need them for per-head advantages)
_orig_get = RolloutBuffer.get
def _get_with_idx(self, batch_size=None):
    assert self.full, "Rollout buffer must be full before training"
    indices = np.random.permutation(self.buffer_size * self.n_envs)
    if not self.generator_ready:
        for tensor in ["observations", "actions", "values", "log_probs", "advantages", "returns"]:
            self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
        self.generator_ready = True
    if batch_size is None:
        batch_size = self.buffer_size * self.n_envs
    start_idx = 0
    while start_idx < self.buffer_size * self.n_envs:
        batch_idx = indices[start_idx : start_idx + batch_size]
        self._last_batch_idx = batch_idx  # expose for multi-head PPO
        yield self._get_samples(batch_idx)
        start_idx += batch_size
RolloutBuffer.get = _get_with_idx


# ═══════════════════════════════════════════════════════════════════════════
# METRICS CALLBACK
# ═══════════════════════════════════════════════════════════════════════════

class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.highest_score = 0

    def _on_step(self):
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info and 'score' in info:
                    score = info['score']
                    self.highest_score = max(self.highest_score, score)
                    kills = score // 100
                    level = info.get('level', 1)
                    self.logger.record('robotron/episode_score', score)
                    self.logger.record('robotron/highest_score', self.highest_score)
                    self.logger.record('robotron/episode_kills', kills)
                    self.logger.record('robotron/episode_level', level)
                    if self.verbose > 0 and score > 0:
                        print(f"Episode: score={score}, kills={kills}, level={level}, best={self.highest_score}")
        return True


class VecNormalizeCallback(BaseCallback):
    def __init__(self, vec_normalize_env, save_path, save_freq=100_000, verbose=0):
        super().__init__(verbose)
        self.vec_normalize_env = vec_normalize_env
        self.save_path = save_path
        self.save_freq = save_freq
        self.last_save = -1

    def _on_step(self):
        if self.num_timesteps >= self.save_freq and self.num_timesteps % self.save_freq == 0:
            if self.num_timesteps != self.last_save:
                self.vec_normalize_env.save(self.save_path)
                self.last_save = self.num_timesteps
        return True


# ═══════════════════════════════════════════════════════════════════════════
# ENV FACTORY
# ═══════════════════════════════════════════════════════════════════════════

class _MultiDiscreteActionAdapter(gym.ActionWrapper):
    """Expose MultiDiscrete([8,8]) externally, convert to Discrete(64) for the engine.

    action[0] = movement (0-7), action[1] = shooting (0-7)
    Engine encoding: discrete = move * 8 + shoot
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.MultiDiscrete([8, 8])

    def action(self, action):
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        return int(action[0]) * 8 + int(action[1])


def make_env(config_path='config.yaml', level=1, lives=3, rank=0, seed=0, headless=True, frame_skip=4):
    def _init():
        env = RobotronEnv(
            level=level,
            lives=lives,
            fps=0,
            config_path=config_path,
            always_move=True,
            headless=headless,
        )
        env = _MultiDiscreteActionAdapter(env)

        if frame_skip > 1:
            env = FrameSkipWrapper(env, skip=frame_skip)

        env = MultiHeadRewardWrapper(env, score_scale=10.0)
        env = GroundTruthPositionWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main(config_path='config.yaml', device='cpu', num_envs=16, total_timesteps=3_000_000,
         bc_checkpoint=None, start_level=1, lives=3,
         lr=2e-4, clip_range=0.2, ent_coef=0.02, vf_coef=0.5):

    fine_tuning = bc_checkpoint is not None

    run = wandb.init(
        project="robotron",
        group="ppo_multihead",
        config={
            'model': 'ppo_multihead',
            'total_timesteps': total_timesteps,
            'num_envs': num_envs,
            'obs_dim': OBS_DIM,
            'config_path': config_path,
            'bc_checkpoint': bc_checkpoint,
            'start_level': start_level,
            'lr': lr, 'clip_range': clip_range, 'ent_coef': ent_coef, 'vf_coef': vf_coef,
        },
        sync_tensorboard=True,
        save_code=True,
        mode="offline",
    )

    print("=" * 80)
    print("MULTI-HEAD ACTOR-CRITIC TRAINING")
    print("=" * 80)
    print(f"  Mode:        {'BC fine-tune from ' + bc_checkpoint if fine_tuning else 'scratch'}")
    print(f"  Architecture: 2 policy heads (move, shoot) + 2 value heads")
    print(f"  Rewards:     movement (survival) + shooting (kills), 80/20 cross-mix")
    print(f"  Obs:         {OBS_DIM} dims (945 category-slot positions)")
    print(f"  Config:      {config_path}")
    print(f"  Start level: {start_level}")
    print(f"  Timesteps:   {total_timesteps:,}")
    print(f"  HPs:         lr={lr}  clip={clip_range}  ent={ent_coef}  vf={vf_coef}")
    print("=" * 80)

    envs = SubprocVecEnv([
        make_env(config_path, level=start_level, lives=lives, rank=i, seed=42)
        for i in range(num_envs)
    ])
    envs = VecNormalize(envs, norm_obs=True, norm_reward=False, clip_obs=10.)

    model = MultiHeadPPO(
        policy=MultiHeadActorCriticPolicy,
        env=envs,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        device=device,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=0.5,
        learning_rate=lr,
        policy_kwargs={'net_arch': [512, 512]},
    )

    if fine_tuning and bc_checkpoint:
        print(f"\nLoading BC weights from {bc_checkpoint}...")
        # Load the SB3 checkpoint (single-head) and copy what matches
        bc_model = PPO.load(bc_checkpoint, device=device)
        bc_params = bc_model.get_parameters()
        # Copy shared extractor + movement action head + movement value head
        # The shooting value head stays randomly initialized
        policy_state = model.policy.state_dict()
        bc_policy_state = bc_params['policy']
        # Copy matching keys
        copied = 0
        for k, v in bc_policy_state.items():
            if k in policy_state and policy_state[k].shape == v.shape:
                policy_state[k] = v
                copied += 1
        model.policy.load_state_dict(policy_state)
        print(f"Copied {copied}/{len(policy_state)} parameters from BC checkpoint.")

    callbacks = [
        MetricsCallback(verbose=1),
        VecNormalizeCallback(
            vec_normalize_env=envs,
            save_path=f"models/{run.id}/vec_normalize.pkl",
            save_freq=100_000,
        ),
        WandbCallback(
            gradient_save_freq=1000,
            model_save_freq=100_000,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
        CheckpointCallback(
            save_freq=100_000 // num_envs,
            save_path=f"models/{run.id}/checkpoints",
            name_prefix="ppo_multihead_checkpoint",
        ),
    ]

    print("\nStarting training...\n")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    model.save(f"models/{run.id}/final_model")
    envs.save(f"models/{run.id}/vec_normalize.pkl")

    print("\n" + "=" * 80)
    print(f"Training complete! Saved to models/{run.id}/")
    print("=" * 80)
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-head PPO for Robotron")
    parser.add_argument("--config",         type=str, default='config.yaml')
    parser.add_argument("--device",         type=str, default='cpu')
    parser.add_argument("--num-envs",       type=int, default=16)
    parser.add_argument("--timesteps",      type=int, default=3_000_000)
    parser.add_argument("--bc-checkpoint",  type=str, default=None)
    parser.add_argument("--start-level",    type=int, default=1)
    parser.add_argument("--lives",          type=int, default=3)
    parser.add_argument("--lr",             type=float, default=2e-4)
    parser.add_argument("--clip-range",     type=float, default=0.2)
    parser.add_argument("--ent-coef",       type=float, default=0.02)
    parser.add_argument("--vf-coef",        type=float, default=0.5)
    args = parser.parse_args()

    main(
        config_path=args.config,
        device=args.device,
        num_envs=args.num_envs,
        total_timesteps=args.timesteps,
        bc_checkpoint=args.bc_checkpoint,
        start_level=args.start_level,
        lives=args.lives,
        lr=args.lr,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
    )
