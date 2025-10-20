"""
Environment wrappers for additional functionality.

This module provides various wrappers that can be applied to environments
to add features like reward shaping, observation normalization, action clipping,
and more.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, Tuple, Optional, Union
from collections import deque


class RewardShapingWrapper(gym.Wrapper):
    """
    Wrapper for reward shaping and scaling.
    
    This wrapper allows for easy modification of rewards, including
    scaling, clipping, and adding shaped rewards.
    """
    
    def __init__(self, 
                 env: gym.Env,
                 reward_scale: float = 1.0,
                 reward_clip: Optional[Tuple[float, float]] = None,
                 shaped_rewards: Optional[Dict[str, float]] = None):
        """
        Initialize reward shaping wrapper.
        
        Args:
            env: Environment to wrap
            reward_scale: Global reward scaling factor
            reward_clip: Tuple of (min, max) to clip rewards
            shaped_rewards: Dictionary of additional shaped rewards
        """
        super().__init__(env)
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip
        self.shaped_rewards = shaped_rewards or {}
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Apply reward scaling
        reward *= self.reward_scale
        
        # Add shaped rewards
        for reward_name, reward_value in self.shaped_rewards.items():
            reward += reward_value
        
        # Clip reward if specified
        if self.reward_clip is not None:
            reward = np.clip(reward, self.reward_clip[0], self.reward_clip[1])
        
        return obs, reward, terminated, truncated, info


class ObservationNormalizationWrapper(gym.Wrapper):
    """
    Wrapper for observation normalization.
    
    This wrapper normalizes observations using running statistics,
    which can help with training stability.
    """
    
    def __init__(self, 
                 env: gym.Env,
                 normalize: bool = True,
                 epsilon: float = 1e-8):
        """
        Initialize observation normalization wrapper.
        
        Args:
            env: Environment to wrap
            normalize: Whether to normalize observations
            epsilon: Small value to avoid division by zero
        """
        super().__init__(env)
        self.normalize = normalize
        self.epsilon = epsilon
        
        if self.normalize:
            self.obs_mean = np.zeros(env.observation_space.shape, dtype=np.float32)
            self.obs_var = np.ones(env.observation_space.shape, dtype=np.float32)
            self.obs_count = 0
    
    def _update_stats(self, obs):
        """Update running statistics for observation normalization."""
        if not self.normalize:
            return
        
        self.obs_count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.obs_count
        delta2 = obs - self.obs_mean
        self.obs_var += (delta * delta2 - self.obs_var) / self.obs_count
    
    def _normalize_obs(self, obs):
        """Normalize observation using running statistics."""
        if not self.normalize:
            return obs
        
        return (obs - self.obs_mean) / np.sqrt(self.obs_var + self.epsilon)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._update_stats(obs)
        return self._normalize_obs(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update_stats(obs)
        return self._normalize_obs(obs), reward, terminated, truncated, info


class ActionClippingWrapper(gym.Wrapper):
    """
    Wrapper for action clipping and scaling.
    
    This wrapper ensures actions stay within valid bounds and can
    apply additional scaling or clipping.
    """
    
    def __init__(self, 
                 env: gym.Env,
                 clip_actions: bool = True,
                 action_scale: float = 1.0):
        """
        Initialize action clipping wrapper.
        
        Args:
            env: Environment to wrap
            clip_actions: Whether to clip actions to valid range
            action_scale: Scaling factor for actions
        """
        super().__init__(env)
        self.clip_actions = clip_actions
        self.action_scale = action_scale
    
    def step(self, action):
        # Scale actions
        action = action * self.action_scale
        
        # Clip actions to valid range
        if self.clip_actions:
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        
        return self.env.step(action)


class FrameStackWrapper(gym.Wrapper):
    """
    Wrapper for frame stacking.
    
    This wrapper stacks multiple consecutive frames to provide
    temporal information to the agent.
    """
    
    def __init__(self, env: gym.Env, num_frames: int = 4):
        """
        Initialize frame stacking wrapper.
        
        Args:
            env: Environment to wrap
            num_frames: Number of frames to stack
        """
        super().__init__(env)
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)
        
        # Update observation space
        low = np.tile(env.observation_space.low, num_frames)
        high = np.tile(env.observation_space.high, num_frames)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype
        )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Fill frame buffer with initial observation
        for _ in range(self.num_frames):
            self.frames.append(obs)
        return self._get_obs(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=0)


class EpisodeStatsWrapper(gym.Wrapper):
    """
    Wrapper for tracking episode statistics.
    
    This wrapper tracks various statistics during episodes,
    such as total reward, episode length, and custom metrics.
    """
    
    def __init__(self, env: gym.Env):
        """Initialize episode statistics wrapper."""
        super().__init__(env)
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_stats = {}
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_stats = {}
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update episode statistics
        self.episode_reward += reward
        self.episode_length += 1
        
        # Add episode statistics to info
        info["episode"] = {
            "r": self.episode_reward,
            "l": self.episode_length,
            **self.episode_stats
        }
        
        return obs, reward, terminated, truncated, info
    
    def add_episode_stat(self, name: str, value: float):
        """Add a custom episode statistic."""
        self.episode_stats[name] = value


class TimeLimitWrapper(gym.Wrapper):
    """
    Wrapper for adding time limits to episodes.
    
    This wrapper terminates episodes after a specified number of steps,
    which can be useful for training stability.
    """
    
    def __init__(self, env: gym.Env, max_episode_steps: int):
        """
        Initialize time limit wrapper.
        
        Args:
            env: Environment to wrap
            max_episode_steps: Maximum number of steps per episode
        """
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self.step_count = 0
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.step_count += 1
        
        # Check if time limit reached
        if self.step_count >= self.max_episode_steps:
            truncated = True
        
        return obs, reward, terminated, truncated, info


def make_hopper_env_with_wrappers(
    render_mode: Optional[str] = None,
    normalize_obs: bool = True,
    frame_stack: int = 1,
    reward_scale: float = 1.0,
    max_episode_steps: int = 1000,
    **env_kwargs
) -> gym.Env:
    """
    Create a Hopper environment with commonly used wrappers.
    
    Args:
        render_mode: Rendering mode
        normalize_obs: Whether to normalize observations
        frame_stack: Number of frames to stack (1 = no stacking)
        reward_scale: Global reward scaling factor
        max_episode_steps: Maximum steps per episode
        **env_kwargs: Additional arguments for HopperEnv
        
    Returns:
        gym.Env: Wrapped hopper environment
    """
    from .hopper_env import HopperEnv
    
    # Create base environment
    env = HopperEnv(render_mode=render_mode, **env_kwargs)
    
    # Apply wrappers
    if max_episode_steps > 0:
        env = TimeLimitWrapper(env, max_episode_steps)
    
    if normalize_obs:
        env = ObservationNormalizationWrapper(env)
    
    if frame_stack > 1:
        env = FrameStackWrapper(env, frame_stack)
    
    if reward_scale != 1.0:
        env = RewardShapingWrapper(env, reward_scale=reward_scale)
    
    env = EpisodeStatsWrapper(env)
    env = ActionClippingWrapper(env)
    
    return env
