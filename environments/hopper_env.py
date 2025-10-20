"""
MuJoCo Hopper Environment Implementation.

This module provides a custom MuJoCo Hopper environment that can be used
with standard RL algorithms. The environment is compatible with OpenAI Gym
and provides additional features for training and evaluation.
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional
import os


class HopperEnv(gym.Env):
    """
    MuJoCo Hopper Environment.
    
    A continuous control environment where the agent must learn to hop forward
    as far as possible without falling. The hopper has 3 joints (thigh, leg, foot)
    and must balance while moving forward.
    
    Action Space:
        Box(3,): Joint torques for [thigh, leg, foot] joints
        Range: [-1, 1] for each joint
        
    Observation Space:
        Box(11,): State vector containing:
        - qpos[1:] (5): Joint positions (excluding root x)
        - qvel (3): Joint velocities  
        - qpos[0] (1): Root x position
        - qvel[0] (1): Root x velocity
        - contact (1): Foot contact indicator
        
    Reward:
        - Forward progress reward
        - Survival bonus
        - Control penalty
        - Fall penalty
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 render_mode: Optional[str] = None,
                 max_episode_steps: int = 1000,
                 reward_scale: float = 1.0,
                 control_cost_weight: float = 0.001,
                 forward_reward_weight: float = 1.0,
                 healthy_reward: float = 1.0,
                 terminate_when_unhealthy: bool = True,
                 healthy_z_range: Tuple[float, float] = (0.7, float('inf')),
                 healthy_angle_range: Tuple[float, float] = (-0.2, 0.2),
                 reset_noise_scale: float = 0.1):
        """
        Initialize the Hopper environment.
        
        Args:
            model_path: Path to MuJoCo XML model file
            render_mode: Rendering mode ('human', 'rgb_array', None)
            max_episode_steps: Maximum steps per episode
            reward_scale: Global reward scaling factor
            control_cost_weight: Weight for control cost penalty
            forward_reward_weight: Weight for forward progress reward
            healthy_reward: Reward for staying healthy
            terminate_when_unhealthy: Whether to terminate on unhealthy state
            healthy_z_range: Valid z position range for torso
            healthy_angle_range: Valid angle range for torso
            reset_noise_scale: Scale of random noise added to reset state
        """
        super().__init__()
        
        # Set default model path if not provided
        if model_path is None:
            # Use gymnasium's hopper model
            model_path = "hopper.xml"
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Environment parameters
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.reward_scale = reward_scale
        self.control_cost_weight = control_cost_weight
        self.forward_reward_weight = forward_reward_weight
        self.healthy_reward = healthy_reward
        self.terminate_when_unhealthy = terminate_when_unhealthy
        self.healthy_z_range = healthy_z_range
        self.healthy_angle_range = healthy_angle_range
        self.reset_noise_scale = reset_noise_scale
        
        # Action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )
        
        # Episode tracking
        self.step_count = 0
        self.episode_reward = 0.0
        
        # Initialize renderer if needed
        if self.render_mode == "human":
            self._setup_renderer()
    
    def _setup_renderer(self):
        """Setup MuJoCo renderer for human visualization."""
        try:
            import mujoco_viewer
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        except ImportError:
            print("Warning: mujoco_viewer not available. Install with: pip install mujoco_viewer")
            self.viewer = None
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        # Extract state information
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        
        # Get contact information
        contact = 0.0
        for i in range(self.data.ncon):
            contact_geom1 = self.data.contact[i].geom1
            contact_geom2 = self.data.contact[i].geom2
            # Check if foot is in contact with ground
            if (contact_geom1 == 0 and contact_geom2 == 0) or (contact_geom1 == 0 and contact_geom2 == 0):
                contact = 1.0
                break
        
        # Construct observation vector
        obs = np.concatenate([
            qpos[1:],  # Joint positions (excluding root x)
            qvel,      # Joint velocities
            [qpos[0]], # Root x position
            [qvel[0]], # Root x velocity  
            [contact]  # Contact indicator
        ])
        
        return obs.astype(np.float32)
    
    def _is_healthy(self) -> bool:
        """Check if the hopper is in a healthy state."""
        z_pos = self.data.qpos[1]  # Torso z position
        angle = self.data.qpos[2]  # Torso angle
        
        return (self.healthy_z_range[0] <= z_pos <= self.healthy_z_range[1] and
                self.healthy_angle_range[0] <= angle <= self.healthy_angle_range[1])
    
    def _get_reward(self, action: np.ndarray) -> float:
        """Calculate reward for current state and action."""
        # Forward progress reward
        forward_reward = self.forward_reward_weight * self.data.qvel[0]
        
        # Control cost penalty
        control_cost = self.control_cost_weight * np.sum(np.square(action))
        
        # Healthy reward
        healthy_reward = self.healthy_reward if self._is_healthy() else 0.0
        
        # Total reward
        reward = forward_reward - control_cost + healthy_reward
        
        return reward * self.reward_scale
    
    def _get_terminated(self) -> bool:
        """Check if episode should be terminated."""
        if not self._is_healthy():
            return self.terminate_when_unhealthy
        return False
    
    def _get_truncated(self) -> bool:
        """Check if episode should be truncated."""
        return self.step_count >= self.max_episode_steps
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset MuJoCo simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Add random noise to initial state
        if self.reset_noise_scale > 0:
            noise = self.reset_noise_scale * np.random.normal(size=self.data.qpos.shape)
            self.data.qpos += noise
            self.data.qvel += noise
        
        # Forward simulation to stabilize
        mujoco.mj_forward(self.model, self.data)
        
        # Reset episode tracking
        self.step_count = 0
        self.episode_reward = 0.0
        
        obs = self._get_obs()
        info = {"episode": {"r": 0.0, "l": 0}}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        # Clip actions to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Apply actions to MuJoCo
        self.data.ctrl[:] = action
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Get observation and reward
        obs = self._get_obs()
        reward = self._get_reward(action)
        
        # Check termination conditions
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        
        # Update episode tracking
        self.step_count += 1
        self.episode_reward += reward
        
        # Prepare info dict
        info = {
            "episode": {
                "r": self.episode_reward,
                "l": self.step_count
            }
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human" and self.viewer is not None:
            self.viewer.render()
        elif self.render_mode == "rgb_array":
            # Return RGB array for video recording
            # This would require additional setup for offscreen rendering
            pass
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.close()
    
    def seed(self, seed: Optional[int] = None):
        """Set random seed for reproducibility."""
        if seed is not None:
            np.random.seed(seed)
        return [seed]


def make_hopper_env(**kwargs) -> HopperEnv:
    """
    Create a Hopper environment with default parameters.
    
    Args:
        **kwargs: Additional arguments passed to HopperEnv
        
    Returns:
        HopperEnv: Configured hopper environment
    """
    return HopperEnv(**kwargs)


if __name__ == "__main__":
    # Test the environment
    env = make_hopper_env(render_mode="human")
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step}: reward={reward:.3f}, terminated={terminated}, truncated={truncated}")
        
        if terminated or truncated:
            break
    
    env.close()
