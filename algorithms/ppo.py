"""
Proximal Policy Optimization (PPO) implementation.

This module provides a clean implementation of PPO that can be used
with the MuJoCo Hopper environment and other continuous control tasks.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, Any, Optional, Tuple, List
from collections import deque
import time
from tqdm import tqdm

from .base import BaseAlgorithm


class PPOPolicy(nn.Module):
    """
    PPO Policy network with actor-critic architecture.
    
    This network outputs both action probabilities and state values,
    using separate heads for the policy and value function.
    """
    
    def __init__(self, 
                 obs_dim: int, 
                 action_dim: int, 
                 hidden_dims: List[int] = [64, 64],
                 activation: str = "tanh",
                 log_std_init: float = -0.5):
        """
        Initialize PPO policy network.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ("tanh", "relu", "leaky_relu")
            log_std_init: Initial log standard deviation
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Activation function
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network layers
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self.activation
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Policy head (actor)
        self.policy_mean = nn.Linear(prev_dim, action_dim)
        self.policy_log_std = nn.Parameter(torch.full((action_dim,), log_std_init))
        
        # Value head (critic)
        self.value_head = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            obs: Observation tensor
            
        Returns:
            Tuple of (action_mean, action_log_std, value)
        """
        shared_features = self.shared_layers(obs)
        
        # Policy outputs
        action_mean = self.policy_mean(shared_features)
        action_log_std = self.policy_log_std.expand_as(action_mean)
        
        # Value output
        value = self.value_head(shared_features)
        
        return action_mean, action_log_std, value
    
    def get_action(self, 
                   obs: torch.Tensor, 
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action from policy.
        
        Args:
            obs: Observation tensor
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        action_mean, action_log_std, value = self.forward(obs)
        
        if deterministic:
            action = action_mean
            log_prob = None
        else:
            action_std = torch.exp(action_log_std)
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob, value


class PPO(BaseAlgorithm):
    """
    Proximal Policy Optimization (PPO) algorithm.
    
    PPO is a policy gradient method that uses a clipped objective function
    to ensure stable policy updates. This implementation includes:
    - Actor-critic architecture
    - Generalized Advantage Estimation (GAE)
    - Clipped surrogate objective
    - Value function loss
    - Entropy bonus
    """
    
    def __init__(self,
                 env,
                 learning_rate: float = 3e-4,
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: float = 0.2,
                 ent_coef: float = 0.0,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 target_kl: Optional[float] = None,
                 device: str = "auto",
                 seed: Optional[int] = None,
                 verbose: int = 1,
                 **kwargs):
        """
        Initialize PPO algorithm.
        
        Args:
            env: Environment to train on
            learning_rate: Learning rate for optimization
            n_steps: Number of steps to collect per update
            batch_size: Batch size for training
            n_epochs: Number of training epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clipping parameter
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Maximum gradient norm for clipping
            target_kl: Target KL divergence for early stopping
            device: Device to run on
            seed: Random seed
            verbose: Verbosity level
        """
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        
        super().__init__(env, learning_rate, device, seed, verbose)
    
    def _setup_algorithm(self):
        """Setup PPO-specific components."""
        # Get environment dimensions
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        # Create policy network
        self.policy = PPOPolicy(obs_dim, action_dim).to(self.device)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        
        # Training statistics
        self.num_timesteps = 0
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        # Rollout buffer
        self.rollout_buffer = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "log_probs": [],
            "dones": [],
            "advantages": [],
            "returns": []
        }
    
    def _collect_rollouts(self) -> Dict[str, float]:
        """Collect rollout data."""
        # Reset buffer
        for key in self.rollout_buffer:
            self.rollout_buffer[key].clear()
        
        # Reset environment
        obs, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        # Collect rollouts
        for step in range(self.n_steps):
            # Get action from policy
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action, log_prob, value = self.policy.get_action(obs_tensor)
                
                action_np = action.cpu().numpy().flatten()
                log_prob_np = log_prob.cpu().numpy().flatten() if log_prob is not None else np.array([0.0])
                value_np = value.cpu().numpy().flatten()
            
            # Take step in environment
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated
            
            # Store transition
            self.rollout_buffer["observations"].append(obs.copy())
            self.rollout_buffer["actions"].append(action_np.copy())
            self.rollout_buffer["rewards"].append(reward)
            self.rollout_buffer["values"].append(value_np[0])
            self.rollout_buffer["log_probs"].append(log_prob_np[0])
            self.rollout_buffer["dones"].append(done)
            
            # Update episode tracking
            episode_reward += reward
            episode_length += 1
            
            # Handle episode end
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                if self.verbose >= 2:
                    print(f"Episode {len(self.episode_rewards)}: "
                          f"reward={episode_reward:.2f}, length={episode_length}")
                
                obs, _ = self.env.reset()
                episode_reward = 0.0
                episode_length = 0
            else:
                obs = next_obs
        
        # Calculate statistics
        stats = {
            "mean_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            "mean_length": np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
            "total_timesteps": self.n_steps
        }
        
        return stats
    
    def _compute_gae(self, last_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation."""
        rewards = np.array(self.rollout_buffer["rewards"])
        values = np.array(self.rollout_buffer["values"] + [last_value])
        dones = np.array(self.rollout_buffer["dones"])
        
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        
        # Compute GAE backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values[:-1]
        
        return advantages, returns
    
    def _update_policy(self) -> Dict[str, float]:
        """Update policy using PPO objective."""
        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(self.rollout_buffer["observations"])).to(self.device)
        actions_tensor = torch.FloatTensor(np.array(self.rollout_buffer["actions"])).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(np.array(self.rollout_buffer["log_probs"])).to(self.device)
        advantages_tensor = torch.FloatTensor(self.rollout_buffer["advantages"]).to(self.device)
        returns_tensor = torch.FloatTensor(self.rollout_buffer["returns"]).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Training statistics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        kl_divergences = []
        
        # Training epochs
        for epoch in range(self.n_epochs):
            # Create batches
            batch_indices = np.random.permutation(len(obs_tensor))
            
            for start_idx in range(0, len(obs_tensor), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(obs_tensor))
                batch_indices_epoch = batch_indices[start_idx:end_idx]
                
                # Get batch data
                obs_batch = obs_tensor[batch_indices_epoch]
                actions_batch = actions_tensor[batch_indices_epoch]
                old_log_probs_batch = old_log_probs_tensor[batch_indices_epoch]
                advantages_batch = advantages_tensor[batch_indices_epoch]
                returns_batch = returns_tensor[batch_indices_epoch]
                
                # Forward pass
                action_mean, action_log_std, values = self.policy(obs_batch)
                
                # Compute log probabilities
                action_std = torch.exp(action_log_std)
                dist = Normal(action_mean, action_std)
                log_probs = dist.log_prob(actions_batch).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1)
                
                # Compute ratios
                ratios = torch.exp(log_probs - old_log_probs_batch)
                
                # Compute surrogate losses
                surr1 = ratios * advantages_batch
                surr2 = torch.clamp(ratios, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value function loss
                value_loss = nn.MSELoss()(values.squeeze(), returns_batch)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Store losses
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                
                # Compute KL divergence for early stopping
                with torch.no_grad():
                    kl_div = (old_log_probs_batch - log_probs).mean()
                    kl_divergences.append(kl_div.item())
                
                # Early stopping if KL divergence is too high
                if self.target_kl is not None and kl_div > 1.5 * self.target_kl:
                    if self.verbose >= 2:
                        print(f"Early stopping at epoch {epoch} due to high KL divergence")
                    break
        
        # Return training statistics
        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy_loss": np.mean(entropy_losses),
            "kl_divergence": np.mean(kl_divergences),
            "explained_variance": self._explained_variance(
                np.array(self.rollout_buffer["values"]),
                np.array(self.rollout_buffer["returns"])
            )
        }
    
    def _explained_variance(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute explained variance."""
        var_y = np.var(y_true)
        return 1 - np.var(y_true - y_pred) / var_y if var_y > 0 else 0.0
    
    def learn(self, 
              total_timesteps: int,
              callback=None,
              log_interval: int = 4,
              eval_env=None,
              eval_freq: int = -1,
              n_eval_episodes: int = 5,
              tb_log_name: str = "PPO",
              reset_num_timesteps: bool = True,
              progress_bar: bool = True) -> "PPO":
        """
        Train the PPO algorithm.
        
        Args:
            total_timesteps: Total number of timesteps to train
            callback: Callback function called during training
            log_interval: Log every log_interval updates
            eval_env: Environment for evaluation
            eval_freq: Evaluate every eval_freq timesteps (-1 to disable)
            n_eval_episodes: Number of episodes for evaluation
            tb_log_name: TensorBoard log name
            reset_num_timesteps: Whether to reset timestep counter
            progress_bar: Whether to show progress bar
            
        Returns:
            Self for method chaining
        """
        if reset_num_timesteps:
            self.num_timesteps = 0
        
        # Training loop
        num_updates = total_timesteps // self.n_steps
        
        if progress_bar:
            pbar = tqdm(range(num_updates), desc="Training PPO")
        else:
            pbar = range(num_updates)
        
        for update in pbar:
            # Collect rollouts
            rollout_stats = self._collect_rollouts()
            
            # Compute advantages and returns
            last_value = 0.0  # Could be improved by getting value of last state
            advantages, returns = self._compute_gae(last_value)
            
            # Store in buffer
            self.rollout_buffer["advantages"] = advantages
            self.rollout_buffer["returns"] = returns
            
            # Update policy
            train_stats = self._update_policy()
            
            # Update timestep counter
            self.num_timesteps += self.n_steps
            
            # Logging
            if update % log_interval == 0:
                if self.verbose >= 1:
                    print(f"Update {update}/{num_updates}")
                    print(f"  Mean reward: {rollout_stats['mean_reward']:.2f}")
                    print(f"  Mean length: {rollout_stats['mean_length']:.2f}")
                    print(f"  Policy loss: {train_stats['policy_loss']:.4f}")
                    print(f"  Value loss: {train_stats['value_loss']:.4f}")
                    print(f"  Entropy loss: {train_stats['entropy_loss']:.4f}")
                    print(f"  KL divergence: {train_stats['kl_divergence']:.4f}")
                    print(f"  Explained variance: {train_stats['explained_variance']:.4f}")
            
            # Evaluation
            if eval_env is not None and eval_freq > 0 and self.num_timesteps % eval_freq == 0:
                eval_stats = self.evaluate(eval_env, n_eval_episodes)
                if self.verbose >= 1:
                    print(f"  Eval mean reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
            
            # Callback
            if callback is not None:
                callback(locals(), globals())
            
            # Update progress bar
            if progress_bar:
                pbar.set_postfix({
                    "reward": f"{rollout_stats['mean_reward']:.2f}",
                    "policy_loss": f"{train_stats['policy_loss']:.4f}",
                    "value_loss": f"{train_stats['value_loss']:.4f}"
                })
        
        return self
    
    def predict(self, 
                observation: np.ndarray, 
                deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict action for given observation.
        
        Args:
            observation: Current observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, value)
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            action, _, value = self.policy.get_action(obs_tensor, deterministic=deterministic)
            
            action_np = action.cpu().numpy().flatten()
            value_np = value.cpu().numpy().flatten() if value is not None else None
            
            return action_np, value_np
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get algorithm parameters."""
        return {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "target_kl": self.target_kl,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_timesteps": self.num_timesteps
        }
    
    def set_parameters(self, parameters: Dict[str, Any]):
        """Set algorithm parameters."""
        self.learning_rate = parameters["learning_rate"]
        self.n_steps = parameters["n_steps"]
        self.batch_size = parameters["batch_size"]
        self.n_epochs = parameters["n_epochs"]
        self.gamma = parameters["gamma"]
        self.gae_lambda = parameters["gae_lambda"]
        self.clip_range = parameters["clip_range"]
        self.ent_coef = parameters["ent_coef"]
        self.vf_coef = parameters["vf_coef"]
        self.max_grad_norm = parameters["max_grad_norm"]
        self.target_kl = parameters["target_kl"]
        
        # Load model states
        if "policy_state_dict" in parameters:
            self.policy.load_state_dict(parameters["policy_state_dict"])
        if "optimizer_state_dict" in parameters:
            self.optimizer.load_state_dict(parameters["optimizer_state_dict"])
        if "num_timesteps" in parameters:
            self.num_timesteps = parameters["num_timesteps"]
