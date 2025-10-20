"""
PPO with Entropy Weighting implementation.

This module provides an enhanced version of PPO that uses adaptive entropy
coefficient to encourage exploration during training. The entropy coefficient
is automatically adjusted based on the current policy's entropy.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional, Tuple
from collections import deque

from .ppo import PPO, PPOPolicy


class PPOEntropy(PPO):
    """
    PPO with adaptive entropy weighting.
    
    This algorithm extends standard PPO by automatically adjusting the entropy
    coefficient based on the current policy's entropy. This helps maintain
    exploration throughout training and can lead to better performance.
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
                 ent_coef: float = 0.01,  # Initial entropy coefficient
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 target_kl: Optional[float] = None,
                 target_entropy: Optional[float] = None,
                 ent_coef_lr: float = 0.001,  # Learning rate for entropy coefficient
                 ent_coef_decay: float = 0.999,  # Decay rate for entropy coefficient
                 min_ent_coef: float = 0.001,  # Minimum entropy coefficient
                 max_ent_coef: float = 1.0,  # Maximum entropy coefficient
                 device: str = "auto",
                 seed: Optional[int] = None,
                 verbose: int = 1,
                 **kwargs):
        """
        Initialize PPO with entropy weighting.
        
        Args:
            env: Environment to train on
            learning_rate: Learning rate for optimization
            n_steps: Number of steps to collect per update
            batch_size: Batch size for training
            n_epochs: Number of training epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clipping parameter
            ent_coef: Initial entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Maximum gradient norm for clipping
            target_kl: Target KL divergence for early stopping
            target_entropy: Target entropy for adaptive coefficient
            ent_coef_lr: Learning rate for entropy coefficient
            ent_coef_decay: Decay rate for entropy coefficient
            min_ent_coef: Minimum entropy coefficient
            max_ent_coef: Maximum entropy coefficient
            device: Device to run on
            seed: Random seed
            verbose: Verbosity level
        """
        # Store entropy-specific parameters
        self.target_entropy = target_entropy
        self.ent_coef_lr = ent_coef_lr
        self.ent_coef_decay = ent_coef_decay
        self.min_ent_coef = min_ent_coef
        self.max_ent_coef = max_ent_coef
        
        # Initialize base PPO
        super().__init__(
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            device=device,
            seed=seed,
            verbose=verbose,
            **kwargs
        )
        
        # Set target entropy if not provided
        if self.target_entropy is None:
            # Use negative action dimension as target entropy
            self.target_entropy = -self.env.action_space.shape[0]
        
        # Entropy coefficient tracking
        self.ent_coef_history = deque(maxlen=1000)
        self.entropy_history = deque(maxlen=1000)
        
        # Create entropy coefficient optimizer
        self.ent_coef_optimizer = optim.Adam([self.ent_coef_tensor], lr=self.ent_coef_lr)
    
    def _setup_algorithm(self):
        """Setup PPO with entropy weighting specific components."""
        # Call parent setup
        super()._setup_algorithm()
        
        # Create learnable entropy coefficient
        self.ent_coef_tensor = torch.tensor(self.ent_coef, dtype=torch.float32, requires_grad=True)
        self.ent_coef_tensor = self.ent_coef_tensor.to(self.device)
    
    def _update_entropy_coefficient(self, current_entropy: float) -> float:
        """
        Update the entropy coefficient based on current policy entropy.
        
        Args:
            current_entropy: Current policy entropy
            
        Returns:
            Updated entropy coefficient
        """
        # Compute entropy coefficient loss
        ent_coef_loss = -self.ent_coef_tensor * (current_entropy - self.target_entropy)
        
        # Update entropy coefficient
        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        self.ent_coef_optimizer.step()
        
        # Clip entropy coefficient to valid range
        with torch.no_grad():
            self.ent_coef_tensor.clamp_(self.min_ent_coef, self.max_ent_coef)
        
        # Get updated coefficient
        updated_ent_coef = self.ent_coef_tensor.item()
        
        # Store history
        self.ent_coef_history.append(updated_ent_coef)
        self.entropy_history.append(current_entropy)
        
        return updated_ent_coef
    
    def _update_policy(self) -> Dict[str, float]:
        """Update policy using PPO objective with adaptive entropy weighting."""
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
        ent_coefs = []
        
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
                
                # Compute log probabilities and entropy
                action_std = torch.exp(action_log_std)
                dist = torch.distributions.Normal(action_mean, action_std)
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
                
                # Entropy loss with current coefficient
                current_ent_coef = self.ent_coef_tensor.item()
                entropy_loss = -current_ent_coef * entropy.mean()
                
                # Total loss
                total_loss = policy_loss + self.vf_coef * value_loss + entropy_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Update entropy coefficient
                current_entropy = entropy.mean().item()
                updated_ent_coef = self._update_entropy_coefficient(current_entropy)
                
                # Store losses
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                kl_divergences.append((old_log_probs_batch - log_probs).mean().item())
                ent_coefs.append(updated_ent_coef)
                
                # Early stopping if KL divergence is too high
                if self.target_kl is not None and (old_log_probs_batch - log_probs).mean() > 1.5 * self.target_kl:
                    if self.verbose >= 2:
                        print(f"Early stopping at epoch {epoch} due to high KL divergence")
                    break
        
        # Return training statistics
        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy_loss": np.mean(entropy_losses),
            "kl_divergence": np.mean(kl_divergences),
            "entropy_coefficient": np.mean(ent_coefs),
            "current_entropy": current_entropy,
            "target_entropy": self.target_entropy,
            "explained_variance": self._explained_variance(
                np.array(self.rollout_buffer["values"]),
                np.array(self.rollout_buffer["returns"])
            )
        }
    
    def learn(self, 
              total_timesteps: int,
              callback=None,
              log_interval: int = 4,
              eval_env=None,
              eval_freq: int = -1,
              n_eval_episodes: int = 5,
              tb_log_name: str = "PPO_Entropy",
              reset_num_timesteps: bool = True,
              progress_bar: bool = True) -> "PPOEntropy":
        """
        Train the PPO with entropy weighting algorithm.
        
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
            pbar = tqdm(range(num_updates), desc="Training PPO with Entropy Weighting")
        else:
            pbar = range(num_updates)
        
        for update in pbar:
            # Collect rollouts
            rollout_stats = self._collect_rollouts()
            
            # Compute advantages and returns
            last_value = 0.0
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
                    print(f"  Entropy coefficient: {train_stats['entropy_coefficient']:.4f}")
                    print(f"  Current entropy: {train_stats['current_entropy']:.4f}")
                    print(f"  Target entropy: {train_stats['target_entropy']:.4f}")
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
                    "ent_coef": f"{train_stats['entropy_coefficient']:.4f}",
                    "entropy": f"{train_stats['current_entropy']:.4f}"
                })
        
        return self
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get algorithm parameters."""
        base_params = super().get_parameters()
        
        # Add entropy-specific parameters
        entropy_params = {
            "target_entropy": self.target_entropy,
            "ent_coef_lr": self.ent_coef_lr,
            "ent_coef_decay": self.ent_coef_decay,
            "min_ent_coef": self.min_ent_coef,
            "max_ent_coef": self.max_ent_coef,
            "ent_coef_tensor": self.ent_coef_tensor.item(),
            "ent_coef_optimizer_state_dict": self.ent_coef_optimizer.state_dict(),
            "ent_coef_history": list(self.ent_coef_history),
            "entropy_history": list(self.entropy_history)
        }
        
        return {**base_params, **entropy_params}
    
    def set_parameters(self, parameters: Dict[str, Any]):
        """Set algorithm parameters."""
        # Set base parameters
        super().set_parameters(parameters)
        
        # Set entropy-specific parameters
        if "target_entropy" in parameters:
            self.target_entropy = parameters["target_entropy"]
        if "ent_coef_lr" in parameters:
            self.ent_coef_lr = parameters["ent_coef_lr"]
        if "ent_coef_decay" in parameters:
            self.ent_coef_decay = parameters["ent_coef_decay"]
        if "min_ent_coef" in parameters:
            self.min_ent_coef = parameters["min_ent_coef"]
        if "max_ent_coef" in parameters:
            self.max_ent_coef = parameters["max_ent_coef"]
        if "ent_coef_tensor" in parameters:
            self.ent_coef_tensor.data = torch.tensor(parameters["ent_coef_tensor"], dtype=torch.float32)
        if "ent_coef_optimizer_state_dict" in parameters:
            self.ent_coef_optimizer.load_state_dict(parameters["ent_coef_optimizer_state_dict"])
        if "ent_coef_history" in parameters:
            self.ent_coef_history = deque(parameters["ent_coef_history"], maxlen=1000)
        if "entropy_history" in parameters:
            self.entropy_history = deque(parameters["entropy_history"], maxlen=1000)
    
    def get_entropy_coefficient_history(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get entropy coefficient and entropy history.
        
        Returns:
            Tuple of (entropy_coefficients, entropies)
        """
        return np.array(self.ent_coef_history), np.array(self.entropy_history)
    
    def plot_entropy_adaptation(self, save_path: Optional[str] = None):
        """
        Plot entropy coefficient adaptation over time.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        import matplotlib.pyplot as plt
        
        ent_coefs, entropies = self.get_entropy_coefficient_history()
        
        if len(ent_coefs) == 0:
            print("No entropy coefficient history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot entropy coefficient
        ax1.plot(ent_coefs, label='Entropy Coefficient')
        ax1.axhline(y=self.target_entropy, color='r', linestyle='--', label=f'Target Entropy ({self.target_entropy:.2f})')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Entropy Coefficient')
        ax1.set_title('Entropy Coefficient Adaptation')
        ax1.legend()
        ax1.grid(True)
        
        # Plot policy entropy
        ax2.plot(entropies, label='Policy Entropy', color='green')
        ax2.axhline(y=self.target_entropy, color='r', linestyle='--', label=f'Target Entropy ({self.target_entropy:.2f})')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Policy Entropy')
        ax2.set_title('Policy Entropy Over Time')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
