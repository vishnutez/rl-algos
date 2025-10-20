"""
Base algorithm class for RL implementations.

This module provides the base class that all RL algorithms should inherit from,
ensuring a consistent interface for training, evaluation, and model saving/loading.
"""

import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import pickle
import os
from pathlib import Path


class BaseAlgorithm(ABC):
    """
    Abstract base class for RL algorithms.
    
    This class defines the interface that all RL algorithms must implement,
    providing a consistent API for training, evaluation, and model management.
    """
    
    def __init__(self, 
                 env,
                 learning_rate: float = 3e-4,
                 device: str = "auto",
                 seed: Optional[int] = None,
                 verbose: int = 1):
        """
        Initialize the base algorithm.
        
        Args:
            env: Environment to train on
            learning_rate: Learning rate for optimization
            device: Device to run on ("auto", "cpu", "cuda")
            seed: Random seed for reproducibility
            verbose: Verbosity level (0, 1, 2)
        """
        self.env = env
        self.learning_rate = learning_rate
        self.verbose = verbose
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Set random seeds
        if seed is not None:
            self.set_seed(seed)
        
        # Initialize algorithm-specific components
        self._setup_algorithm()
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    @abstractmethod
    def _setup_algorithm(self):
        """Setup algorithm-specific components (networks, optimizers, etc.)."""
        pass
    
    @abstractmethod
    def learn(self, 
              total_timesteps: int,
              callback=None,
              log_interval: int = 4,
              eval_env=None,
              eval_freq: int = -1,
              n_eval_episodes: int = 5,
              tb_log_name: str = "run",
              reset_num_timesteps: bool = True,
              progress_bar: bool = True) -> "BaseAlgorithm":
        """
        Train the algorithm.
        
        Args:
            total_timesteps: Total number of timesteps to train
            callback: Callback function called during training
            log_interval: Log every log_interval timesteps
            eval_env: Environment for evaluation
            eval_freq: Evaluate every eval_freq timesteps (-1 to disable)
            n_eval_episodes: Number of episodes for evaluation
            tb_log_name: TensorBoard log name
            reset_num_timesteps: Whether to reset timestep counter
            progress_bar: Whether to show progress bar
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, 
                observation: np.ndarray, 
                deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict action for given observation.
        
        Args:
            observation: Current observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, value) where value may be None
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get algorithm parameters."""
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]):
        """Set algorithm parameters."""
        pass
    
    def save(self, path: str):
        """
        Save the algorithm to disk.
        
        Args:
            path: Path to save the model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save algorithm state
        save_dict = {
            "algorithm_class": self.__class__.__name__,
            "parameters": self.get_parameters(),
            "env": self.env,
        }
        
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)
        
        if self.verbose >= 1:
            print(f"Model saved to {path}")
    
    def load(self, path: str):
        """
        Load the algorithm from disk.
        
        Args:
            path: Path to load the model from
        """
        with open(path, "rb") as f:
            save_dict = pickle.load(f)
        
        # Verify algorithm class
        if save_dict["algorithm_class"] != self.__class__.__name__:
            raise ValueError(f"Algorithm class mismatch: expected {self.__class__.__name__}, "
                           f"got {save_dict['algorithm_class']}")
        
        # Set parameters
        self.set_parameters(save_dict["parameters"])
        
        if self.verbose >= 1:
            print(f"Model loaded from {path}")
    
    def evaluate(self, 
                 eval_env,
                 n_eval_episodes: int = 10,
                 deterministic: bool = True,
                 render: bool = False) -> Dict[str, float]:
        """
        Evaluate the algorithm on the given environment.
        
        Args:
            eval_env: Environment to evaluate on
            n_eval_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic policy
            render: Whether to render during evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_eval_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
                
                if render:
                    eval_env.render()
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        # Calculate statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        std_length = np.std(episode_lengths)
        
        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_length": mean_length,
            "std_length": std_length,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths
        }
    
    def get_env(self):
        """Get the training environment."""
        return self.env
    
    def set_env(self, env):
        """Set a new training environment."""
        self.env = env
