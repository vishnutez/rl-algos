"""
Training utilities and helper functions.

This module provides various utilities for training RL algorithms,
including logging, monitoring, and evaluation tools.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import json
import os
from pathlib import Path
import time
from collections import defaultdict, deque


class TrainingMonitor:
    """
    Monitor training progress and log statistics.
    
    This class provides comprehensive monitoring of training progress,
    including reward tracking, loss monitoring, and performance metrics.
    """
    
    def __init__(self, 
                 log_dir: str = "logs",
                 save_freq: int = 100,
                 plot_freq: int = 1000,
                 verbose: int = 1):
        """
        Initialize training monitor.
        
        Args:
            log_dir: Directory to save logs
            save_freq: Frequency to save logs (in updates)
            plot_freq: Frequency to generate plots (in updates)
            verbose: Verbosity level
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_freq = save_freq
        self.plot_freq = plot_freq
        self.verbose = verbose
        
        # Training statistics
        self.stats = defaultdict(list)
        self.episode_rewards = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        self.training_losses = defaultdict(list)
        
        # Performance tracking
        self.best_reward = -np.inf
        self.best_model_path = None
        
        # Timing
        self.start_time = time.time()
        self.last_save_time = time.time()
    
    def log_episode(self, 
                   episode_reward: float, 
                   episode_length: int, 
                   episode_num: int):
        """Log episode statistics."""
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        # Update best reward
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            if self.verbose >= 1:
                print(f"New best reward: {self.best_reward:.2f} (Episode {episode_num})")
    
    def log_training_step(self, 
                         update: int, 
                         losses: Dict[str, float],
                         additional_stats: Optional[Dict[str, float]] = None):
        """Log training step statistics."""
        # Store losses
        for key, value in losses.items():
            self.training_losses[key].append(value)
        
        # Store additional statistics
        if additional_stats:
            for key, value in additional_stats.items():
                self.stats[key].append(value)
        
        # Log to console
        if self.verbose >= 1 and update % 10 == 0:
            print(f"Update {update}:")
            for key, value in losses.items():
                print(f"  {key}: {value:.4f}")
            if additional_stats:
                for key, value in additional_stats.items():
                    print(f"  {key}: {value:.4f}")
    
    def log_evaluation(self, 
                      eval_stats: Dict[str, float], 
                      update: int):
        """Log evaluation statistics."""
        for key, value in eval_stats.items():
            self.stats[f"eval_{key}"].append(value)
        
        if self.verbose >= 1:
            print(f"Evaluation (Update {update}):")
            print(f"  Mean reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
            print(f"  Mean length: {eval_stats['mean_length']:.2f} ± {eval_stats['std_length']:.2f}")
    
    def save_logs(self, update: int):
        """Save training logs to disk."""
        if update % self.save_freq == 0:
            # Save statistics
            stats_path = self.log_dir / f"training_stats_{update}.json"
            with open(stats_path, 'w') as f:
                json.dump({
                    "episode_rewards": list(self.episode_rewards),
                    "episode_lengths": list(self.episode_lengths),
                    "training_losses": dict(self.training_losses),
                    "stats": dict(self.stats),
                    "best_reward": self.best_reward,
                    "update": update
                }, f, indent=2)
            
            if self.verbose >= 1:
                print(f"Training logs saved to {stats_path}")
    
    def plot_training_curves(self, 
                            update: int, 
                            save_path: Optional[str] = None):
        """Plot training curves."""
        if update % self.plot_freq == 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Episode rewards
            if self.episode_rewards:
                axes[0, 0].plot(self.episode_rewards)
                axes[0, 0].set_title('Episode Rewards')
                axes[0, 0].set_xlabel('Episode')
                axes[0, 0].set_ylabel('Reward')
                axes[0, 0].grid(True)
                
                # Add moving average
                if len(self.episode_rewards) > 10:
                    window = min(50, len(self.episode_rewards) // 10)
                    moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
                    axes[0, 0].plot(range(window-1, len(self.episode_rewards)), moving_avg, 
                                   color='red', linewidth=2, label=f'Moving Avg ({window})')
                    axes[0, 0].legend()
            
            # Episode lengths
            if self.episode_lengths:
                axes[0, 1].plot(self.episode_lengths)
                axes[0, 1].set_title('Episode Lengths')
                axes[0, 1].set_xlabel('Episode')
                axes[0, 1].set_ylabel('Length')
                axes[0, 1].grid(True)
            
            # Training losses
            if self.training_losses:
                for i, (loss_name, losses) in enumerate(self.training_losses.items()):
                    if losses:
                        axes[1, 0].plot(losses, label=loss_name)
                axes[1, 0].set_title('Training Losses')
                axes[1, 0].set_xlabel('Update')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            # Evaluation rewards
            if 'eval_mean_reward' in self.stats:
                axes[1, 1].plot(self.stats['eval_mean_reward'])
                axes[1, 1].set_title('Evaluation Rewards')
                axes[1, 1].set_xlabel('Update')
                axes[1, 1].set_ylabel('Mean Reward')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Training curves saved to {save_path}")
            else:
                plt.show()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        summary = {
            "total_episodes": len(self.episode_rewards),
            "best_reward": self.best_reward,
            "mean_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            "std_reward": np.std(self.episode_rewards) if self.episode_rewards else 0.0,
            "mean_length": np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
            "std_length": np.std(self.episode_lengths) if self.episode_lengths else 0.0,
            "training_time": time.time() - self.start_time
        }
        
        return summary


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting.
    
    This class monitors training progress and stops training when
    performance stops improving.
    """
    
    def __init__(self, 
                 patience: int = 100,
                 min_delta: float = 0.01,
                 monitor: str = "mean_reward",
                 mode: str = "max"):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of updates to wait before stopping
            min_delta: Minimum change to qualify as improvement
            monitor: Metric to monitor
            mode: "max" for maximizing, "min" for minimizing
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        
        self.best_score = -np.inf if mode == "max" else np.inf
        self.wait = 0
        self.stopped_epoch = 0
    
    def __call__(self, metrics: Dict[str, float], epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            metrics: Dictionary of current metrics
            epoch: Current epoch
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.monitor not in metrics:
            return False
        
        current_score = metrics[self.monitor]
        
        if self.mode == "max":
            improved = current_score > self.best_score + self.min_delta
        else:
            improved = current_score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = current_score
            self.wait = 0
        else:
            self.wait += 1
        
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            return True
        
        return False


def create_training_callback(monitor: TrainingMonitor, 
                           early_stopping: Optional[EarlyStopping] = None):
    """
    Create a training callback function.
    
    Args:
        monitor: Training monitor instance
        early_stopping: Early stopping instance (optional)
        
    Returns:
        Callback function
    """
    def callback(locals_dict, globals_dict):
        update = locals_dict.get('update', 0)
        
        # Log training step
        if 'train_stats' in locals_dict:
            monitor.log_training_step(update, locals_dict['train_stats'])
        
        # Log evaluation
        if 'eval_stats' in locals_dict:
            monitor.log_evaluation(locals_dict['eval_stats'], update)
        
        # Save logs
        monitor.save_logs(update)
        
        # Plot training curves
        if update % monitor.plot_freq == 0:
            plot_path = monitor.log_dir / f"training_curves_{update}.png"
            monitor.plot_training_curves(update, str(plot_path))
        
        # Check early stopping
        if early_stopping is not None:
            if 'rollout_stats' in locals_dict:
                metrics = {
                    'mean_reward': locals_dict['rollout_stats'].get('mean_reward', 0.0)
                }
                if early_stopping(metrics, update):
                    print(f"Early stopping triggered at update {update}")
                    return False
        
        return True
    
    return callback


def save_model(algorithm, path: str, update: int = None):
    """
    Save algorithm model to disk.
    
    Args:
        algorithm: Algorithm instance to save
        path: Path to save the model
        update: Current update number (optional)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add update number to filename if provided
    if update is not None:
        stem = path.stem
        suffix = path.suffix
        path = path.parent / f"{stem}_update_{update}{suffix}"
    
    algorithm.save(str(path))
    print(f"Model saved to {path}")


def load_model(algorithm, path: str):
    """
    Load algorithm model from disk.
    
    Args:
        algorithm: Algorithm instance to load into
        path: Path to load the model from
    """
    algorithm.load(path)
    print(f"Model loaded from {path}")


def compare_algorithms(algorithms: Dict[str, Any], 
                      eval_env,
                      n_eval_episodes: int = 10,
                      render: bool = False) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple algorithms on the same environment.
    
    Args:
        algorithms: Dictionary of {name: algorithm} pairs
        eval_env: Environment to evaluate on
        n_eval_episodes: Number of episodes for evaluation
        render: Whether to render during evaluation
        
    Returns:
        Dictionary of evaluation results for each algorithm
    """
    results = {}
    
    for name, algorithm in algorithms.items():
        print(f"Evaluating {name}...")
        
        eval_stats = algorithm.evaluate(
            eval_env, 
            n_eval_episodes=n_eval_episodes, 
            deterministic=True, 
            render=render
        )
        
        results[name] = eval_stats
        
        print(f"  Mean reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
        print(f"  Mean length: {eval_stats['mean_length']:.2f} ± {eval_stats['std_length']:.2f}")
    
    return results


def plot_algorithm_comparison(results: Dict[str, Dict[str, float]], 
                            save_path: Optional[str] = None):
    """
    Plot comparison of multiple algorithms.
    
    Args:
        results: Results from compare_algorithms
        save_path: Path to save the plot (optional)
    """
    algorithms = list(results.keys())
    mean_rewards = [results[alg]['mean_reward'] for alg in algorithms]
    std_rewards = [results[alg]['std_reward'] for alg in algorithms]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Mean rewards comparison
    bars1 = ax1.bar(algorithms, mean_rewards, yerr=std_rewards, capsize=5)
    ax1.set_title('Mean Episode Rewards')
    ax1.set_ylabel('Reward')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars1, mean_rewards, std_rewards):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                f'{mean:.2f}±{std:.2f}', ha='center', va='bottom')
    
    # Mean lengths comparison
    mean_lengths = [results[alg]['mean_length'] for alg in algorithms]
    std_lengths = [results[alg]['std_length'] for alg in algorithms]
    
    bars2 = ax2.bar(algorithms, mean_lengths, yerr=std_lengths, capsize=5, color='orange')
    ax2.set_title('Mean Episode Lengths')
    ax2.set_ylabel('Length')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars2, mean_lengths, std_lengths):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
                f'{mean:.2f}±{std:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()
