# Reinforcement Learning Algorithms for MuJoCo Hopper

This repository contains a comprehensive implementation of various reinforcement learning algorithms for training on the MuJoCo Hopper environment. The project focuses on modular, clean implementations that can be easily extended and compared.

## 🎯 Project Overview

The MuJoCo Hopper is a classic continuous control task where an agent must learn to hop forward as far as possible without falling. This environment is ideal for testing and comparing different RL algorithms due to its:

- **Continuous action space**: Joint torques for the hopper's actuators
- **High-dimensional state space**: Joint positions, velocities, and contact information
- **Challenging dynamics**: Requires learning balance and forward momentum
- **Clear success metrics**: Distance traveled and episode length

## 🚀 Features

- **Multiple RL Algorithms**: PPO, PPO with entropy weighting, and more
- **Modular Design**: Clean separation of environments, algorithms, and training
- **Easy Experimentation**: Simple configuration files and training scripts
- **Comprehensive Evaluation**: Metrics, visualization, and comparison tools
- **Reproducible Results**: Fixed seeds and consistent evaluation protocols

## 📁 Project Structure

```
rl-algos/
├── environments/          # Environment implementations
│   ├── __init__.py
│   ├── hopper_env.py     # MuJoCo Hopper environment
│   └── wrappers.py       # Gym wrappers and utilities
├── algorithms/           # RL algorithm implementations
│   ├── __init__.py
│   ├── ppo.py           # PPO implementation
│   ├── ppo_entropy.py   # PPO with entropy weighting
│   └── base.py          # Base algorithm class
├── training/            # Training scripts and utilities
│   ├── __init__.py
│   ├── train_ppo.py     # PPO training script
│   ├── train_ppo_entropy.py  # PPO with entropy training
│   └── utils.py         # Training utilities
├── evaluation/          # Evaluation and visualization
│   ├── __init__.py
│   ├── evaluate.py      # Evaluation scripts
│   └── visualize.py     # Visualization tools
├── configs/             # Configuration files
│   ├── ppo_config.yaml
│   └── ppo_entropy_config.yaml
├── results/             # Training results and logs
├── environment.yaml     # Conda environment specification
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Conda or Miniconda
- MuJoCo 2.3.0+

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd rl-algos
   ```

2. **Create conda environment**:
   ```bash
   conda env create -f environment.yaml
   conda activate rl-algos
   ```

3. **Install MuJoCo** (if not already installed):
   ```bash
   pip install mujoco
   ```

## 🎮 Usage

### Quick Start

Train a PPO agent on the Hopper environment:

```bash
python training/train_ppo.py --config configs/ppo_config.yaml
```

### Available Algorithms

- **PPO (Proximal Policy Optimization)**: Standard PPO implementation
- **PPO with Entropy Weighting**: PPO with adaptive entropy coefficient
- **More algorithms coming soon...**

### Training Examples

```bash
# Train PPO
python training/train_ppo.py --config configs/ppo_config.yaml --seed 42

# Train PPO with entropy weighting
python training/train_ppo_entropy.py --config configs/ppo_entropy_config.yaml --seed 42

# Evaluate trained model
python evaluation/evaluate.py --model_path results/ppo_hopper_model.pkl
```

## 📊 Results and Evaluation

The repository includes comprehensive evaluation tools:

- **Performance Metrics**: Episode rewards, success rate, episode length
- **Training Curves**: Learning progress visualization
- **Policy Analysis**: Action distributions and value function analysis
- **Comparison Tools**: Side-by-side algorithm comparison

## 🔬 Experiments

### Algorithm Comparison

Compare different algorithms on the Hopper task:

```bash
# Run all experiments
python experiments/run_comparison.py --algorithms ppo ppo_entropy --seeds 0 1 2 3 4
```

### Hyperparameter Tuning

```bash
# Grid search for PPO hyperparameters
python experiments/hyperparameter_search.py --algorithm ppo --param_grid configs/ppo_grid.yaml
```

## 📈 Expected Results

On the MuJoCo Hopper environment:
- **PPO**: Should achieve ~2000+ average reward over 100 episodes
- **PPO with Entropy Weighting**: May show improved exploration and stability
- **Training Time**: ~1-2 hours on modern GPU for 1M timesteps

## 🤝 Contributing

Contributions are welcome! Please see our contributing guidelines for:
- Adding new algorithms
- Improving existing implementations
- Adding new environments
- Enhancing evaluation tools

## 📚 References

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [OpenAI Gym](https://gym.openai.com/)
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
