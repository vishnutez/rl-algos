#!/usr/bin/env python3
"""
Setup script for the RL Algorithms project.

This script helps set up the environment and install dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False

def check_conda():
    """Check if conda is available."""
    try:
        result = subprocess.run("conda --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Conda found: {result.stdout.strip()}")
            return True
    except:
        pass
    
    print("✗ Conda not found. Please install Anaconda or Miniconda first.")
    return False

def setup_environment():
    """Set up the conda environment."""
    print("Setting up RL Algorithms environment...")
    print("=" * 50)
    
    # Check if conda is available
    if not check_conda():
        print("\nPlease install Anaconda or Miniconda first:")
        print("  https://docs.conda.io/en/latest/miniconda.html")
        return False
    
    # Create conda environment
    if not run_command("conda env create -f environment.yaml", "Creating conda environment"):
        print("Failed to create conda environment. Trying alternative approach...")
        
        # Try creating environment manually
        if not run_command("conda create -n rl-algos python=3.9 -y", "Creating base environment"):
            return False
        
        # Install dependencies
        if not run_command("conda activate rl-algos && pip install -r requirements.txt", "Installing dependencies"):
            return False
    
    print("\n✓ Environment setup completed!")
    print("\nTo activate the environment, run:")
    print("  conda activate rl-algos")
    
    return True

def test_installation():
    """Test the installation."""
    print("\nTesting installation...")
    print("-" * 30)
    
    # Test imports
    test_imports = [
        "import numpy as np",
        "import torch",
        "import gymnasium as gym",
        "import mujoco",
        "import matplotlib.pyplot as plt"
    ]
    
    for import_stmt in test_imports:
        try:
            exec(import_stmt)
            print(f"✓ {import_stmt}")
        except ImportError as e:
            print(f"✗ {import_stmt} - {e}")
            return False
    
    print("\n✓ All imports successful!")
    return True

def main():
    """Main setup function."""
    print("RL Algorithms for MuJoCo Hopper - Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("environment.yaml"):
        print("✗ Please run this script from the project root directory")
        return False
    
    # Setup environment
    if not setup_environment():
        print("\n✗ Setup failed!")
        return False
    
    # Test installation
    if not test_installation():
        print("\n✗ Installation test failed!")
        return False
    
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Activate the environment: conda activate rl-algos")
    print("2. Run the demo: python main.py --demo all")
    print("3. Train a model: python training/train_ppo.py --config configs/ppo_config.yaml")
    print("4. Evaluate a model: python evaluation/evaluate.py --model_path models/ppo_hopper_final.pkl --algorithm ppo")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
