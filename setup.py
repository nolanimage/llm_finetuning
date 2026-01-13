#!/usr/bin/env python3
"""
Setup script for ChatGLM3 Fine-tuning Framework
"""

import os
import sys
import subprocess
from pathlib import Path

def create_directories():
    """Create necessary directories."""
    directories = [
        'data',
        'outputs',
        'logs',
        'cache',
        'configs',
        'docs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")

def check_python_version():
    """Check Python version."""
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        sys.exit(1)
    print(f"✓ Python version: {sys.version.split()[0]}")

def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        sys.exit(1)

def verify_installation():
    """Verify installation."""
    print("\n🔍 Verifying installation...")
    try:
        import torch
        import transformers
        import peft
        import datasets
        
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ Transformers: {transformers.__version__}")
        print(f"✓ PEFT: {peft.__version__}")
        print(f"✓ Datasets: {datasets.__version__}")
        
        # Check for MPS/CUDA
        if torch.backends.mps.is_available():
            print("✓ MPS (Apple Silicon) available")
        elif torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CPU only (no GPU acceleration)")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main setup function."""
    print("=" * 60)
    print("ChatGLM3 Fine-tuning Framework - Setup")
    print("=" * 60)
    
    # Check Python version
    check_python_version()
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Install dependencies
    install_dependencies()
    
    # Verify installation
    if verify_installation():
        print("\n" + "=" * 60)
        print("✅ Setup completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Place your training data in data/ directory")
        print("2. Configure training in configs/training_config.yaml")
        print("3. Run: python train.py")
        print("=" * 60)
    else:
        print("\n❌ Setup completed with errors. Please check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
