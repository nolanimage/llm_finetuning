#!/usr/bin/env python3
"""
Quick test script to verify the setup is working correctly.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers: {e}")
        return False
    
    try:
        import peft
        print(f"✓ PEFT {peft.__version__}")
    except ImportError as e:
        print(f"✗ PEFT: {e}")
        return False
    
    try:
        import datasets
        print(f"✓ Datasets {datasets.__version__}")
    except ImportError as e:
        print(f"✗ Datasets: {e}")
        return False
    
    try:
        import yaml
        print(f"✓ PyYAML")
    except ImportError as e:
        print(f"✗ PyYAML: {e}")
        return False
    
    return True

def test_project_structure():
    """Test that project structure is correct."""
    print("\nTesting project structure...")
    
    required_dirs = ['src', 'configs', 'data', 'outputs', 'logs', 'cache']
    missing = []
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✓ {dir_name}/")
        else:
            print(f"✗ {dir_name}/ (missing)")
            missing.append(dir_name)
    
    if missing:
        print(f"\n⚠️  Missing directories: {', '.join(missing)}")
        print("   Run: python setup.py to create them")
        return False
    
    return True

def test_source_modules():
    """Test that source modules can be imported."""
    print("\nTesting source modules...")
    
    try:
        from src.utils.config_loader import load_config
        print("✓ config_loader")
    except Exception as e:
        print(f"✗ config_loader: {e}")
        return False
    
    try:
        from src.utils.device_utils import get_device
        print("✓ device_utils")
    except Exception as e:
        print(f"✗ device_utils: {e}")
        return False
    
    try:
        from src.data.dataset_loader import load_chinese_wiki_dataset
        print("✓ dataset_loader")
    except Exception as e:
        print(f"✗ dataset_loader: {e}")
        return False
    
    try:
        from src.training.trainer import ChatGLMTrainer
        print("✓ trainer")
    except Exception as e:
        print(f"✗ trainer: {e}")
        return False
    
    return True

def test_config_file():
    """Test that config file exists and is valid."""
    print("\nTesting configuration...")
    
    config_path = Path('configs/training_config.yaml')
    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        return False
    
    try:
        from src.utils.config_loader import load_config
        config = load_config(str(config_path))
        print(f"✓ Config file loaded successfully")
        print(f"  Model: {config['model']['name']}")
        print(f"  LoRA enabled: {config['lora']['enabled']}")
        print(f"  Epochs: {config['training']['num_train_epochs']}")
        return True
    except Exception as e:
        print(f"✗ Config file error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("ChatGLM3 Fine-tuning Framework - Setup Test")
    print("=" * 60)
    print()
    
    tests = [
        ("Imports", test_imports),
        ("Project Structure", test_project_structure),
        ("Source Modules", test_source_modules),
        ("Configuration", test_config_file),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed! Setup is correct.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
