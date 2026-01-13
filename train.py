#!/usr/bin/env python3
"""
Main training script for ChatGLM3 fine-tuning.
This is the unified entry point for all training operations.
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.trainer import ChatGLMTrainer
from src.utils.config_loader import load_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(
        description='Fine-tune ChatGLM3 model on Chinese Wikipedia data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default configuration
  python train.py

  # Use custom configuration file
  python train.py --config configs/custom_config.yaml

  # Override specific parameters
  python train.py --epochs 2 --learning-rate 1e-3
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training_config.yaml',
        help='Path to configuration YAML file (default: configs/training_config.yaml)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Override number of training epochs'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        help='Override learning rate'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Override batch size'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Override config with command line arguments
    if args.epochs:
        config['training']['num_train_epochs'] = args.epochs
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.batch_size:
        config['training']['per_device_train_batch_size'] = args.batch_size
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    
    # Create output directory
    output_dir = config['training']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize and run trainer
    try:
        trainer = ChatGLMTrainer(config)
        train_result = trainer.train()
        
        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info(f"Final training loss: {train_result.training_loss:.4f}")
        logger.info(f"Model saved to: {output_dir}")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
