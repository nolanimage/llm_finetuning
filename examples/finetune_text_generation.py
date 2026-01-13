#!/usr/bin/env python3
"""
Example script for fine-tuning a text generation model.
This example fine-tunes GPT-2 for creative writing using a custom dataset.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, 
    TrainingArguments, Trainer
)
from datasets import load_dataset
import torch
import numpy as np

from src.utils.device_utils import get_device, get_device_info
from src.data.dataset_utils import prepare_text_generation_data
from src.training.trainer import FineTuneTrainer, create_training_args

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_texts():
    """Create sample texts for text generation fine-tuning."""
    return [
        "Once upon a time, there was a magical forest where every tree whispered ancient secrets.",
        "The spaceship hummed softly as it approached the distant planet, its crew ready for the unknown.",
        "In the bustling city, neon lights flickered against the rain-soaked streets, creating a symphony of colors.",
        "Deep in the ocean, mysterious creatures swam through coral reefs that glowed with bioluminescent beauty.",
        "The old library was filled with dusty books that contained forgotten knowledge from centuries past.",
        "A young artist painted with passion, each brushstroke bringing her imagination to life on the canvas.",
        "The mountain peaks reached toward the sky, their snow-capped summits glistening in the morning sun.",
        "Time seemed to stand still in the ancient temple, where echoes of prayers lingered in the air.",
        "The robot's eyes glowed with artificial intelligence as it processed the world around it.",
        "A gentle breeze carried the scent of blooming flowers through the peaceful garden sanctuary."
    ]

def main():
    """Main function for fine-tuning a text generation model."""
    
    # Device setup
    device = get_device()
    device_info = get_device_info()
    logger.info(f"Device info: {device_info}")
    
    # Model and tokenizer setup
    model_name = "gpt2"  # You can change this to other models like "microsoft/DialoGPT-medium"
    
    logger.info(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare sample data (in practice, you'd load your own dataset)
    logger.info("Creating sample dataset for text generation")
    texts = create_sample_texts()
    
    # Repeat texts to create a larger dataset
    texts = texts * 50  # Create 500 samples
    
    logger.info(f"Dataset size: {len(texts)}")
    
    # Prepare training and validation data
    train_dataset, val_dataset, train_loader, val_loader = prepare_text_generation_data(
        texts=texts,
        tokenizer=tokenizer,
        test_size=0.2,
        max_length=128  # Shorter sequences for faster training
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Setup trainer
    trainer = FineTuneTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        use_wandb=False
    )
    
    # Optional: Setup LoRA for efficient fine-tuning
    # trainer.setup_lora(r=16, lora_alpha=32, lora_dropout=0.1)
    
    # Training configuration
    output_dir = "./outputs/text_generation_finetune"
    
    # Start training
    logger.info("Starting fine-tuning...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
        learning_rate=5e-5,  # Slightly higher for generation tasks
        weight_decay=0.01,
        warmup_steps=50,
        save_steps=100,
        eval_steps=100,
        output_dir=output_dir
    )
    
    # Test the fine-tuned model
    logger.info("Testing the fine-tuned model...")
    test_prompts = [
        "Once upon a time",
        "The future holds",
        "In a world where",
        "The magic began when",
        "She discovered that"
    ]
    
    model.eval()
    with torch.no_grad():
        for prompt in test_prompts:
            inputs = tokenizer(
                prompt,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            ).to(device)
            
            # Generate text
            outputs = model.generate(
                **inputs,
                max_length=200,
                num_return_sequences=1,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated_text}")
            print("-" * 80)
    
    logger.info("Fine-tuning completed successfully!")
    logger.info(f"Model saved to: {output_dir}")

if __name__ == "__main__":
    main() 