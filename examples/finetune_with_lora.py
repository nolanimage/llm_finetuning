#!/usr/bin/env python3
"""
Example script for fine-tuning with LoRA (Low-Rank Adaptation).
This demonstrates efficient fine-tuning with reduced memory usage.
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

from src.utils.device_utils import get_device, get_device_info, get_memory_info
from src.data.dataset_utils import prepare_text_classification_data
from src.training.trainer import FineTuneTrainer, create_training_args

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function for fine-tuning with LoRA."""
    
    # Device setup
    device = get_device()
    device_info = get_device_info()
    memory_info = get_memory_info(device)
    
    logger.info(f"Device info: {device_info}")
    logger.info(f"Memory info: {memory_info}")
    
    # Model and tokenizer setup
    model_name = "microsoft/DialoGPT-medium"  # Smaller model for demonstration
    
    logger.info(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create sample conversation data
    logger.info("Creating sample conversation dataset")
    conversations = [
        "Hello, how are you today? I'm doing great, thank you for asking!",
        "What's the weather like? It's sunny and warm outside.",
        "Can you help me with a problem? Of course, I'd be happy to help!",
        "What time is it? It's currently 3:30 PM.",
        "How was your day? It was quite productive, thank you for asking.",
        "Do you like movies? Yes, I enjoy watching movies very much.",
        "What's your favorite color? I like blue, it's very calming.",
        "Can you cook? I can help with recipes and cooking tips!",
        "What's the capital of France? The capital of France is Paris.",
        "How do you learn? I learn through data and training processes."
    ]
    
    # Repeat conversations to create a larger dataset
    conversations = conversations * 100  # Create 1000 samples
    
    logger.info(f"Dataset size: {len(conversations)}")
    
    # Prepare training and validation data
    train_dataset, val_dataset, train_loader, val_loader = prepare_text_generation_data(
        texts=conversations,
        tokenizer=tokenizer,
        test_size=0.2,
        max_length=128
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
    
    # Setup LoRA for efficient fine-tuning
    logger.info("Setting up LoRA configuration...")
    trainer.setup_lora(
        r=8,  # Rank (lower = less memory, less parameters)
        lora_alpha=16,  # Scaling factor
        lora_dropout=0.1  # Dropout probability
    )
    
    # Check memory usage after LoRA setup
    memory_info_after = get_memory_info(device)
    logger.info(f"Memory info after LoRA setup: {memory_info_after}")
    
    # Training configuration
    output_dir = "./outputs/lora_finetune"
    
    # Start training with LoRA
    logger.info("Starting LoRA fine-tuning...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        learning_rate=1e-4,  # Higher learning rate for LoRA
        weight_decay=0.01,
        warmup_steps=50,
        save_steps=100,
        eval_steps=100,
        output_dir=output_dir
    )
    
    # Test the LoRA fine-tuned model
    logger.info("Testing the LoRA fine-tuned model...")
    test_prompts = [
        "Hello, how are you?",
        "What's your name?",
        "Can you help me?",
        "Tell me a joke",
        "What do you think about AI?"
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
            
            # Generate response
            outputs = model.generate(
                **inputs,
                max_length=150,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Prompt: {prompt}")
            print(f"Response: {generated_text}")
            print("-" * 80)
    
    # Save LoRA adapter
    logger.info("Saving LoRA adapter...")
    lora_output_dir = os.path.join(output_dir, "lora_adapter")
    os.makedirs(lora_output_dir, exist_ok=True)
    
    # Save only the LoRA adapter weights
    model.save_pretrained(lora_output_dir)
    tokenizer.save_pretrained(lora_output_dir)
    
    logger.info("LoRA fine-tuning completed successfully!")
    logger.info(f"Full model saved to: {output_dir}")
    logger.info(f"LoRA adapter saved to: {lora_output_dir}")
    
    # Demonstrate loading LoRA adapter
    logger.info("Demonstrating LoRA adapter loading...")
    loaded_model = AutoModelForCausalLM.from_pretrained(lora_output_dir)
    loaded_model.to(device)
    
    # Test loaded model
    test_input = tokenizer("Hello!", return_tensors="pt").to(device)
    with torch.no_grad():
        output = loaded_model.generate(
            **test_input,
            max_length=50,
            temperature=0.7,
            do_sample=True
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Loaded model response: {response}")

if __name__ == "__main__":
    main() 