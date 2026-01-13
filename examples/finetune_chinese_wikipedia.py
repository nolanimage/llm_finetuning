#!/usr/bin/env python3
"""
Example script for fine-tuning Chinese language models using Wikipedia data.
This script downloads and processes Chinese Wikipedia dumps for fine-tuning.
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
import torch
import numpy as np

from src.utils.device_utils import get_device, get_device_info
from src.data.dataset_utils import prepare_text_generation_data
from src.data.wikipedia_processor import ChineseWikipediaProcessor
from src.training.trainer import FineTuneTrainer, create_training_args

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function for fine-tuning with Chinese Wikipedia data."""
    
    # Device setup
    device = get_device()
    device_info = get_device_info()
    logger.info(f"Device info: {device_info}")
    
    # Initialize Wikipedia processor
    wiki_processor = ChineseWikipediaProcessor(data_dir="./data")
    
    # Check if we already have processed data
    processed_file = "./data/chinese_wiki_articles.jsonl"
    
    if not os.path.exists(processed_file):
        logger.info("Processing Chinese Wikipedia data...")
        
        # Download Wikipedia dump (this will take a while - ~3GB)
        logger.info("Downloading Chinese Wikipedia dump...")
        dump_file = wiki_processor.download_wikipedia_dump(
            date="20250320",
            file_type="pages-articles"
        )
        
        # Process the dump and create training dataset
        logger.info("Creating training dataset from Wikipedia dump...")
        processed_file = wiki_processor.create_training_dataset(
            dump_file=dump_file,
            output_file="chinese_wiki_articles.jsonl",
            min_length=200,  # Minimum 200 characters
            max_length=1500,  # Maximum 1500 characters
            max_articles=50000  # Limit to 50k articles for faster training
        )
        
        # Get dataset statistics
        stats = wiki_processor.get_dataset_stats(processed_file)
        logger.info(f"Dataset statistics: {stats}")
    else:
        logger.info(f"Using existing processed data: {processed_file}")
        stats = wiki_processor.get_dataset_stats(processed_file)
        logger.info(f"Dataset statistics: {stats}")
    
    # Load articles
    articles = wiki_processor.load_articles_from_jsonl(processed_file)
    texts = [article['text'] for article in articles]
    
    logger.info(f"Loaded {len(texts)} articles for training")
    
    # Model and tokenizer setup for Chinese
    model_name = "THUDM/chatglm2-6b"  # Good Chinese model, or try "fnlp/moss-moon-003-base"
    
    logger.info(f"Loading model and tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float16,  # Use half precision to save memory
            device_map="auto"  # Automatically handle device placement
        )
    except Exception as e:
        logger.warning(f"Failed to load {model_name}, trying alternative...")
        # Fallback to a smaller Chinese model
        model_name = "fnlp/moss-moon-003-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare training and validation data
    logger.info("Preparing data for training...")
    train_dataset, val_dataset, train_loader, val_loader = prepare_text_generation_data(
        texts=texts,
        tokenizer=tokenizer,
        test_size=0.1,  # 10% for validation
        max_length=512  # Shorter sequences for Chinese text
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
    
    # Setup LoRA for efficient fine-tuning (recommended for large models)
    logger.info("Setting up LoRA for efficient fine-tuning...")
    trainer.setup_lora(
        r=8,  # Lower rank for memory efficiency
        lora_alpha=16,
        lora_dropout=0.1
    )
    
    # Training configuration
    output_dir = "./outputs/chinese_wiki_finetune"
    
    # Start training
    logger.info("Starting fine-tuning with Chinese Wikipedia data...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,  # Start with fewer epochs
        learning_rate=1e-4,  # Lower learning rate for Chinese
        weight_decay=0.01,
        warmup_steps=100,
        save_steps=200,
        eval_steps=200,
        output_dir=output_dir
    )
    
    # Test the fine-tuned model
    logger.info("Testing the fine-tuned Chinese model...")
    test_prompts = [
        "人工智能是",
        "中国的历史",
        "机器学习技术",
        "自然语言处理",
        "深度学习在"
    ]
    
    model.eval()
    with torch.no_grad():
        for prompt in test_prompts:
            inputs = tokenizer(
                prompt,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)
            
            # Generate text
            outputs = model.generate(
                **inputs,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated_text}")
            print("-" * 80)
    
    logger.info("Chinese Wikipedia fine-tuning completed successfully!")
    logger.info(f"Model saved to: {output_dir}")

if __name__ == "__main__":
    main() 