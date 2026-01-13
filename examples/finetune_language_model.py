#!/usr/bin/env python3
"""
Example script for fine-tuning a language model for text classification.
This example fine-tunes BERT for sentiment analysis using the IMDB dataset.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, 
    TrainingArguments, Trainer
)
from datasets import load_dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.utils.device_utils import get_device, get_device_info
from src.data.dataset_utils import prepare_text_classification_data
from src.training.trainer import FineTuneTrainer, create_training_args

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function for fine-tuning a language model."""
    
    # Device setup
    device = get_device()
    device_info = get_device_info()
    logger.info(f"Device info: {device_info}")
    
    # Model and tokenizer setup
    model_name = "bert-base-uncased"  # You can change this to other models
    num_labels = 2  # Binary classification for sentiment analysis
    
    logger.info(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    logger.info("Loading IMDB dataset for sentiment analysis")
    dataset = load_dataset("imdb", split="train[:1000]")  # Using subset for faster training
    
    # Prepare data
    texts = dataset["text"]
    labels = dataset["label"]
    
    logger.info(f"Dataset size: {len(texts)}")
    logger.info(f"Label distribution: {np.bincount(labels)}")
    
    # Prepare training and validation data
    train_dataset, val_dataset, train_loader, val_loader = prepare_text_classification_data(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        test_size=0.2,
        max_length=256  # Reduced for faster training
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Setup trainer
    trainer = FineTuneTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        use_wandb=False  # Set to True if you want to use Weights & Biases
    )
    
    # Optional: Setup LoRA for efficient fine-tuning
    # trainer.setup_lora(r=16, lora_alpha=32, lora_dropout=0.1)
    
    # Training configuration
    output_dir = "./outputs/language_model_finetune"
    
    # Start training
    logger.info("Starting fine-tuning...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,  # Reduced for faster training
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=50,
        save_steps=100,
        eval_steps=100,
        output_dir=output_dir
    )
    
    # Test the fine-tuned model
    logger.info("Testing the fine-tuned model...")
    test_examples = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible film, waste of time. Don't watch it.",
        "It was okay, nothing special but not bad either.",
        "Amazing performance by all actors, highly recommended!"
    ]
    
    model.eval()
    with torch.no_grad():
        for text in test_examples:
            inputs = tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt"
            ).to(device)
            
            outputs = model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=-1)
            predicted_label = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_label].item()
            
            sentiment = "Positive" if predicted_label == 1 else "Negative"
            print(f"Text: {text}")
            print(f"Sentiment: {sentiment} (confidence: {confidence:.3f})")
            print("-" * 50)
    
    logger.info("Fine-tuning completed successfully!")
    logger.info(f"Model saved to: {output_dir}")

if __name__ == "__main__":
    main() 