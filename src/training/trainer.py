"""
Training module for ChatGLM3 fine-tuning with LoRA.
"""

import torch
import logging
import os
from typing import Dict, Any, Optional
from transformers import (
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

from ..utils.device_utils import get_device, setup_device
from ..data.dataset_loader import load_chinese_wiki_dataset, prepare_dataset

logger = logging.getLogger(__name__)


class ChatGLMTrainer:
    """
    Trainer class for fine-tuning ChatGLM3 models with LoRA.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trainer with configuration.
        
        Args:
            config: Configuration dictionary from YAML file
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self.trainer = None
        
        # Setup device
        device_config = config.get('device', {})
        self.device = get_device(
            force_cpu=device_config.get('force_cpu', False),
            force_mps=device_config.get('force_mps', False)
        )
        setup_device(self.device)
    
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer."""
        model_config = self.config['model']
        logger.info(f"Loading model: {model_config['name']}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config['name'],
            trust_remote_code=model_config.get('trust_remote_code', True),
            cache_dir=model_config.get('cache_dir', './cache')
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        torch_dtype_str = model_config.get('torch_dtype', 'float16')
        if isinstance(torch_dtype_str, str):
            torch_dtype = getattr(torch, torch_dtype_str)
        else:
            torch_dtype = torch_dtype_str
        
        self.model = AutoModel.from_pretrained(
            model_config['name'],
            trust_remote_code=model_config.get('trust_remote_code', True),
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_config.get('low_cpu_mem_usage', True),
            cache_dir=model_config.get('cache_dir', './cache')
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        logger.info(f"Model loaded and moved to {self.device}")
    
    def setup_lora(self):
        """Setup LoRA configuration and apply to model."""
        if not self.config.get('lora', {}).get('enabled', True):
            logger.info("LoRA disabled, using full fine-tuning")
            return
        
        lora_config_dict = self.config['lora']
        logger.info("Setting up LoRA configuration")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=lora_config_dict.get('inference_mode', False),
            r=lora_config_dict.get('r', 4),
            lora_alpha=lora_config_dict.get('lora_alpha', 8),
            lora_dropout=lora_config_dict.get('lora_dropout', 0.1),
            target_modules=lora_config_dict.get('target_modules', [
                'query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h'
            ])
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        logger.info("LoRA configuration applied successfully")
    
    def load_and_prepare_data(self):
        """Load and prepare training data."""
        data_config = self.config['data']
        train_file = data_config['train_file']
        
        # Load dataset
        data = load_chinese_wiki_dataset(train_file)
        
        # Prepare dataset
        train_dataset, eval_dataset, _ = prepare_dataset(
            data=data,
            tokenizer=self.tokenizer,
            max_length=data_config.get('max_length', 96),
            test_size=data_config.get('test_size', 0.1)
        )
        
        return train_dataset, eval_dataset
    
    def create_training_arguments(self) -> TrainingArguments:
        """Create training arguments from configuration."""
        training_config = self.config['training']
        
        # Convert torch_dtype string to type if needed
        fp16 = training_config.get('fp16', False)
        bf16 = training_config.get('bf16', False)
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=training_config['output_dir'],
            num_train_epochs=training_config.get('num_train_epochs', 3),
            per_device_train_batch_size=training_config.get('per_device_train_batch_size', 2),
            per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 2),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 2),
            learning_rate=training_config.get('learning_rate', 2e-3),
            lr_scheduler_type=training_config.get('lr_scheduler_type', 'constant'),
            warmup_steps=training_config.get('warmup_steps', 0),
            weight_decay=training_config.get('weight_decay', 0.01),
            max_grad_norm=training_config.get('max_grad_norm', 1.0),
            logging_steps=training_config.get('logging_steps', 5),
            save_steps=training_config.get('save_steps', 50),
            save_strategy=training_config.get('save_strategy', 'steps'),
            eval_steps=training_config.get('eval_steps', 50),
            evaluation_strategy=training_config.get('evaluation_strategy', 'steps'),
            save_total_limit=training_config.get('save_total_limit', 3),
            fp16=fp16,
            bf16=bf16,
            dataloader_pin_memory=training_config.get('dataloader_pin_memory', False),
            dataloader_num_workers=training_config.get('dataloader_num_workers', 0),
            load_best_model_at_end=training_config.get('load_best_model_at_end', True),
            metric_for_best_model=training_config.get('metric_for_best_model', 'eval_loss'),
            greater_is_better=training_config.get('greater_is_better', False),
            report_to=training_config.get('report_to', []),
            logging_dir=training_config.get('logging_dir', './logs')
        )
        
        return training_args
    
    def train(self):
        """Run the training process."""
        logger.info("Starting training process")
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Setup LoRA
        self.setup_lora()
        
        # Load and prepare data
        train_dataset, eval_dataset = self.load_and_prepare_data()
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
        
        # Create training arguments
        training_args = self.create_training_arguments()
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Train
        logger.info("Starting training...")
        train_result = self.trainer.train()
        
        # Save final model
        logger.info("Saving final model...")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        logger.info(f"Training completed! Model saved to {training_args.output_dir}")
        logger.info(f"Training loss: {train_result.training_loss:.4f}")
        
        return train_result
