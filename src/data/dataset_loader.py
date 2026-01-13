"""
Dataset loading and preprocessing utilities for Chinese Wikipedia data.
"""

import json
import os
from typing import List, Dict, Optional
from datasets import Dataset
from transformers import PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)


def load_chinese_wiki_dataset(file_path: str) -> List[Dict[str, str]]:
    """
    Load Chinese Wikipedia data from JSONL file.
    
    Args:
        file_path: Path to JSONL file containing Chinese Wikipedia articles
        
    Returns:
        List of dictionaries with 'text' field
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is empty or invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    logger.info(f"Loading dataset from {file_path}")
    data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    if 'text' not in item:
                        logger.warning(f"Line {line_num}: Missing 'text' field, skipping")
                        continue
                    if not item['text'].strip():
                        logger.warning(f"Line {line_num}: Empty text, skipping")
                        continue
                    data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: Invalid JSON, skipping: {e}")
                    continue
        
        if not data:
            raise ValueError(f"No valid data found in {file_path}")
        
        logger.info(f"Successfully loaded {len(data)} articles")
        return data
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def prepare_dataset(
    data: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 96,
    test_size: float = 0.1
) -> tuple:
    """
    Prepare dataset for training by tokenizing and splitting.
    
    Args:
        data: List of data dictionaries with 'text' field
        tokenizer: Tokenizer to use for encoding
        max_length: Maximum sequence length
        test_size: Fraction of data to use for validation
        
    Returns:
        Tuple of (train_dataset, eval_dataset, tokenized_dataset)
    """
    logger.info(f"Preparing dataset with max_length={max_length}")
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(data)
    
    # Tokenization function
    def tokenize_function(examples):
        texts = examples['text']
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors=None  # Return lists, not tensors
        )
        # For causal LM, labels are the same as input_ids
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    # Split into train and eval
    if test_size > 0:
        split_dataset = tokenized_dataset.train_test_split(test_size=test_size, seed=42)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
        logger.info(f"Split dataset: {len(train_dataset)} train, {len(eval_dataset)} eval")
    else:
        train_dataset = tokenized_dataset
        eval_dataset = None
        logger.info(f"Using full dataset for training: {len(train_dataset)} samples")
    
    return train_dataset, eval_dataset, tokenized_dataset
