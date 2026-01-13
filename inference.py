#!/usr/bin/env python3
"""
Inference script for fine-tuned ChatGLM3 model.
Use this to test and interact with your fine-tuned model.
"""

import argparse
import torch
import logging
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str, base_model_name: str = "THUDM/chatglm3-6b", cache_dir: str = "./cache"):
    """
    Load fine-tuned model with LoRA adapters.
    
    Args:
        model_path: Path to fine-tuned model directory
        base_model_name: Name of base model
        cache_dir: Cache directory for models
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModel.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        cache_dir=cache_dir
    )
    
    logger.info(f"Loading LoRA adapters from: {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded and ready on {device}")
    return model, tokenizer, device


def generate_text(model, tokenizer, device, prompt: str, max_length: int = 100, temperature: float = 0.7):
    """
    Generate text from a prompt.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        device: Device to run on
        prompt: Input prompt
        max_length: Maximum generation length
        temperature: Sampling temperature
        
    Returns:
        Generated text
    """
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from response
    response = response.replace(prompt, '').strip()
    return response


def interactive_mode(model, tokenizer, device):
    """Interactive chat mode."""
    print("\n" + "=" * 60)
    print("Interactive Chat Mode")
    print("Type 'quit' or 'exit' to end the session")
    print("=" * 60 + "\n")
    
    while True:
        try:
            prompt = input("You: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not prompt:
                continue
            
            print("Model: ", end="", flush=True)
            response = generate_text(model, tokenizer, device, prompt)
            print(response + "\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run inference with fine-tuned ChatGLM3 model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='./outputs/chatglm3_finetuned',
        help='Path to fine-tuned model directory (default: ./outputs/chatglm3_finetuned)'
    )
    
    parser.add_argument(
        '--base-model',
        type=str,
        default='THUDM/chatglm3-6b',
        help='Base model name (default: THUDM/chatglm3-6b)'
    )
    
    parser.add_argument(
        '--prompt',
        type=str,
        help='Single prompt to generate from (if not provided, enters interactive mode)'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=100,
        help='Maximum generation length (default: 100)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature (default: 0.7)'
    )
    
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='./cache',
        help='Cache directory for models (default: ./cache)'
    )
    
    args = parser.parse_args()
    
    # Load model
    try:
        model, tokenizer, device = load_model(
            args.model_path,
            args.base_model,
            args.cache_dir
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Run inference
    if args.prompt:
        # Single prompt mode
        response = generate_text(
            model, tokenizer, device,
            args.prompt,
            args.max_length,
            args.temperature
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"Response: {response}\n")
    else:
        # Interactive mode
        interactive_mode(model, tokenizer, device)


if __name__ == "__main__":
    main()
