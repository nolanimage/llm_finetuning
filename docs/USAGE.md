# Usage Guide

This guide provides detailed instructions for using the ChatGLM3 fine-tuning framework.

## Table of Contents

1. [Data Preparation](#data-preparation)
2. [Training](#training)
3. [Inference](#inference)
4. [Configuration](#configuration)
5. [Advanced Usage](#advanced-usage)

## Data Preparation

### Data Format

Your training data should be in JSONL (JSON Lines) format, where each line is a JSON object with a `text` field:

```json
{"text": "人工智能是计算机科学的一个分支..."}
{"text": "机器学习是人工智能的一个子集..."}
{"text": "深度学习使用神经网络来模拟人脑..."}
```

### Data Location

Place your data file in the `data/` directory:

```bash
data/
├── chinese_wiki_5000.jsonl
├── your_custom_data.jsonl
└── ...
```

### Data Requirements

- **Encoding**: UTF-8
- **Format**: JSONL (one JSON object per line)
- **Field**: Must contain `text` field
- **Size**: Recommended 1000+ samples for good results

### Example: Creating Your Own Dataset

```python
import json

# Your text data
texts = [
    "这是第一段中文文本...",
    "这是第二段中文文本...",
    # ... more texts
]

# Save as JSONL
with open('data/my_data.jsonl', 'w', encoding='utf-8') as f:
    for text in texts:
        json.dump({'text': text}, f, ensure_ascii=False)
        f.write('\n')
```

## Training

### Basic Training

The simplest way to train:

```bash
python train.py
```

This uses the default configuration from `configs/training_config.yaml`.

### Custom Configuration File

Create your own config file:

```bash
cp configs/training_config.yaml configs/my_config.yaml
# Edit my_config.yaml
python train.py --config configs/my_config.yaml
```

### Command-Line Overrides

Override specific parameters without editing config:

```bash
python train.py --epochs 2 --learning-rate 1e-3 --batch-size 4
```

### Training Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config` | Path to config file | `configs/training_config.yaml` |
| `--epochs` | Number of training epochs | From config |
| `--learning-rate` | Learning rate | From config |
| `--batch-size` | Batch size per device | From config |
| `--output-dir` | Output directory | From config |

### Monitoring Training

#### Console Output

Training progress is shown in real-time:
```
2024-01-01 10:00:00 - INFO - Starting training...
2024-01-01 10:00:05 - INFO - Epoch 1/4: Loss 5.1234
2024-01-01 10:00:10 - INFO - Step 50: Loss 4.5678
...
```

#### Log Files

Check `training.log` for detailed logs:
```bash
tail -f training.log
```

#### TensorBoard

If TensorBoard is enabled:
```bash
tensorboard --logdir logs
# Open http://localhost:6006
```

### Training Outputs

After training, you'll find:

```
outputs/chatglm3_finetuned/
├── adapter_config.json      # LoRA configuration
├── adapter_model.bin        # LoRA weights
├── training_args.bin        # Training arguments
└── checkpoint-*/            # Checkpoints (if saved)
```

## Inference

### Interactive Mode

Start an interactive chat session:

```bash
python inference.py
```

Example session:
```
You: 中国的首都是
Model: 北京。北京是中华人民共和国的首都，也是中国的政治、文化中心...

You: 人工智能是什么
Model: 人工智能（AI）是指由机器展现出的智能...
```

### Single Prompt

Generate text from a single prompt:

```bash
python inference.py --prompt "人工智能是指"
```

### Custom Model

Use a different trained model:

```bash
python inference.py \
    --model-path ./outputs/my_custom_model \
    --prompt "你的问题"
```

### Inference Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model-path` | Path to trained model | `./outputs/chatglm3_finetuned` |
| `--base-model` | Base model name | `THUDM/chatglm3-6b` |
| `--prompt` | Input prompt | Interactive mode if not provided |
| `--max-length` | Max generation length | 100 |
| `--temperature` | Sampling temperature | 0.7 |
| `--cache-dir` | Model cache directory | `./cache` |

### Temperature Guide

- **0.1-0.3**: Very focused, deterministic
- **0.7-0.9**: Balanced creativity and coherence (recommended)
- **1.0-1.5**: More creative, less focused

## Configuration

### Configuration File Structure

```yaml
# Model settings
model:
  name: "THUDM/chatglm3-6b"
  cache_dir: "./cache"
  torch_dtype: "float16"

# LoRA settings
lora:
  enabled: true
  r: 4
  lora_alpha: 8
  # ...

# Training settings
training:
  num_train_epochs: 4
  learning_rate: 2e-3
  # ...

# Data settings
data:
  train_file: "data/chinese_wiki_5000.jsonl"
  max_length: 96
  # ...
```

### Recommended Configurations

#### Fast Training (Quick Tests)
```yaml
lora:
  r: 4
data:
  max_length: 64
training:
  num_train_epochs: 1
  per_device_train_batch_size: 4
```

#### Balanced (Recommended)
```yaml
lora:
  r: 4
data:
  max_length: 96
training:
  num_train_epochs: 4
  per_device_train_batch_size: 2
```

#### High Quality (More Memory)
```yaml
lora:
  r: 8
data:
  max_length: 128
training:
  num_train_epochs: 4
  per_device_train_batch_size: 2
```

## Advanced Usage

### Programmatic Usage

Use the framework in your own Python code:

```python
from src.training.trainer import ChatGLMTrainer
from src.utils.config_loader import load_config

# Load config
config = load_config('configs/training_config.yaml')

# Customize config
config['training']['num_train_epochs'] = 2
config['lora']['r'] = 8

# Create and run trainer
trainer = ChatGLMTrainer(config)
trainer.train()
```

### Custom Data Processing

```python
from src.data.dataset_loader import prepare_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm3-6b')
data = [{'text': '你的文本...'}]

train_dataset, eval_dataset, _ = prepare_dataset(
    data=data,
    tokenizer=tokenizer,
    max_length=128,
    test_size=0.2
)
```

### Resume Training

To resume from a checkpoint:

```python
from src.training.trainer import ChatGLMTrainer
from src.utils.config_loader import load_config

config = load_config('configs/training_config.yaml')
trainer = ChatGLMTrainer(config)
trainer.load_model_and_tokenizer()
trainer.setup_lora()

# Load checkpoint
trainer.trainer = Trainer(...)
trainer.trainer.train(resume_from_checkpoint='./outputs/checkpoint-100')
```

### Multi-GPU Training

For multi-GPU setups, use Hugging Face Accelerate:

```bash
accelerate config
accelerate launch train.py
```

## Tips and Tricks

1. **Start Small**: Test with 100-1000 samples first
2. **Monitor Loss**: Watch for sudden drops (gradient explosion)
3. **Save Often**: Checkpoints are saved every 50 steps
4. **Validate**: Use validation split to detect overfitting
5. **Experiment**: Try different LoRA ranks and learning rates
6. **Backup**: Keep copies of successful configurations

## Next Steps

- Read the [API Documentation](API.md)
- Check [Troubleshooting](../README.md#troubleshooting)
- Explore [Examples](../examples/)
