# Quick Start Guide

Get up and running with ChatGLM3 fine-tuning in 5 minutes!

## Step 1: Setup (2 minutes)

```bash
# Install dependencies
pip install -r requirements.txt

# Or use the setup script
python setup.py
```

## Step 2: Prepare Data (1 minute)

Place your Chinese text data in JSONL format:

```bash
# Create data file
cat > data/my_data.jsonl << EOF
{"text": "人工智能是计算机科学的一个分支..."}
{"text": "机器学习是人工智能的一个子集..."}
EOF
```

Or use the existing data:
```bash
# Check if data exists
ls data/*.jsonl
```

## Step 3: Configure (Optional, 1 minute)

Edit `configs/training_config.yaml` if needed, or use defaults.

## Step 4: Train (Variable time)

```bash
# Start training
python train.py
```

For quick test:
```bash
python train.py --epochs 1 --batch-size 4
```

## Step 5: Test Your Model (1 minute)

```bash
# Interactive mode
python inference.py

# Or single prompt
python inference.py --prompt "中国的首都是"
```

## Docker Quick Start

```bash
# Build and run
docker-compose up --build
```

## Common Commands

```bash
# Training with custom config
python train.py --config configs/training_config.yaml

# Training with overrides
python train.py --epochs 2 --learning-rate 1e-3

# Inference with custom model
python inference.py --model-path ./outputs/chatglm3_finetuned

# View logs
tail -f training.log
```

## Troubleshooting

**Out of memory?**
- Reduce batch size: `--batch-size 1`
- Reduce sequence length in config: `max_length: 64`

**Training too slow?**
- Use smaller dataset for testing
- Reduce epochs: `--epochs 1`

**Model not loading?**
- Check cache directory exists
- Ensure internet connection for model download

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Check [docs/USAGE.md](docs/USAGE.md) for advanced usage
- Explore configuration options in `configs/training_config.yaml`

Happy fine-tuning! 🚀
