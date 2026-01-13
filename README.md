# ChatGLM3 Fine-tuning Framework

A professional, production-ready framework for fine-tuning ChatGLM3-6B models on Chinese text data using LoRA (Low-Rank Adaptation). This project provides a complete end-to-end solution: from downloading and processing Chinese Wikipedia data to training and deploying fine-tuned models.

## 🎯 Project Purpose

This project enables you to:

1. **Download Chinese Wikipedia Data** - Automatically fetch and process Chinese Wikipedia dumps
2. **Fine-tune ChatGLM3-6B** - Efficiently train the model using LoRA (only ~7MB of parameters instead of 12GB)
3. **Generate Chinese Text** - Use the fine-tuned model for Chinese text generation tasks

### Why This Project?

- **Efficient Training**: LoRA reduces memory usage by 99%+ (7MB vs 12GB)
- **Complete Workflow**: From raw Wikipedia dumps to trained models
- **Production Ready**: Clean code, Docker support, comprehensive documentation
- **Easy to Use**: Simple commands, YAML configuration, clear documentation
- **Optimized for Mac**: Works great on Apple Silicon (M1/M2/M3) with MPS acceleration

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [What This Project Does](#-what-this-project-does)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
- [Configuration](#-configuration)
- [Docker Usage](#-docker-usage)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Prepare Training Data

**Option A: Use Wikipedia Data (Recommended)**

```bash
# Download and process Chinese Wikipedia (5000 articles)
python prepare_data.py --download --max-articles 5000

# This will:
# 1. Download Wikipedia dump (~3GB, one-time)
# 2. Extract and clean articles
# 3. Create training dataset: data/chinese_wiki_articles.jsonl
```

**Option B: Use Existing Data**

If you already have `data/chinese_wiki_5000.jsonl`, you can skip this step!

**Option C: Use Your Own Data**

Place your JSONL file in `data/` directory with format:
```json
{"text": "你的中文文本内容..."}
```

### Step 3: Configure Training (Optional)

Edit `configs/training_config.yaml` to customize:
- Number of epochs
- Learning rate
- Batch size
- LoRA rank
- etc.

Or use command-line arguments (see below).

### Step 4: Start Training

```bash
# Train with default configuration
python train.py

# Or customize via command-line
python train.py --epochs 2 --learning-rate 1e-3 --batch-size 4
```

### Step 5: Test Your Model

```bash
# Interactive mode
python inference.py

# Or single prompt
python inference.py --prompt "中国的首都是"
```

**That's it!** Your fine-tuned model is ready to use.

## 📖 What This Project Does

### The Complete Workflow

```
1. Download Wikipedia Dump
   ↓
2. Extract & Clean Articles
   ↓
3. Create Training Dataset (JSONL)
   ↓
4. Fine-tune ChatGLM3 with LoRA
   ↓
5. Generate Chinese Text
```

### Detailed Process

#### 1. Data Preparation (`prepare_data.py`)

- **Downloads** Chinese Wikipedia dump from Wikimedia (https://dumps.wikimedia.org/zhwiki/)
- **Extracts** articles from XML dump file
- **Cleans** text by removing:
  - Wikipedia markup (`[[links]]`, `{{templates}}`)
  - References and categories
  - HTML tags and formatting
- **Filters** articles by length (min/max characters)
- **Saves** to JSONL format ready for training

#### 2. Training (`train.py`)

- **Loads** ChatGLM3-6B base model (6 billion parameters)
- **Applies LoRA** (Low-Rank Adaptation) - only trains ~7MB of parameters
- **Trains** on your Chinese text data
- **Saves** checkpoints and final model to `outputs/`

#### 3. Inference (`inference.py`)

- **Loads** your fine-tuned model
- **Generates** Chinese text from prompts
- **Supports** interactive chat mode

### Key Features

- ✅ **Automatic Wikipedia Processing** - Download and process Chinese Wikipedia dumps
- ✅ **Efficient LoRA Training** - Train with minimal memory (7MB vs 12GB)
- ✅ **Production Ready** - Clean code, error handling, logging
- ✅ **Docker Support** - Containerized training environment
- ✅ **Well Documented** - Comprehensive guides and examples
- ✅ **Mac Optimized** - Works great on Apple Silicon

## 🔧 Installation

### Prerequisites

- **Python 3.10+**
- **16GB+ RAM** (32GB recommended)
- **CUDA-capable GPU** (optional, but recommended) or **Apple Silicon** (M1/M2/M3)
- **20GB+ free disk space** for model cache
- **~5GB free space** for Wikipedia dump (if downloading)

### Step-by-Step Installation

1. **Clone or navigate to the project**:
```bash
cd fine_tune
```

2. **Create virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Verify installation**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
```

## 📚 Usage Guide

### Complete Workflow Example

```bash
# 1. Prepare data from Wikipedia
python prepare_data.py --download --max-articles 5000

# 2. Train the model
python train.py

# 3. Test the model
python inference.py --prompt "人工智能是指"
```

### Data Preparation

#### Download and Process Wikipedia

```bash
# Download latest Wikipedia dump and process 5000 articles
python prepare_data.py --download --max-articles 5000

# Use existing dump file
python prepare_data.py --dump-file data/zhwiki-20250320-pages-articles.xml.bz2 --max-articles 5000

# Show statistics for existing dataset
python prepare_data.py --stats data/chinese_wiki_5000.jsonl
```

#### Use Your Own Data

Create a JSONL file in `data/` directory:

```json
{"text": "第一段中文文本..."}
{"text": "第二段中文文本..."}
{"text": "第三段中文文本..."}
```

Then update `configs/training_config.yaml`:
```yaml
data:
  train_file: "data/your_data.jsonl"
```

### Training

#### Basic Training

```bash
# Use default configuration
python train.py
```

#### Custom Configuration

```bash
# Use custom config file
python train.py --config configs/my_config.yaml

# Override specific parameters
python train.py --epochs 2 --learning-rate 1e-3 --batch-size 4
```

#### Training Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config` | Path to config file | `configs/training_config.yaml` |
| `--epochs` | Number of training epochs | From config |
| `--learning-rate` | Learning rate | From config |
| `--batch-size` | Batch size per device | From config |
| `--output-dir` | Output directory | From config |

### Inference

#### Interactive Mode

```bash
python inference.py
```

Example session:
```
You: 中国的首都是
Model: 北京。北京是中华人民共和国的首都...

You: 人工智能是什么
Model: 人工智能（AI）是指由机器展现出的智能...
```

#### Single Prompt

```bash
python inference.py --prompt "人工智能是指"
```

#### Custom Model

```bash
python inference.py \
    --model-path ./outputs/chatglm3_finetuned \
    --prompt "你的问题" \
    --max-length 200 \
    --temperature 0.8
```

## ⚙️ Configuration

The training configuration is managed through YAML files. The main configuration file is `configs/training_config.yaml`.

### Key Configuration Sections

#### Model Configuration
```yaml
model:
  name: "THUDM/chatglm3-6b"
  cache_dir: "./cache"
  torch_dtype: "float16"
```

#### LoRA Configuration
```yaml
lora:
  enabled: true
  r: 4  # Rank: 4 (stable), 8 (quality), 16 (maximum - may cause issues)
  lora_alpha: 8
  lora_dropout: 0.1
  target_modules:
    - "query_key_value"
    - "dense"
    - "dense_h_to_4h"
    - "dense_4h_to_h"
```

#### Training Configuration
```yaml
training:
  num_train_epochs: 4
  per_device_train_batch_size: 2
  learning_rate: 2e-3
  lr_scheduler_type: "constant"
  max_grad_norm: 1.0
```

#### Data Configuration
```yaml
data:
  train_file: "data/chinese_wiki_5000.jsonl"
  max_length: 96
  test_size: 0.1
```

See `configs/training_config.yaml` for all available options and detailed comments.

### Configuration Best Practices

1. **For Stability**: Use `r=4`, `max_length=96`, `learning_rate=2e-3`
2. **For Quality**: Use `r=8`, `max_length=128`, `learning_rate=2e-3` (may require more memory)
3. **For Speed**: Use `r=4`, `max_length=64`, fewer epochs

## 🐳 Docker Usage

### Build and Run

```bash
# Build the Docker image
docker-compose build

# Start training
docker-compose up

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop training
docker-compose down
```

### Custom Configuration with Docker

```bash
# Override command
docker-compose run --rm training python train.py --epochs 2
```

### GPU Support

For GPU support, uncomment the GPU section in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Then rebuild:
```bash
docker-compose build
docker-compose up
```

## 📁 Project Structure

```
fine_tune/
├── src/                    # Source code
│   ├── data/              # Data loading utilities
│   │   ├── dataset_loader.py    # Load JSONL datasets
│   │   └── wikipedia_processor.py  # Download & process Wikipedia
│   ├── training/          # Training modules
│   │   └── trainer.py
│   └── utils/             # Utility functions
│       ├── config_loader.py
│       └── device_utils.py
├── configs/              # Configuration files
│   ├── default_config.yaml
│   └── training_config.yaml
├── data/                  # Data directory
│   ├── *.jsonl           # Processed training data
│   └── zhwiki-*.xml.bz2  # Wikipedia dumps (if downloaded)
├── outputs/               # Training outputs (models, checkpoints)
├── logs/                  # Training logs
├── cache/                 # Model cache
├── docs/                  # Documentation
│   ├── USAGE.md
│   └── DATA_PREPARATION.md
├── scripts/               # Utility scripts
├── prepare_data.py       # Data preparation script (Wikipedia)
├── train.py              # Main training script
├── inference.py          # Inference script
├── setup.py              # Setup script
├── test_setup.py         # Verification script
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── Makefile              # Common tasks
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Out of Memory
**Solution**: Reduce batch size or sequence length in config:
```yaml
training:
  per_device_train_batch_size: 1
data:
  max_length: 64
```

#### 2. Model Loading Stuck
**Solution**: Ensure you're using `float16` and `low_cpu_mem_usage: true`:
```yaml
model:
  torch_dtype: "float16"
  low_cpu_mem_usage: true
```

#### 3. Gradient Explosion
**Solution**: Use lower LoRA rank and gradient clipping:
```yaml
lora:
  r: 4  # Instead of 8 or 16
training:
  max_grad_norm: 1.0
```

#### 4. Slow Training
**Solution**: 
- Use smaller sequence length (`max_length: 64`)
- Reduce number of epochs
- Use smaller LoRA rank (`r: 4`)

#### 5. MPS (Apple Silicon) Issues
**Solution**: Ensure these settings in config:
```yaml
training:
  fp16: false
  bf16: false
  dataloader_pin_memory: false
  dataloader_num_workers: 0
```

#### 6. Wikipedia Download Fails
**Solution**: 
- Check internet connection
- Try different date: `--date 20240101`
- Download manually and use `--dump-file` option

### Getting Help

1. Check the logs in `logs/` directory
2. Review configuration in `configs/training_config.yaml`
3. See `docs/DATA_PREPARATION.md` for data preparation help
4. See `docs/USAGE.md` for detailed usage instructions

## 📊 Expected Results

### Training Performance

With default configuration on 5000 Chinese Wikipedia articles:
- **Training Time**: 3-4 hours (MPS) / 1-2 hours (CUDA)
- **Final Loss**: 2.0-2.5 (from initial ~5.0)
- **Model Size**: ~7MB LoRA adapters (0.12% of base model)
- **Memory Usage**: ~12-16GB RAM

### Monitoring Training

Training progress is logged to:
- Console output (real-time)
- `training.log` file
- TensorBoard (if enabled): `tensorboard --logdir logs`

## 🎓 Best Practices

1. **Start Small**: Test with a small dataset first (100-1000 samples)
2. **Monitor Loss**: Watch for gradient explosion (loss → 0 or NaN)
3. **Save Checkpoints**: Models are saved every 50 steps by default
4. **Validate**: Use validation split to monitor overfitting
5. **Experiment**: Try different LoRA ranks and learning rates
6. **Backup**: Keep backups of successful training runs

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [ChatGLM3](https://github.com/THUDM/ChatGLM3) by THUDM
- [Hugging Face Transformers](https://huggingface.co/transformers)
- [PEFT](https://github.com/huggingface/peft) for LoRA implementation
- [Wikimedia](https://dumps.wikimedia.org/) for Wikipedia dumps

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Last Updated**: 2025  
**Version**: 1.0.0  
**Status**: ✅ Production Ready
