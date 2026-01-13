# Data Preparation Guide

This project supports two ways to get training data:

1. **Download and process Chinese Wikipedia** (automatic)
2. **Use your own data** (manual)

## Option 1: Wikipedia Data (Recommended)

### Step 1: Download and Process Wikipedia

The project can automatically download and process Chinese Wikipedia dumps:

```bash
# Download latest Wikipedia dump and process 5000 articles
python prepare_data.py --download --max-articles 5000

# Or use existing dump file
python prepare_data.py --dump-file data/zhwiki-20250320-pages-articles.xml.bz2 --max-articles 5000
```

### What It Does

1. **Downloads Wikipedia dump** (~3GB compressed XML file)
   - From: `https://dumps.wikimedia.org/zhwiki/`
   - Contains all Chinese Wikipedia articles

2. **Extracts articles** from the XML dump
   - Filters by length (min/max characters)
   - Removes Wikipedia markup, templates, links
   - Cleans text for training

3. **Creates JSONL dataset** ready for training
   - Format: One JSON object per line with `text` field
   - Saved to `data/chinese_wiki_articles.jsonl`

### Options

```bash
python prepare_data.py --help  # See all options

# Common options:
--download              # Download Wikipedia dump
--dump-file FILE        # Use existing dump file
--max-articles N        # Limit number of articles
--min-length N          # Minimum article length (default: 100)
--max-length N          # Maximum article length (default: 2000)
--output-file NAME      # Output filename (default: chinese_wiki_articles.jsonl)
```

### Examples

```bash
# Quick test with 100 articles
python prepare_data.py --download --max-articles 100 --output-file test_100.jsonl

# Process 5000 articles from existing dump
python prepare_data.py --dump-file data/zhwiki-20250320-pages-articles.xml.bz2 --max-articles 5000

# Show stats for existing dataset
python prepare_data.py --stats data/chinese_wiki_5000.jsonl
```

### Step 2: Update Config

After preparing data, update `configs/training_config.yaml`:

```yaml
data:
  train_file: "data/chinese_wiki_articles.jsonl"  # Your output file
```

### Step 3: Train

```bash
python train.py
```

## Option 2: Your Own Data

### Format

Your data should be in **JSONL format** (one JSON object per line):

```json
{"text": "第一段中文文本内容..."}
{"text": "第二段中文文本内容..."}
{"text": "第三段中文文本内容..."}
```

### Create Your Dataset

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

### Update Config

```yaml
data:
  train_file: "data/my_data.jsonl"
```

### Train

```bash
python train.py
```

## Data Requirements

- **Format**: JSONL (one JSON per line)
- **Field**: Must have `text` field
- **Encoding**: UTF-8
- **Size**: Recommended 1000+ samples for good results
- **Language**: Chinese text (for ChatGLM3)

## Current Data Files

Check what data you have:

```bash
ls -lh data/*.jsonl
```

You already have:
- `chinese_wiki_5000.jsonl` - 5000 Wikipedia articles (ready to use)
- `chinese_wiki_100_articles.jsonl` - Small test set
- Other sample files

## Wikipedia Processing Details

The `ChineseWikipediaProcessor` class:

1. **Downloads** from Wikimedia dumps
2. **Extracts** articles from XML
3. **Cleans** text:
   - Removes Wikipedia markup (`[[links]]`, `{{templates}}`)
   - Removes references and categories
   - Cleans whitespace and formatting
4. **Filters** by length
5. **Saves** to JSONL format

See `src/data/wikipedia_processor.py` for implementation details.

## Tips

- Start with small datasets (100-1000 articles) for testing
- Use `--max-articles` to limit processing time
- Wikipedia dumps are large (~3GB), download only once
- Processed JSONL files are much smaller and faster to load
