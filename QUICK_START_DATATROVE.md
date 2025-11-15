# Quick Start: DataTrove Pipeline → HuggingFace Hub

Complete end-to-end guide for processing pretraining data and uploading to HuggingFace Hub.

## Overview

This workflow replicates **SmolLM3's exact data processing pipeline**:

```
Raw Datasets → DataTrove Processing → HuggingFace Hub → Training
  (41 datasets)    (Gopher filters)      (Public/Private)    (Your LLM)
```

## Installation

```bash
# Install all dependencies
pip install -r requirements_datatrove.txt

# Login to HuggingFace (required for uploading)
huggingface-cli login
```

## Complete Workflow

### Step 1: Test with Simple Example (2-5 minutes)

```bash
# Test DataTrove with a small example
python datatrove_simple_example.py simple

# Check output
ls -lh test_output/
```

**What happens**: Downloads a small dataset, applies quality filters, saves processed data.

### Step 2: Test SmolLM3 Pipeline with 5 Datasets (~10-30 minutes)

```bash
# Process 5 datasets from SmolLM3 config
python smollm3_stage1_datatrove_pipeline.py \
  --config smollm3_weighted_config.yaml \
  --output-folder ./datatrove_output/test \
  --max-datasets 5 \
  --tasks 2 \
  --workers 2
```

**What happens**:
- Loads 5 datasets from SmolLM3 collection
- Applies Gopher quality filters + language detection
- Filters spam using bilingual keywords
- Saves to `./datatrove_output/test/`

**Check output**:
```bash
# See what was created
ls -R datatrove_output/test/

# Check size
du -sh datatrove_output/test/

# View a sample document
zcat datatrove_output/test/*/default/0.jsonl.gz | head -1 | python -m json.tool
```

### Step 3: Test Upload to HuggingFace Hub (~2-5 minutes)

```bash
# Upload test data (1,000 documents)
python push_datatrove_to_hub.py \
  --input-folder ./datatrove_output/test \
  --repo-name your-username/test-pretrain-data \
  --max-documents 1000 \
  --output-dir ./test_backup
```

**What happens**:
- Reads all JSONL.gz files
- Combines into HuggingFace Dataset
- Saves local backup to `./test_backup`
- Pushes to `https://huggingface.co/datasets/your-username/test-pretrain-data`

**Verify**:
1. Go to https://huggingface.co/datasets/your-username/test-pretrain-data
2. Check the dataset viewer
3. Verify train/test splits
4. Check dataset card

### Step 4: Full SmolLM3 Stage 1 Pipeline (~4-48 hours)

```bash
# Process all 41 datasets
python smollm3_stage1_datatrove_pipeline.py \
  --config smollm3_weighted_config.yaml \
  --output-folder ./datatrove_output/stage1 \
  --tasks 10 \
  --workers 4
```

**What happens**:
- Processes all 41 SmolLM3 datasets
- Applies production quality filters
- Weighted sampling (37% web, 33% edu, 12% code, 3% math, etc.)
- May take several hours to days depending on hardware

**Monitor progress**:
```bash
# Watch logs
tail -f datatrove_logs/main.log

# Check output size
watch -n 60 'du -sh datatrove_output/stage1/'
```

### Step 5: Upload Full Dataset to Hub

```bash
# Upload all processed data
python push_datatrove_to_hub.py \
  --input-folder ./datatrove_output/stage1 \
  --repo-name your-username/smollm3-stage1-pretrain \
  --output-dir ./stage1_backup \
  --train-test-split 0.95
```

**What happens**:
- Reads millions of documents from JSONL.gz files
- Creates 95% train / 5% test split
- Saves local backup
- Uploads to HuggingFace Hub
- May take several hours for large datasets

**Best practices**:
- Keep `--output-dir` for local backup
- Use meaningful repo names
- Add dataset card (README.md) on HuggingFace after upload

### Step 6: Use Your Dataset for Training

```python
from datasets import load_dataset

# Load your uploaded dataset
dataset = load_dataset("your-username/smollm3-stage1-pretrain")

# Access train split
train_data = dataset['train']
print(f"Training samples: {len(train_data):,}")

# Iterate for training
for sample in train_data:
    text = sample['text']
    source = sample['source']
    weight = sample['weight']
    # Use in your training loop...
```

## Command Reference

### DataTrove Processing

```bash
# Basic usage
python smollm3_stage1_datatrove_pipeline.py \
  --config CONFIG_FILE \
  --output-folder OUTPUT_DIR

# Common options
--max-datasets N      # Process only N datasets (testing)
--tasks N             # Number of parallel tasks (default: 10)
--workers N           # Workers per task (default: 4)
--logging-dir DIR     # Where to save logs
--run-dedup           # Run deduplication after processing
```

### Push to HuggingFace

```bash
# Basic usage
python push_datatrove_to_hub.py \
  --input-folder INPUT_DIR \
  --repo-name username/dataset-name

# Common options
--max-documents N     # Limit documents (testing)
--sample-ratio 0.1    # Use 10% random sample
--output-dir DIR      # Save local backup
--private             # Make dataset private
--no-push             # Only save locally, don't push
--train-test-split 0.95  # Train/test ratio
```

## Troubleshooting

### DataTrove Pipeline Issues

**Issue**: Out of memory
```bash
# Solution: Reduce workers
--workers 1
```

**Issue**: Too slow
```bash
# Solution: Increase parallelization
--tasks 20 --workers 4
```

**Issue**: Dataset not loading
```bash
# Check internet connection
ping huggingface.co

# Check dataset exists
# Go to https://huggingface.co/datasets/DATASET_NAME
```

### Push to Hub Issues

**Issue**: "No JSONL files found"
```bash
# Check files exist
ls -R datatrove_output/stage1/

# Look for .jsonl.gz files
find datatrove_output/stage1/ -name "*.jsonl.gz"
```

**Issue**: Authentication failed
```bash
# Login again
huggingface-cli login

# Check token
huggingface-cli whoami
```

**Issue**: Upload too slow
```bash
# Use sampling for faster test
--sample-ratio 0.1

# Or limit documents
--max-documents 10000
```

## Resource Requirements

### For Testing (5 datasets)
- **CPU**: 4 cores
- **RAM**: 8 GB
- **Disk**: 50 GB
- **Time**: ~10-30 minutes

### For Full Pipeline (41 datasets)
- **CPU**: 16+ cores (recommended)
- **RAM**: 32 GB
- **Disk**: 500 GB - 1 TB
- **Time**: 4-48 hours (depends on hardware)

### For Upload
- **Internet**: Broadband connection
- **Time**: Minutes to hours (depends on dataset size)

## Example Timings

Based on typical hardware:

| Stage | Hardware | Datasets | Time |
|-------|----------|----------|------|
| **Test Example** | 4 cores, 8GB RAM | 1 | 2-5 min |
| **Small Test** | 4 cores, 8GB RAM | 5 | 10-30 min |
| **Full Stage 1** | 16 cores, 32GB RAM | 41 | 8-12 hours |
| **Full Stage 1** | 8 cores, 16GB RAM | 41 | 24-48 hours |
| **Upload Test** | Broadband | 1K docs | 1-2 min |
| **Upload Full** | Broadband | 1M+ docs | 1-4 hours |

## What You Get

After completing this workflow, you have:

✅ **Production-quality processed data** with Gopher filters
✅ **Public dataset on HuggingFace Hub** for sharing/training
✅ **Local backup** of processed data
✅ **Train/test splits** ready for model training
✅ **Metadata preserved** (source, weight, config)
✅ **Same pipeline as SmolLM3** - proven at scale

## Next Steps

### Process Stage 2 and Stage 3

```bash
# Stage 2: Educational focus (8T-9T tokens)
python smollm3_stage1_datatrove_pipeline.py \
  --config smollm3_stage2_config.yaml \
  --output-folder ./datatrove_output/stage2

python push_datatrove_to_hub.py \
  --input-folder ./datatrove_output/stage2 \
  --repo-name username/smollm3-stage2-pretrain

# Stage 3: Reasoning & advanced code (9T-11T tokens)
python smollm3_stage1_datatrove_pipeline.py \
  --config smollm3_stage3_config.yaml \
  --output-folder ./datatrove_output/stage3

python push_datatrove_to_hub.py \
  --input-folder ./datatrove_output/stage3 \
  --repo-name username/smollm3-stage3-pretrain
```

### Add Dataset Card

After uploading, add a README.md to your dataset on HuggingFace:

```markdown
# SmolLM3 Stage 1 Pretraining Dataset

Preprocessed text data for LLM pretraining using SmolLM3's pipeline.

## Dataset Details
- **Total documents**: 1.2M
- **Processing**: DataTrove with Gopher filters
- **Languages**: English (99.675%) + Vietnamese (0.325%)
- **Sources**: 41 datasets from SmolLM3 collection

## Data Composition
- 37% General Web (dclm-baseline)
- 33.3% Educational Web (fineweb-edu)
- 11.9% Code (stack-edu, github)
- 2.7% Math (finemath)
- 9% Multilingual

## Usage
\`\`\`python
from datasets import load_dataset
dataset = load_dataset("username/smollm3-stage1-pretrain")
\`\`\`

## Citation
Based on SmolLM3: https://huggingface.co/blog/smollm3
```

### Use in Training

Your dataset is now ready for training any LLM framework:
- **PyTorch**: DataLoader with HF datasets
- **TensorFlow**: tf.data pipeline
- **JAX/Flax**: Array streaming
- **Megatron-LM**: Direct from HF Hub

## Documentation

- **DATATROVE_GUIDE.md** - Complete DataTrove documentation
- **PUSH_TO_HUB_GUIDE.md** - Complete upload guide
- **DATATROVE_SUMMARY.md** - Quick implementation overview
- **README.md** - Main project documentation

## Summary

You now have a **complete production pipeline**:

1. ✅ **Process** with DataTrove (same as SmolLM3)
2. ✅ **Upload** to HuggingFace Hub (one command)
3. ✅ **Train** your LLM (ready to use)

Start with the test examples, verify the output, then scale up to the full pipeline!

🚀 **Ready to process your pretraining data like SmolLM3!**
