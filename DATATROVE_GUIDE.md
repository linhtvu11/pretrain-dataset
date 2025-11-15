# DataTrove Pipeline for SmolLM3 Pretraining

Complete guide to using DataTrove for production-grade LLM pretraining data preparation, following SmolLM3's approach.

## Table of Contents

1. [What is DataTrove?](#what-is-datatrove)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [SmolLM3 Stage 1 Pipeline](#smollm3-stage-1-pipeline)
5. [Understanding the Pipeline](#understanding-the-pipeline)
6. [Configuration](#configuration)
7. [Advanced Features](#advanced-features)
8. [Production Deployment](#production-deployment)

---

## What is DataTrove?

**DataTrove** is HuggingFace's production-grade library for processing, filtering, and deduplicating text data at massive scale. It's the same tool used to create:

- **FineWeb** - One of the highest quality web datasets
- **SmolLM3** - 11.2T tokens processed across 3 training stages
- **The Stack v2** - 67.5TB of code data

### Key Features

✅ **Production-tested** - Used by HuggingFace for billion-scale datasets
✅ **Memory efficient** - Processes massive datasets with low RAM usage
✅ **Platform agnostic** - Runs locally or on SLURM clusters
✅ **Resumable** - Automatic checkpointing and task completion tracking
✅ **Modular** - Compose pipelines from reusable blocks

### Why DataTrove vs Our Custom Scripts?

| Feature | Custom Scripts | DataTrove |
|---------|---------------|-----------|
| Scale | Limited by memory | Unlimited (streaming) |
| Quality Filters | Basic keyword matching | Production-tested (Gopher, C4, FineWeb) |
| Deduplication | Simple hash-based | Advanced (MinHash, sentence-level, exact substring) |
| Parallelization | Manual | Built-in |
| Checkpointing | Manual | Automatic |
| Used by | Our project | HuggingFace production |

---

## Installation

### Step 1: Install DataTrove

```bash
# Install with all features
pip install datatrove[all]

# Or install our requirements file
pip install -r requirements_datatrove.txt
```

### Step 2: Verify Installation

```bash
python -c "import datatrove; print(f'DataTrove {datatrove.__version__} installed')"
```

### Step 3: Test with Simple Example

```bash
# Run a simple test
python datatrove_simple_example.py simple
```

This will:
- Download a small dataset (Skylion007/openwebtext)
- Apply quality filters
- Save processed data to `./test_output`
- Take ~2-5 minutes

---

## Quick Start

### Example 1: Process Single Dataset

```python
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.filters import LanguageFilter, GopherQualityFilter
from datatrove.pipeline.writers import JsonlWriter

# Define pipeline
pipeline = [
    # Read from HuggingFace
    HuggingFaceDatasetReader(
        dataset="Skylion007/openwebtext",
        dataset_options={"split": "train", "streaming": True},
        text_key="text",
    ),

    # Filter English only
    LanguageFilter(languages=["en"], language_threshold=0.65),

    # Quality filter
    GopherQualityFilter(min_doc_words=50, max_doc_words=100000),

    # Save output
    JsonlWriter(output_folder="./output", output_filename="${rank}.jsonl.gz"),
]

# Execute
executor = LocalPipelineExecutor(
    pipeline=pipeline,
    tasks=4,  # 4 parallel tasks
    workers=2,  # 2 workers per task
    logging_dir="./logs",
)
executor.run()
```

### Example 2: Process Vietnamese Dataset

```bash
python datatrove_simple_example.py vietnamese
```

### Example 3: Process Multilingual Dataset

```bash
python datatrove_simple_example.py multilingual
```

---

## SmolLM3 Stage 1 Pipeline

### Overview

The SmolLM3 Stage 1 pipeline processes **41 datasets** with **weighted sampling** to create a balanced pretraining corpus matching SmolLM3's exact data mix:

- **73.2%** General Web (dclm, fineweb-edu, multilingual)
- **11.9%** Code (Python, C++, Java, etc.)
- **2.7%** Math (finemath, infiwebmath)
- **0.325%** Vietnamese

### Running the Pipeline

#### Quick Test (5 datasets, ~10-30 minutes)

```bash
python smollm3_stage1_datatrove_pipeline.py \
  --config smollm3_weighted_config.yaml \
  --output-folder ./datatrove_output/test \
  --tasks 4 \
  --workers 2 \
  --max-datasets 5
```

#### Full Pipeline (41 datasets, ~hours to days)

```bash
python smollm3_stage1_datatrove_pipeline.py \
  --config smollm3_weighted_config.yaml \
  --output-folder ./datatrove_output/stage1 \
  --tasks 10 \
  --workers 4
```

#### With Deduplication (recommended for production)

```bash
python smollm3_stage1_datatrove_pipeline.py \
  --config smollm3_weighted_config.yaml \
  --output-folder ./datatrove_output/stage1 \
  --tasks 10 \
  --workers 4 \
  --run-dedup
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `smollm3_weighted_config.yaml` | Config file with datasets and weights |
| `--output-folder` | `./datatrove_output/stage1` | Where to save processed data |
| `--tasks` | `10` | Number of parallel tasks |
| `--workers` | `4` | Workers per task (cores) |
| `--logging-dir` | `./datatrove_logs` | Where to save logs |
| `--max-datasets` | `None` | Limit datasets (for testing) |
| `--run-dedup` | `False` | Run deduplication after processing |

---

## Understanding the Pipeline

### Pipeline Stages

The DataTrove pipeline processes data through multiple stages:

```
┌─────────────┐
│   Reader    │  Load data from HuggingFace/S3/local
└──────┬──────┘
       │
┌──────▼──────┐
│   Sampler   │  Sample based on weight (optional)
└──────┬──────┘
       │
┌──────▼──────┐
│   Filters   │  Multiple quality filters in sequence:
│             │  - Language filter (en/vi)
│             │  - Repetition filter (Gopher)
│             │  - Quality filter (Gopher)
│             │  - Spam keyword filter (custom)
└──────┬──────┘
       │
┌──────▼──────┐
│   Writer    │  Save to compressed JSONL
└─────────────┘
```

### Quality Filters Used

#### 1. **LanguageFilter**
```python
LanguageFilter(
    languages=["en", "vi"],  # English and Vietnamese
    language_threshold=0.65  # 65% confidence required
)
```

Removes documents that aren't clearly in English or Vietnamese.

#### 2. **GopherRepetitionFilter**
```python
GopherRepetitionFilter(
    dup_line_frac=0.3,      # Max 30% duplicate lines
    dup_para_frac=0.3,       # Max 30% duplicate paragraphs
    top_n_grams=(2, 3, 4),   # Check 2,3,4-grams
    dup_n_grams=(0.25, 0.25, 0.25),  # Max 25% duplicate n-grams
)
```

Removes documents with excessive repetition (common in low-quality web data).

#### 3. **GopherQualityFilter**
```python
GopherQualityFilter(
    min_doc_words=20,        # At least 20 words (from config min_length)
    max_doc_words=200000,    # Max 200k words (from config max_length)
    min_avg_word_length=3,   # Average word at least 3 chars
    max_avg_word_length=10,  # Average word max 10 chars
    max_symbol_word_ratio=0.1,  # Max 10% symbol-to-word ratio
    max_bullet_lines_ratio=0.9,  # Max 90% bullet point lines
    max_ellipsis_lines_ratio=0.3,  # Max 30% lines with "..."
    max_non_alpha_words_ratio=0.8,  # Max 80% non-alphabetic words
)
```

Comprehensive quality filter based on DeepMind's Gopher paper.

#### 4. **Custom Spam Filter**
```python
LambdaFilter(
    lambda doc: not any(kw in doc.text.lower() for kw in spam_keywords)
)
```

Removes documents containing spam keywords from our config (e.g., "subscribe now", "đăng ký ngay").

---

## Configuration

### Using Weighted Configs

DataTrove pipeline reads from our existing weighted configs:

- `smollm3_weighted_config.yaml` - Stage 1 (0-8T tokens)
- `smollm3_stage2_config.yaml` - Stage 2 (8T-9T tokens)
- `smollm3_stage3_config.yaml` - Stage 3 (9T-11T tokens)

Each dataset entry includes:

```yaml
- name: "HuggingFaceFW/fineweb-edu"
  description: "Educational web content"
  text_field: "text"
  config: null
  split: "train"
  streaming: true
  weight: 0.333  # 33.3% of total data
  categories: ["general", "education"]
  verified: true
```

The pipeline automatically:
1. Reads all datasets
2. Samples according to weights
3. Applies quality filters
4. Saves to organized output folders

### Output Structure

```
datatrove_output/stage1/
├── mlfoundations_dclm-baseline-1.0/
│   └── default/
│       ├── 0.jsonl.gz
│       ├── 1.jsonl.gz
│       └── ...
├── HuggingFaceFW_fineweb-edu/
│   └── default/
│       ├── 0.jsonl.gz
│       └── ...
└── HuggingFaceFW_fineweb-2/
    ├── vie_Latn/
    │   ├── 0.jsonl.gz
    │   └── ...
    └── eng_Latn/
        ├── 0.jsonl.gz
        └── ...
```

Each `.jsonl.gz` file contains processed documents in format:

```json
{"text": "...", "id": "...", "metadata": {"source": "...", "weight": 0.333}}
```

---

## Advanced Features

### 1. Deduplication

DataTrove supports multiple deduplication strategies:

#### Sentence-Level Deduplication

Removes documents with duplicate 3-sentence spans (used in C4):

```python
from datatrove.pipeline.dedup import (
    SentenceDedupSignature,
    SentenceFindDedups,
    SentenceDedupFilter,
)

# Stage 1: Compute signatures
pipeline_sig = [
    JsonlReader("./processed/"),
    SentenceDedupSignature(
        output_folder="./dedup/signatures",
        n_sentences=3,
    ),
]

# Stage 2: Find duplicates
pipeline_find = [
    SentenceFindDedups(
        data_folder="./dedup/signatures",
        output_folder="./dedup/duplicates",
    ),
]

# Stage 3: Filter duplicates
pipeline_filter = [
    JsonlReader("./processed/"),
    SentenceDedupFilter(data_folder="./dedup/duplicates"),
    JsonlWriter("./dedup/final/"),
]
```

#### MinHash Deduplication

For near-duplicate detection at scale (used in FineWeb):

```python
from datatrove.pipeline.dedup import (
    MinhashConfig,
    MinhashDedupSignature,
    MinhashDedupBuckets,
    MinhashDedupFilter,
)

config = MinhashConfig(
    num_buckets=14,
    hashes_per_bucket=8,
    n_grams=5,
)

# Similar 3-stage process as sentence dedup
```

### 2. Parallel Processing

#### Local Multi-Core

```python
LocalPipelineExecutor(
    pipeline=pipeline,
    tasks=10,      # 10 parallel tasks
    workers=4,     # 4 CPU cores per task
)
```

Efficiently uses all CPU cores on your machine.

#### SLURM Cluster

```python
from datatrove.executor import SlurmPipelineExecutor

SlurmPipelineExecutor(
    pipeline=pipeline,
    tasks=100,
    time="10:00:00",  # 10 hours
    partition="gpu",
    cpus_per_task=16,
    mem_per_cpu_gb=4,
)
```

Automatically submits SLURM jobs and manages dependencies.

### 3. Resumable Execution

DataTrove automatically tracks completed tasks:

```python
executor = LocalPipelineExecutor(
    pipeline=pipeline,
    tasks=10,
    workers=4,
    skip_completed=True,  # Skip already-processed tasks
)
```

If pipeline crashes or is interrupted:
- Restart with same command
- Already-completed tasks are skipped
- Processing continues from where it stopped

### 4. Statistics Collection

Collect statistics about your data:

```python
from datatrove.pipeline.stats import (
    StatSigner,
    DocStats,
)

pipeline = [
    JsonlReader("./data/"),
    StatSigner(output_folder="./stats/"),
    JsonlWriter("./output/"),
]
```

Generates statistics about:
- Document lengths
- Language distribution
- Character distributions
- Token counts

---

## Production Deployment

### Resource Requirements

For SmolLM3 Stage 1 (41 datasets):

#### Minimal (Testing)
- **CPU**: 4 cores
- **RAM**: 8 GB
- **Disk**: 100 GB
- **Time**: ~1-2 days
- **Command**: `--tasks 2 --workers 2 --max-datasets 10`

#### Recommended (Production)
- **CPU**: 16+ cores
- **RAM**: 32 GB
- **Disk**: 500 GB - 1 TB
- **Time**: ~12-48 hours
- **Command**: `--tasks 10 --workers 4`

#### High-Performance
- **CPU**: 64+ cores (or SLURM cluster)
- **RAM**: 128 GB+
- **Disk**: 2 TB+ (SSD preferred)
- **Time**: ~4-12 hours
- **Command**: `--tasks 20 --workers 8`

### Monitoring Progress

#### Check Logs

```bash
# Watch overall progress
tail -f datatrove_logs/main.log

# Check specific dataset
tail -f datatrove_logs/HuggingFaceFW_fineweb-edu/task_0.log
```

#### Check Output Size

```bash
# Total processed data size
du -sh datatrove_output/stage1

# Per-dataset breakdown
du -sh datatrove_output/stage1/*
```

#### Count Processed Documents

```bash
# Count documents in output
zcat datatrove_output/stage1/*/*/*.jsonl.gz | wc -l
```

### Performance Optimization

#### 1. Adjust Tasks and Workers

```bash
# More tasks = better parallelization of datasets
# More workers = better CPU utilization per task

# Good for many small datasets:
--tasks 20 --workers 2

# Good for few large datasets:
--tasks 4 --workers 8
```

#### 2. Use Faster Storage

- **SSD** is 10-100x faster than HDD for random I/O
- Use `/tmp` or `/dev/shm` for temporary files if available

#### 3. Stream Large Datasets

All datasets are processed in streaming mode by default - no full download needed!

### Troubleshooting

#### Issue: Out of Memory

**Solution**: Reduce workers per task

```bash
--workers 1  # Use only 1 worker per task
```

#### Issue: Too Slow

**Solutions**:
1. Increase parallelization: `--tasks 20 --workers 4`
2. Use SSD storage
3. Reduce sample size for testing: `--max-datasets 5`

#### Issue: Dataset Not Loading

**Check**:
1. Dataset name is correct (verify on HuggingFace Hub)
2. You have internet connection
3. Dataset is public (or you're logged in with `huggingface-cli login`)

#### Issue: Pipeline Crashes

**Solutions**:
1. Check logs in `datatrove_logs/`
2. Restart with same command (automatic resume)
3. Reduce parallelization to identify problematic dataset

---

## Comparison: Custom Scripts vs DataTrove

### Custom Scripts (merge_datasets.py)

✅ **Pros**:
- Simpler to understand
- Good for small-scale (~1-10M documents)
- Easy to customize

❌ **Cons**:
- Limited scalability
- Basic quality filters
- No production-tested deduplication
- Manual parallelization
- No checkpointing

### DataTrove (smollm3_stage1_datatrove_pipeline.py)

✅ **Pros**:
- Production-grade (used by HuggingFace)
- Unlimited scale (streaming)
- Advanced quality filters (Gopher, C4, FineWeb)
- Multiple deduplication strategies
- Automatic parallelization
- Built-in checkpointing and resume
- Used to create SmolLM3's actual training data

❌ **Cons**:
- More complex to understand
- Larger dependency footprint
- Requires more resources for optimal performance

### Recommendation

- **For experimentation** (<1M documents): Use custom scripts
- **For production training** (>1M documents): Use DataTrove
- **For SmolLM3 replication**: Use DataTrove (it's what they used!)

---

## Next Steps

### 1. Quick Test

```bash
# Test with 1 dataset
python datatrove_simple_example.py simple
```

### 2. Small-Scale Test

```bash
# Test with 5 datasets from SmolLM3 config
python smollm3_stage1_datatrove_pipeline.py \
  --max-datasets 5 \
  --tasks 2 \
  --workers 2
```

### 3. Full Stage 1 Pipeline

```bash
# Process all 41 datasets
python smollm3_stage1_datatrove_pipeline.py \
  --config smollm3_weighted_config.yaml \
  --output-folder ./datatrove_output/stage1 \
  --tasks 10 \
  --workers 4
```

### 4. Stage 2 and Stage 3

Modify the script to use:
- `smollm3_stage2_config.yaml` for Stage 2
- `smollm3_stage3_config.yaml` for Stage 3

### 5. Merge and Prepare for Training

After processing, you'll have JSONL files ready for training. Convert to your training framework's format:

```python
# Example: Convert to Parquet for efficient training
import pyarrow.parquet as pq
import json
import gzip

# Read JSONL
docs = []
with gzip.open('datatrove_output/stage1/dataset/0.jsonl.gz', 'rt') as f:
    for line in f:
        docs.append(json.loads(line))

# Convert to Parquet
# (Implementation depends on your training framework)
```

---

## References

- **DataTrove GitHub**: https://github.com/huggingface/datatrove
- **SmolLM3 Blog**: https://huggingface.co/blog/smollm3
- **FineWeb Dataset**: https://huggingface.co/datasets/HuggingFaceFW/fineweb
- **Gopher Paper**: https://arxiv.org/abs/2112.11446 (quality filters)
- **C4 Paper**: https://arxiv.org/abs/1910.10683 (deduplication)

---

## Support

- **Issues**: Report bugs or questions on our GitHub repo
- **HuggingFace Forums**: https://discuss.huggingface.co/
- **DataTrove Issues**: https://github.com/huggingface/datatrove/issues

---

**Ready to process your data like SmolLM3? Start with the simple examples and scale up! 🚀**
