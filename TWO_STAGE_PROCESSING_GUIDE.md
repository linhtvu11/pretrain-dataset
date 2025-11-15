# Three-Stage Dataset Processing Guide

This guide explains the improved three-stage approach for dataset processing.

## Why Three Stages?

### Old Approach (One-Stage)
```
Download → Clean → Save ❌
```
- Re-download every time you change cleaning parameters
- Can't inspect raw data
- Wastes bandwidth
- Slow iteration

### New Approach (Three-Stage)
```
Stage 1: Download → Sample by weight → Save raw        ✅
Stage 2: Load raw → Clean/filter → Save cleaned        ✅
Stage 3: Combine all → Push to HuggingFace Hub         ✅
```
- Download once, clean many times
- Inspect raw data before cleaning
- Fast iteration on cleaning logic
- Combine and publish to HuggingFace Hub
- Saves bandwidth and time

## Quick Start

### Stage 1: Download and Sample

```bash
python merge_datasets_two_stage.py \
  --stage 1 \
  --config smollm3_weighted_config.yaml \
  --raw-folder ./raw_data \
  --target-samples 100000 \
  --seed 42
```

**What it does**:
1. Reads your weighted config (e.g., 37% web, 33% edu, 12% code...)
2. Calculates samples per dataset based on weights
3. Downloads from HuggingFace with streaming
4. **Randomly samples** exact number needed for each dataset
5. Saves raw (uncleaned) data to `./raw_data/`

**Example output**:
```
Dataset sampling plan:
  mlfoundations/dclm-baseline-1.0: 37,000 samples (37.00%)
  HuggingFaceFW/fineweb-edu: 33,300 samples (33.30%)
  HuggingFaceTB/stack-edu_python: 2,500 samples (2.50%)
  HuggingFaceFW/fineweb-2_vie_Latn: 325 samples (0.33%)
  ...
```

### Stage 2: Clean and Filter

```bash
python merge_datasets_two_stage.py \
  --stage 2 \
  --config smollm3_weighted_config.yaml \
  --raw-folder ./raw_data \
  --cleaned-folder ./cleaned_data
```

**What it does**:
1. Loads all raw data from `./raw_data/`
2. Applies cleaning (HTML removal, Unicode normalization, etc.)
3. Applies filtering (spam detection, length checks, quality checks)
4. Applies deduplication
5. Saves cleaned data to `./cleaned_data/`

**Preserves metadata**:
- Source dataset
- Weight
- Config
- All original metadata

### Stage 3: Combine and Push to Hub

```bash
python merge_datasets_two_stage.py \
  --stage 3 \
  --cleaned-folder ./cleaned_data \
  --repo-id username/my-pretrain-dataset \
  --private \
  --train-test-split 0.1
```

**What it does**:
1. Loads all cleaned data from `./cleaned_data/`
2. Combines all datasets into a single HuggingFace Dataset
3. Shows statistics by source dataset
4. Shuffles for better train/test distribution
5. Splits into train/test (default 90%/10%, use `--train-test-split 0` to disable)
6. Pushes to HuggingFace Hub

**Output Dataset Features**:
- `text`: The cleaned text content
- `source`: Original dataset name
- `config`: Dataset configuration used
- `weight`: Weight used for sampling

**Parameters**:
- `--repo-id`: Required. Your HuggingFace repo (e.g., `username/dataset-name`)
- `--private`: Optional. Make dataset private (default: public)
- `--train-test-split`: Optional. Test split ratio (default: 0.1 = 10% test, use 0 to disable)
- `--max-shard-size`: Optional. Max shard size for upload (default: "500MB")

**Requirements**:
- Must be logged in: `huggingface-cli login`
- Repository will be created automatically if it doesn't exist

## Usage Examples

### Example 1: Quick Test (10,000 samples)

```bash
# Stage 1: Download 10K samples
python merge_datasets_two_stage.py \
  --stage 1 \
  --target-samples 10000 \
  --raw-folder ./test_raw

# Stage 2: Clean them
python merge_datasets_two_stage.py \
  --stage 2 \
  --raw-folder ./test_raw \
  --cleaned-folder ./test_cleaned

# Stage 3: Push to Hub
python merge_datasets_two_stage.py \
  --stage 3 \
  --cleaned-folder ./test_cleaned \
  --repo-id username/test-dataset \
  --train-test-split 0.1
```

Time: ~5-10 minutes total

### Example 2: Medium Dataset (1M samples)

```bash
# Stage 1: Download 1M samples (may take 30-60 minutes)
python merge_datasets_two_stage.py \
  --stage 1 \
  --target-samples 1000000 \
  --raw-folder ./raw_1M

# Stage 2: Clean them (may take 10-20 minutes)
python merge_datasets_two_stage.py \
  --stage 2 \
  --raw-folder ./raw_1M \
  --cleaned-folder ./cleaned_1M

# Stage 3: Push to Hub (may take 5-10 minutes)
python merge_datasets_two_stage.py \
  --stage 3 \
  --cleaned-folder ./cleaned_1M \
  --repo-id username/pretrain-1M \
  --train-test-split 0.1
```

Time: ~1-2 hours total

### Example 3: Large Dataset (10M samples)

```bash
# Stage 1: Download 10M samples
python merge_datasets_two_stage.py \
  --stage 1 \
  --target-samples 10000000 \
  --raw-folder ./raw_10M

# Stage 2: Clean them
python merge_datasets_two_stage.py \
  --stage 2 \
  --raw-folder ./raw_10M \
  --cleaned-folder ./cleaned_10M

# Stage 3: Push to Hub (large dataset may take 30-60 minutes)
python merge_datasets_two_stage.py \
  --stage 3 \
  --cleaned-folder ./cleaned_10M \
  --repo-id username/pretrain-10M \
  --train-test-split 0.05 \
  --max-shard-size "1GB"
```

Time: Several hours to a day

### Example 4: Complete Workflow (All Three Stages)

```bash
# Login to HuggingFace first
huggingface-cli login

# Stage 1: Download 100K samples with SmolLM3 weights
python merge_datasets_two_stage.py \
  --stage 1 \
  --config smollm3_weighted_config.yaml \
  --target-samples 100000 \
  --raw-folder ./my_dataset_raw \
  --seed 42

# Stage 2: Clean and filter
python merge_datasets_two_stage.py \
  --stage 2 \
  --config smollm3_weighted_config.yaml \
  --raw-folder ./my_dataset_raw \
  --cleaned-folder ./my_dataset_cleaned

# Stage 3: Push to HuggingFace Hub
python merge_datasets_two_stage.py \
  --stage 3 \
  --cleaned-folder ./my_dataset_cleaned \
  --repo-id your-username/smollm3-pretrain-100k \
  --train-test-split 0.1 \
  --private
```

**Result**:
- Dataset available at: `https://huggingface.co/datasets/your-username/smollm3-pretrain-100k`
- Train split: ~90K samples
- Test split: ~10K samples
- Features: text, source, config, weight

## How Sampling Works

### Weighted Random Sampling

For each dataset, the script calculates:

```
target_samples = total_samples × (dataset_weight / total_weight)
```

**Example** with 1M total samples:
- fineweb-edu (weight 0.333): 333,000 samples
- dclm (weight 0.37): 370,000 samples
- stack-edu/python (weight 0.025): 25,000 samples
- fineweb-2/vie_Latn (weight 0.00325): 3,250 samples

### Optimized Shuffle + Take Sampling

Uses **HuggingFace's built-in `.shuffle()` and `.take()` methods** for efficient random sampling:

1. Shuffle dataset with buffer (e.g., 100K samples)
2. Take exactly N samples needed
3. Save to disk

**Why This is Fast**:
- ✅ No need to process millions of samples
- ✅ HuggingFace optimized implementation
- ✅ Only downloads what you need
- ✅ Much faster than manual reservoir sampling

**Example Performance**:
- **Old method (reservoir)**: Process 6.9M samples to collect 37K (20+ minutes)
- **New method (shuffle+take)**: Process ~74K samples to collect 37K (~1 minute)

Benefits:
- ✅ Works with streaming (no need to load full dataset)
- ✅ Good random distribution (buffer-based)
- ✅ Memory efficient
- ✅ Reproducible with seed
- ✅ **10-20x faster** than reservoir sampling

## Output Structure

### After Stage 1 (Raw Data)

```
raw_data/
├── mlfoundations_dclm-baseline-1.0/
│   └── raw_data.jsonl.gz
├── HuggingFaceFW_fineweb-edu/
│   └── raw_data.jsonl.gz
├── HuggingFaceFW_fineweb-2_vie_Latn/
│   └── raw_data.jsonl.gz
└── HuggingFaceTB_stack-edu_python/
    └── raw_data.jsonl.gz
```

**Each JSONL line**:
```json
{
  "text": "Raw uncleaned text...",
  "metadata": {
    "source": "HuggingFaceFW/fineweb-edu",
    "config": "default",
    "weight": 0.333,
    "original_index": 12345
  }
}
```

### After Stage 2 (Cleaned Data)

```
cleaned_data/
├── mlfoundations_dclm-baseline-1.0/
│   └── cleaned_data.jsonl.gz
├── HuggingFaceFW_fineweb-edu/
│   └── cleaned_data.jsonl.gz
└── ...
```

**Each JSONL line**:
```json
{
  "text": "Cleaned and filtered text...",
  "metadata": {
    "source": "HuggingFaceFW/fineweb-edu",
    "config": "default",
    "weight": 0.333,
    "original_index": 12345
  }
}
```

## Inspecting Data

### Check Raw Data Before Cleaning

```bash
# View a few samples from raw data
zcat raw_data/HuggingFaceFW_fineweb-edu/raw_data.jsonl.gz | head -5 | python -m json.tool

# Count raw samples
zcat raw_data/HuggingFaceFW_fineweb-edu/raw_data.jsonl.gz | wc -l

# Check total size
du -sh raw_data/
```

### Check Cleaned Data

```bash
# View cleaned samples
zcat cleaned_data/HuggingFaceFW_fineweb-edu/cleaned_data.jsonl.gz | head -5 | python -m json.tool

# Count cleaned samples (compare to raw)
zcat cleaned_data/HuggingFaceFW_fineweb-edu/cleaned_data.jsonl.gz | wc -l
```

## Iterating on Cleaning

This is where the two-stage approach shines!

### Try Different Cleaning Parameters

1. **Modify your config** (`smollm3_weighted_config.yaml`):
   ```yaml
   filtering:
     min_length: 200  # Change from 100
     max_length: 500000  # Change from 1000000
     # Add/remove keywords...
   ```

2. **Re-run Stage 2 only**:
   ```bash
   # Delete old cleaned data
   rm -rf cleaned_data/

   # Re-clean with new parameters
   python merge_datasets_two_stage.py \
     --stage 2 \
     --raw-folder ./raw_data \
     --cleaned-folder ./cleaned_data
   ```

3. **Compare results**:
   - No re-downloading!
   - Fast iteration (minutes, not hours)
   - Easy to compare different cleaning approaches

## Advanced Usage

### Use Different Configs for Stage 1 and Stage 2

```bash
# Stage 1: Download with Stage 1 weights
python merge_datasets_two_stage.py \
  --stage 1 \
  --config smollm3_weighted_config.yaml \
  --raw-folder ./raw_stage1

# Stage 2: Clean with different filtering config
python merge_datasets_two_stage.py \
  --stage 2 \
  --config my_custom_filtering_config.yaml \
  --raw-folder ./raw_stage1 \
  --cleaned-folder ./cleaned_stage1
```

### Process Multiple Stages

```bash
# Process all 3 SmolLM3 stages

# Stage 1 data
python merge_datasets_two_stage.py --stage 1 --config smollm3_weighted_config.yaml --raw-folder ./raw_s1 --target-samples 1000000
python merge_datasets_two_stage.py --stage 2 --config smollm3_weighted_config.yaml --raw-folder ./raw_s1 --cleaned-folder ./cleaned_s1

# Stage 2 data (different weights)
python merge_datasets_two_stage.py --stage 1 --config smollm3_stage2_config.yaml --raw-folder ./raw_s2 --target-samples 1000000
python merge_datasets_two_stage.py --stage 2 --config smollm3_stage2_config.yaml --raw-folder ./raw_s2 --cleaned-folder ./cleaned_s2

# Stage 3 data (different weights again)
python merge_datasets_two_stage.py --stage 1 --config smollm3_stage3_config.yaml --raw-folder ./raw_s3 --target-samples 1000000
python merge_datasets_two_stage.py --stage 2 --config smollm3_stage3_config.yaml --raw-folder ./raw_s3 --cleaned-folder ./cleaned_s3
```

## Performance Tips

### Stage 1 (Download)

**Optimized Sampling** (NEW):
The script now uses HuggingFace's `.shuffle()` + `.take()` for **10-20x faster** sampling:
- Processes only 2x target samples (instead of 10x)
- Typical speedup: 20 minutes → 1-2 minutes per dataset
- Same random quality with much better performance

**Speed up downloading**:
1. Use faster internet connection
2. Run during off-peak hours
3. If interrupted, just re-run (it skips already downloaded files)
4. For very large datasets, the shuffle buffer automatically adjusts

**Disk space**:
- 10K samples: ~50-100 MB
- 100K samples: ~500 MB - 1 GB
- 1M samples: ~5-10 GB
- 10M samples: ~50-100 GB

### Stage 2 (Clean)

**Speed up cleaning**:
1. Already fast (processing local files)
2. Can run in parallel if you modify the script
3. Typically much faster than Stage 1

**Filter ratio**:
- Expect ~70-95% of raw data to pass filtering
- Depends on your filtering criteria
- Check logs for statistics

## Comparison: Two-Stage vs DataTrove

| Feature | Two-Stage Script | DataTrove Pipeline |
|---------|-----------------|-------------------|
| **Sampling** | Weight-based random sampling | Weight-based interleaving |
| **Cleaning** | Custom bilingual filters | Gopher + Language filters |
| **Deduplication** | Simple hash-based | Advanced (MinHash, sentence) |
| **Best for** | Custom cleaning logic | Production-grade quality filters |
| **Iteration** | Very fast (Stage 2 only) | Slower (re-process all) |
| **Scale** | Up to 100M samples | Unlimited |
| **Complexity** | Simple, easy to understand | More complex |

**Recommendation**:
- **Two-stage script**: For custom filtering, experimentation, <10M samples
- **DataTrove**: For production training, >10M samples, proven quality filters

## Troubleshooting

### Stage 1 Issues

**Issue**: "Dataset not loading"
```bash
# Check dataset exists
# Go to https://huggingface.co/datasets/DATASET_NAME

# Check internet connection
ping huggingface.co
```

**Issue**: "Out of disk space"
```bash
# Check available space
df -h

# Reduce target samples
--target-samples 10000
```

**Issue**: "Too slow"
```bash
# Reduce target samples for testing
--target-samples 10000

# Or just wait - Stage 1 is one-time only
```

### Stage 2 Issues

**Issue**: "No raw data found"
```bash
# Check raw folder exists
ls -R raw_data/

# Make sure Stage 1 completed
```

**Issue**: "Too much filtered out"
```bash
# Check your filtering config
cat smollm3_weighted_config.yaml

# Relax filtering criteria
filtering:
  min_length: 50  # Reduce from 100
  exclude_keywords: []  # Remove some keywords
```

## Next Steps

### After Cleaning

You have cleaned data ready for:

1. **Push to HuggingFace Hub**:
   ```bash
   python push_datatrove_to_hub.py \
     --input-folder ./cleaned_data \
     --repo-name username/smollm3-stage1-pretrain
   ```

2. **Use for training**:
   ```python
   # Load cleaned data
   import json, gzip

   for file in cleaned_files:
       with gzip.open(file, 'rt') as f:
           for line in f:
               doc = json.loads(line)
               text = doc['text']
               # Use in training...
   ```

3. **Merge into single file**:
   ```bash
   # Combine all cleaned data
   cat cleaned_data/*/cleaned_data.jsonl.gz > all_cleaned.jsonl.gz
   ```

## Summary

### Benefits of Two-Stage Approach

✅ **Download once** - Save bandwidth and time
✅ **Iterate fast** - Experiment with cleaning in minutes
✅ **Inspect raw** - See data before cleaning
✅ **Reproducible** - Same seed = same samples
✅ **Weighted sampling** - Exact proportions from config
✅ **Preserves metadata** - Track source, weight, etc.

### Workflow

```bash
# 1. Download and sample (one time)
python merge_datasets_two_stage.py --stage 1 --target-samples 1000000

# 2. Clean (fast, can repeat many times)
python merge_datasets_two_stage.py --stage 2

# 3. Experiment with different cleaning (minutes, not hours)
# Modify config, then:
rm -rf cleaned_data/
python merge_datasets_two_stage.py --stage 2

# 4. Push to HuggingFace when happy with results
python push_datatrove_to_hub.py --input-folder ./cleaned_data --repo-name user/dataset
```

**You now have full control over your dataset processing pipeline! 🚀**
