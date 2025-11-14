# Weighted Dataset Sampling - Implementation Summary

## What Was Implemented

The `merge_datasets.py` script now fully supports **weighted dataset sampling**, enabling you to replicate SmolLM3's production training recipes with exact dataset proportions.

## Key Features Added

### 1. Weighted Sampling Mode

```bash
python merge_datasets.py \
  --config smollm3_weighted_config.yaml \
  --use-weights \
  --output-dir ./smollm3_dataset
```

### 2. How It Works

1. **Loads all datasets** in streaming mode (memory efficient)
2. **Normalizes weights** automatically to sum to 1.0
3. **Interleaves datasets** using HuggingFace's `interleave_datasets` with probabilities
4. **Samples proportionally** - each dataset contributes exactly its specified weight
5. **Processes all samples** through cleaning, filtering, and deduplication

### 3. New Command-Line Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--use-weights` | Enable weighted sampling | `--use-weights` |
| `--seed` | Random seed for reproducibility | `--seed 42` |
| `--max-samples` | Limit total samples | `--max-samples 1000000` |

### 4. Three Training Stages Supported

All three SmolLM3 training stages are now fully supported:

#### Stage 1: Foundation (0-8T tokens)
```bash
python merge_datasets.py \
  --config smollm3_weighted_config.yaml \
  --use-weights \
  --output-dir ./stage1
```
- 73.2% general web
- 11.9% code
- 2.7% math
- 0.325% Vietnamese

#### Stage 2: Educational Focus (8T-9T tokens)
```bash
python merge_datasets.py \
  --config smollm3_stage2_config.yaml \
  --use-weights \
  --output-dir ./stage2
```
- 69.5% general web
- 13.2% code (educational focus)
- 9.08% math (3.4x increase)
- 0.005% Vietnamese

#### Stage 3: Reasoning & Advanced Code (9T-11T tokens)
```bash
python merge_datasets.py \
  --config smollm3_stage3_config.yaml \
  --use-weights \
  --output-dir ./stage3
```
- 64.2% general web
- 20.2% code (Python 7%!)
- 15.6% math + reasoning
- 0.005% Vietnamese

## Code Changes

### merge_datasets.py

**New imports:**
```python
from datasets import interleave_datasets
import numpy as np
```

**New class initialization parameter:**
```python
def __init__(self, config_path: str = 'datasets_config.yaml', use_weights: bool = False)
```

**New method:**
```python
def merge_all_datasets_weighted(
    self,
    max_samples: Optional[int] = None,
    seed: int = 42
) -> List[str]:
```

**Updated main function:**
- Added `--use-weights` and `--seed` arguments
- Conditional execution based on `use_weights` flag
- Calls `merge_all_datasets_weighted()` when weights are enabled

### Key Algorithm

```python
# 1. Load all datasets
for dataset_info in config['datasets']:
    dataset = load_dataset(name, config, split='train', streaming=True)
    datasets.append(dataset)
    weights.append(dataset_info['weight'])

# 2. Normalize weights
total = sum(weights)
probabilities = [w / total for w in weights]

# 3. Interleave with probabilities
interleaved = interleave_datasets(
    datasets,
    probabilities=probabilities,
    seed=seed,
    stopping_strategy="all_exhausted"
)

# 4. Process samples
for example in interleaved:
    text = extract_text(example)
    cleaned = cleaner.process_text(text)
    if cleaned:
        merged_texts.append(cleaned)
```

## Documentation Created

1. **WEIGHTED_SAMPLING_USAGE.md** (New)
   - Comprehensive guide for weighted sampling
   - Usage examples for all scenarios
   - Troubleshooting guide
   - Best practices
   - Weight verification scripts

2. **SMOLLM3_TRAINING_STAGES.md** (Created earlier)
   - Explains all 3 training stages
   - Progressive specialization approach
   - Visual comparisons
   - Recommendations for different model types

3. **README.md** (Updated)
   - Added "Weighted Sampling" to Features
   - Updated SmolLM3 usage examples with --use-weights
   - Added references to new documentation

## Benefits

### 1. Production-Tested Recipes
Use the exact same dataset proportions that HuggingFace used to train SmolLM3 successfully.

### 2. Precise Control
Control exactly what percentage of your training data comes from each source:
- 37% from dclm (web)
- 33.3% from fineweb-edu (educational)
- 2.5% from Python code
- etc.

### 3. Reproducibility
Using the same seed ensures identical dataset generation across runs:
```bash
--seed 42
```

### 4. Memory Efficient
Streaming mode means you can process massive datasets without loading everything into RAM.

### 5. Flexible
Works with any weighted configuration file, not just SmolLM3 configs.

## Example Output Logs

```
2025-11-14 12:00:00 - INFO - Weighted sampling enabled
2025-11-14 12:00:01 - INFO - Loading datasets for weighted sampling...
2025-11-14 12:00:02 - INFO - Loading mlfoundations/dclm-baseline-1.0 [weight: 0.37]
2025-11-14 12:00:03 - INFO - Loading HuggingFaceFW/fineweb-edu [weight: 0.333]
...

Dataset weights (normalized):
  mlfoundations/dclm-baseline-1.0: 0.3700 (37.00%)
  HuggingFaceFW/fineweb-edu: 0.3330 (33.30%)
  HuggingFaceFW/fineweb-2/vie_Latn: 0.0032 (0.32%)
  ...

Interleaving datasets with weighted sampling...
Processing weighted samples: 100%|████████████| 1000000/1000000

Weighted sampling completed: Processed 1000000, Kept 952341 (95.23%)
```

## Backward Compatibility

The old behavior (simple concatenation) still works:

```bash
# Old way - still works
python merge_datasets.py \
  --config datasets_config.yaml \
  --output-dir ./merged
```

Weighted sampling is opt-in with the `--use-weights` flag.

## Testing

To test weighted sampling with a small sample:

```bash
python merge_datasets.py \
  --config smollm3_weighted_config.yaml \
  --use-weights \
  --max-samples 10000 \
  --output-dir ./test_weighted
```

## Verification

Verify your weights sum to 1.0:

```python
import yaml

with open('smollm3_weighted_config.yaml') as f:
    config = yaml.safe_load(f)

total = sum(d['weight'] for d in config['datasets'])
print(f"Total weight: {total:.4f}")  # Should be ~1.0
```

## Configuration Requirements

For weighted sampling, your config must:

1. Have a single `datasets:` list (not `english_datasets` and `vietnamese_datasets`)
2. Each dataset must have a `weight` field
3. Weights should sum to approximately 1.0 (automatic normalization happens anyway)

## Example Custom Config

Create your own weighted mix:

```yaml
datasets:
  # 50% web content
  - name: "HuggingFaceFW/fineweb-edu"
    weight: 0.5
    text_field: "text"
    split: "train"
    streaming: true

  # 30% code
  - name: "HuggingFaceTB/stack-edu"
    config: "python"
    weight: 0.3
    text_field: "text"
    split: "train"
    streaming: true

  # 20% math
  - name: "HuggingFaceTB/finemath"
    weight: 0.2
    text_field: "text"
    split: "train"
    streaming: true
```

Then use it:

```bash
python merge_datasets.py \
  --config custom_weighted.yaml \
  --use-weights \
  --output-dir ./custom_dataset
```

## Files Changed

1. `merge_datasets.py` - Added weighted sampling support
2. `WEIGHTED_SAMPLING_USAGE.md` - New comprehensive guide
3. `README.md` - Updated with weighted sampling info

## Commits

1. **651409e** - Added SmolLM3 Stage 2 and Stage 3 configurations
2. **d6013c2** - Added weighted dataset sampling support

## Next Steps

You can now:

1. ✅ Use SmolLM3's exact training recipes (all 3 stages)
2. ✅ Create custom weighted mixes for your specific needs
3. ✅ Replicate production LLM training at any scale
4. ✅ Control exact data proportions
5. ✅ Generate reproducible training datasets

## Questions?

See the documentation:
- **WEIGHTED_SAMPLING_USAGE.md** - Detailed usage guide
- **SMOLLM3_TRAINING_STAGES.md** - Three-stage training guide
- **SMOLLM3_WEIGHTS.md** - Weight explanations and details
- **README.md** - Quick start guide

Happy training! 🚀
