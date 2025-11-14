# Weighted Dataset Sampling Usage Guide

This guide explains how to use the weighted dataset sampling feature with SmolLM3 configurations.

## Overview

The `merge_datasets.py` script now supports **weighted sampling**, which allows you to:
- Sample from datasets according to specific proportions
- Replicate production training recipes like SmolLM3
- Control the exact mix of data types (web, code, math, etc.)
- Ensure balanced representation across multiple datasets

## Requirements

1. A configuration file with `weight` field for each dataset
2. The `--use-weights` flag when running the script
3. All datasets must be in a single `datasets:` list (not separate `english_datasets` and `vietnamese_datasets`)

## Configuration Format

### Weighted Config Structure

```yaml
datasets:
  - name: "HuggingFaceFW/fineweb-edu"
    description: "Educational web content"
    text_field: "text"
    split: "train"
    streaming: true
    weight: 0.333  # 33.3% of the training data
    categories: ["general", "education"]
    verified: true

  - name: "HuggingFaceFW/fineweb-2"
    description: "Vietnamese web content"
    text_field: "text"
    config: "vie_Latn"
    split: "train"
    streaming: true
    weight: 0.00325  # 0.325% of the training data
    categories: ["general", "multilingual"]
    verified: true

  # ... more datasets ...
```

**Important**: All weights should sum to approximately 1.0 (100%).

## Usage Examples

### Basic Weighted Sampling

```bash
python merge_datasets.py \
  --config smollm3_weighted_config.yaml \
  --use-weights \
  --output-dir ./smollm3_stage1_dataset
```

### With Custom Seed (for reproducibility)

```bash
python merge_datasets.py \
  --config smollm3_weighted_config.yaml \
  --use-weights \
  --seed 12345 \
  --output-dir ./smollm3_stage1_dataset
```

### With Maximum Samples (for testing)

```bash
# Generate 100,000 samples with weighted distribution
python merge_datasets.py \
  --config smollm3_weighted_config.yaml \
  --use-weights \
  --max-samples 100000 \
  --output-dir ./test_weighted_dataset
```

### Push to HuggingFace Hub

```bash
python merge_datasets.py \
  --config smollm3_weighted_config.yaml \
  --use-weights \
  --output-dir ./smollm3_dataset \
  --push-to-hub \
  --hub-repo "your-username/smollm3-pretrain-dataset"
```

## SmolLM3 Three-Stage Training

### Stage 1: Foundation (0-8T tokens)

```bash
python merge_datasets.py \
  --config smollm3_weighted_config.yaml \
  --use-weights \
  --output-dir ./smollm3_stage1_dataset \
  --seed 42
```

**Mix**: 73.2% web, 11.9% code, 2.7% math, 0.325% Vietnamese

### Stage 2: Educational Focus (8T-9T tokens)

```bash
python merge_datasets.py \
  --config smollm3_stage2_config.yaml \
  --use-weights \
  --output-dir ./smollm3_stage2_dataset \
  --seed 42
```

**Mix**: 69.5% web, 13.2% code, 9.1% math, 0.005% Vietnamese

### Stage 3: Reasoning & Advanced Code (9T-11T tokens)

```bash
python merge_datasets.py \
  --config smollm3_stage3_config.yaml \
  --use-weights \
  --output-dir ./smollm3_stage3_dataset \
  --seed 42
```

**Mix**: 64.2% web, 20.2% code, 15.6% math+reasoning, 0.005% Vietnamese

## How Weighted Sampling Works

### 1. Load All Datasets

The script loads all datasets in streaming mode for memory efficiency.

### 2. Normalize Weights

Weights are automatically normalized to sum to 1.0:

```python
total_weight = sum(all_weights)
probabilities = [w / total_weight for w in all_weights]
```

### 3. Interleave with Probabilities

Uses HuggingFace's `interleave_datasets` function:

```python
interleaved = interleave_datasets(
    all_datasets,
    probabilities=probabilities,
    seed=seed,
    stopping_strategy="all_exhausted"
)
```

This ensures that samples are drawn from each dataset with probability proportional to its weight.

### 4. Process Samples

Each sample goes through:
- Text extraction
- Cleaning (HTML removal, normalization)
- Filtering (spam detection, relevance check)
- Deduplication

## Weight Verification

To verify your weights are correct:

```python
import yaml

with open('smollm3_weighted_config.yaml') as f:
    config = yaml.safe_load(f)

total_weight = sum(d['weight'] for d in config['datasets'])
print(f"Total weight: {total_weight:.4f}")  # Should be ~1.0

# Check specific categories
web_weight = sum(d['weight'] for d in config['datasets']
                 if 'general' in d.get('categories', []))
code_weight = sum(d['weight'] for d in config['datasets']
                  if 'code' in d.get('categories', []))
math_weight = sum(d['weight'] for d in config['datasets']
                  if 'math' in d.get('categories', []))

print(f"Web: {web_weight*100:.2f}%")
print(f"Code: {code_weight*100:.2f}%")
print(f"Math: {math_weight*100:.2f}%")
```

## Output Logs

When using weighted sampling, you'll see detailed logs:

```
2025-11-14 12:00:00 - INFO - Weighted sampling enabled
2025-11-14 12:00:01 - INFO - Loading datasets for weighted sampling...
2025-11-14 12:00:02 - INFO - Loading mlfoundations/dclm-baseline-1.0 [weight: 0.37]
2025-11-14 12:00:03 - INFO - Loading HuggingFaceFW/fineweb-edu [weight: 0.333]
...
2025-11-14 12:00:30 - INFO - Dataset weights (normalized):
2025-11-14 12:00:30 - INFO -   mlfoundations/dclm-baseline-1.0: 0.3700 (37.00%)
2025-11-14 12:00:30 - INFO -   HuggingFaceFW/fineweb-edu: 0.3330 (33.30%)
2025-11-14 12:00:30 - INFO -   HuggingFaceFW/fineweb-2/vie_Latn: 0.0032 (0.32%)
...
2025-11-14 12:00:31 - INFO - Interleaving datasets with weighted sampling...
Processing weighted samples: 100%|████████████| 100000/100000
2025-11-14 12:15:00 - INFO - Weighted sampling completed: Processed 100000, Kept 95234 (95.23%)
```

## Comparing Weighted vs Non-Weighted

### Non-Weighted (Simple Concatenation)

```bash
# Takes ALL samples from each dataset
python merge_datasets.py \
  --config datasets_config.yaml \
  --output-dir ./simple_merged
```

- Processes each dataset completely
- No control over proportions
- Large datasets dominate the mix
- Different text field names for each dataset group

### Weighted (Proportional Sampling)

```bash
# Samples according to specified proportions
python merge_datasets.py \
  --config smollm3_weighted_config.yaml \
  --use-weights \
  --output-dir ./weighted_merged
```

- Samples from datasets proportionally
- Precise control over data mix
- Balanced representation
- Production-tested recipes (SmolLM3)

## Creating Custom Weighted Configs

### Step 1: Decide Your Mix

Example: 50% web, 30% code, 20% math

### Step 2: Assign Weights

```yaml
datasets:
  # Web content (50%)
  - name: "HuggingFaceFW/fineweb-edu"
    weight: 0.5
    text_field: "text"
    # ... other fields ...

  # Code (30%)
  - name: "HuggingFaceTB/stack-edu"
    config: "python"
    weight: 0.3
    text_field: "text"
    # ... other fields ...

  # Math (20%)
  - name: "HuggingFaceTB/finemath"
    weight: 0.2
    text_field: "text"
    # ... other fields ...
```

### Step 3: Verify Weights Sum to 1.0

```python
import yaml

with open('custom_config.yaml') as f:
    config = yaml.safe_load(f)

total = sum(d['weight'] for d in config['datasets'])
print(f"Total: {total}")  # Should be 1.0

# Normalize if needed
if abs(total - 1.0) > 0.001:
    for d in config['datasets']:
        d['weight'] /= total
```

### Step 4: Test with Small Sample

```bash
python merge_datasets.py \
  --config custom_config.yaml \
  --use-weights \
  --max-samples 10000 \
  --output-dir ./test_custom
```

## Advanced Tips

### 1. Adjust Vietnamese Content

Keep Vietnamese representation high:

```yaml
- name: "HuggingFaceFW/fineweb-2"
  config: "vie_Latn"
  weight: 0.05  # Increase to 5% instead of 0.325%
```

### 2. Focus on Specific Languages

Python-heavy mix:

```yaml
- name: "HuggingFaceTB/stack-edu"
  config: "python-real"
  weight: 0.4  # 40% Python code!
```

### 3. Control Quality with Filtering

Even with weights, filtering still applies:
- Spam detection
- Length requirements
- Relevance scoring
- Deduplication

Adjust filtering in the config file's `filtering` section.

### 4. Reproducibility

Always use the same seed for reproducible results:

```bash
--seed 42
```

### 5. Memory Management

Weighted sampling uses streaming mode automatically, so memory usage is low even with huge datasets.

## Troubleshooting

### Issue: "No weight defined"

**Error**: `Skipping dataset: no weight defined`

**Solution**: Make sure each dataset in your config has a `weight` field.

### Issue: Weights don't sum to 1.0

**Warning**: Weights automatically normalized by the script.

**Best Practice**: Manually verify and adjust weights to sum to 1.0.

### Issue: Dataset not loading

**Error**: `Error loading dataset`

**Solution**:
- Verify dataset name is correct
- Check if `config` parameter is needed
- Ensure you're logged in to HuggingFace if dataset is gated

### Issue: Out of memory

**Solution**: Use `--max-samples` to limit total samples:

```bash
--max-samples 1000000  # 1 million samples
```

### Issue: Different text fields

The weighted sampler tries all text fields from the config. If extraction fails, the sample is skipped.

## Performance Considerations

### Speed

- **Weighted sampling**: Slightly slower due to interleaving overhead
- **Simple concatenation**: Faster but less control

### Memory

- Both modes use streaming, so memory usage is similar and low

### Disk Space

- Weighted: Generates exactly what you specify
- Simple: Can be very large if all datasets are processed fully

## Best Practices

1. **Start with proven configs**: Use SmolLM3 configs as templates
2. **Test with small samples**: Use `--max-samples 10000` first
3. **Use consistent seeds**: For reproducibility across runs
4. **Monitor logs**: Check the normalized weights output
5. **Verify output**: Inspect a few samples to ensure quality
6. **Document your mix**: Keep notes on what weights you used and why

## References

- SmolLM3 Stage 1 Config: `smollm3_weighted_config.yaml`
- SmolLM3 Stage 2 Config: `smollm3_stage2_config.yaml`
- SmolLM3 Stage 3 Config: `smollm3_stage3_config.yaml`
- Training Stages Guide: `SMOLLM3_TRAINING_STAGES.md`
- SmolLM3 Weights Documentation: `SMOLLM3_WEIGHTS.md`

## Questions?

- For general dataset info: See `DATASETS.md`
- For filtering options: See `FILTERING.md`
- For training stages: See `SMOLLM3_TRAINING_STAGES.md`
- For weight details: See `SMOLLM3_WEIGHTS.md`

Happy training! 🚀
