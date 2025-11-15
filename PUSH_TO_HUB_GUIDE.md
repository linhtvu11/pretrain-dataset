# Push DataTrove Data to HuggingFace Hub

Complete guide for pushing your processed DataTrove data to HuggingFace Hub.

## Overview

After running `smollm3_stage1_datatrove_pipeline.py`, you have JSONL.gz files with processed data:

```
datatrove_output/stage1/
├── mlfoundations_dclm-baseline-1.0/
│   └── default/
│       ├── 0.jsonl.gz
│       ├── 1.jsonl.gz
│       └── ...
├── HuggingFaceFW_fineweb-edu/
│   └── default/
│       └── 0.jsonl.gz
└── ...
```

The `push_datatrove_to_hub.py` script:
1. Reads all JSONL.gz files recursively
2. Combines them into a HuggingFace Dataset
3. Splits into train/test sets
4. Pushes to HuggingFace Hub

## Prerequisites

### 1. Install Dependencies

Already included in requirements:
```bash
pip install datasets  # Already in requirements.txt
```

### 2. Login to HuggingFace

```bash
huggingface-cli login
```

Enter your HuggingFace token when prompted. Get your token from: https://huggingface.co/settings/tokens

### 3. Create Repository (Optional)

You can create the repository manually on HuggingFace, or let the script create it automatically.

Manual creation:
1. Go to https://huggingface.co/new-dataset
2. Enter dataset name (e.g., "smollm3-stage1-pretrain")
3. Choose public or private
4. Click "Create dataset"

## Usage Examples

### Quick Test (Small Sample)

Test with just 1,000 documents:

```bash
python push_datatrove_to_hub.py \
  --input-folder ./datatrove_output/stage1 \
  --repo-name your-username/test-dataset \
  --max-documents 1000 \
  --output-dir ./test_dataset_local
```

This will:
- Load only 1,000 documents
- Save locally to `./test_dataset_local`
- Push to `your-username/test-dataset`

### Push Full Dataset

Push all processed data:

```bash
python push_datatrove_to_hub.py \
  --input-folder ./datatrove_output/stage1 \
  --repo-name your-username/smollm3-stage1-pretrain
```

### Push with Sampling

Push 10% random sample:

```bash
python push_datatrove_to_hub.py \
  --input-folder ./datatrove_output/stage1 \
  --repo-name your-username/smollm3-stage1-sample \
  --sample-ratio 0.1
```

### Save Locally Only (No Push)

```bash
python push_datatrove_to_hub.py \
  --input-folder ./datatrove_output/stage1 \
  --repo-name your-username/dataset \
  --output-dir ./local_dataset \
  --no-push
```

### Push Private Dataset

```bash
python push_datatrove_to_hub.py \
  --input-folder ./datatrove_output/stage1 \
  --repo-name your-username/private-dataset \
  --private
```

### Custom Train/Test Split

Use 90% train, 10% test:

```bash
python push_datatrove_to_hub.py \
  --input-folder ./datatrove_output/stage1 \
  --repo-name your-username/dataset \
  --train-test-split 0.9
```

## Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input-folder` | ✅ Yes | - | Folder with DataTrove JSONL.gz files |
| `--repo-name` | ✅ Yes | - | HuggingFace repo (username/dataset-name) |
| `--private` | No | `False` | Make dataset private |
| `--output-dir` | No | `None` | Save locally (optional) |
| `--max-documents` | No | `None` | Limit documents (for testing) |
| `--sample-ratio` | No | `1.0` | Sample ratio (0.0-1.0) |
| `--train-test-split` | No | `0.95` | Train/test split (0.0-1.0) |
| `--no-push` | No | `False` | Skip HuggingFace push |
| `--token` | No | `None` | HuggingFace token (uses cached if not provided) |

## Output Format

### Dataset Structure

The pushed dataset has this structure:

```python
DatasetDict({
    train: Dataset({
        features: {
            'text': Value(dtype='string'),
            'id': Value(dtype='string'),
            'source': Value(dtype='string'),
            'weight': Value(dtype='float64'),
            'config': Value(dtype='string'),
        },
        num_rows: 950000  # 95% of total
    }),
    test: Dataset({
        features: { ... },
        num_rows: 50000  # 5% of total
    })
})
```

### Fields

- **text**: The actual document text (cleaned and filtered)
- **id**: Unique document identifier
- **source**: Source dataset (e.g., "HuggingFaceFW/fineweb-edu")
- **weight**: Dataset weight from config (e.g., 0.333 for 33.3%)
- **config**: Dataset config/subset (e.g., "vie_Latn" for Vietnamese)

### Example Document

```python
{
    'text': 'This is a sample document about machine learning...',
    'id': 'doc_12345',
    'source': 'HuggingFaceFW/fineweb-edu',
    'weight': 0.333,
    'config': 'default'
}
```

## Workflow Example

### Complete End-to-End Workflow

```bash
# Step 1: Process data with DataTrove (Stage 1)
python smollm3_stage1_datatrove_pipeline.py \
  --config smollm3_weighted_config.yaml \
  --output-folder ./datatrove_output/stage1 \
  --max-datasets 5  # Test with 5 datasets first

# Step 2: Check the output
ls -lh datatrove_output/stage1/
du -sh datatrove_output/stage1/

# Step 3: Test push with small sample
python push_datatrove_to_hub.py \
  --input-folder ./datatrove_output/stage1 \
  --repo-name your-username/test-stage1 \
  --max-documents 1000 \
  --output-dir ./test_local

# Step 4: Verify the test dataset
# Go to https://huggingface.co/datasets/your-username/test-stage1

# Step 5: Push full dataset
python push_datatrove_to_hub.py \
  --input-folder ./datatrove_output/stage1 \
  --repo-name your-username/smollm3-stage1-pretrain \
  --output-dir ./stage1_backup  # Keep local backup
```

## Statistics Output

The script prints detailed statistics:

```
================================================================================
Dataset Statistics
================================================================================

Total documents: 1,234,567

Documents by source:
  HuggingFaceFW/fineweb-edu: 456,789 (37.00%)
  mlfoundations/dclm-baseline-1.0: 411,111 (33.30%)
  HuggingFaceFW/fineweb-2: 111,111 (9.00%)
  ...

Text length statistics:
  Min: 125 chars
  Max: 125,456 chars
  Mean: 2,345 chars
  Median: 1,876 chars
================================================================================
```

## Using the Dataset

Once pushed, you can use the dataset in your training:

### Load from Hub

```python
from datasets import load_dataset

# Load full dataset
dataset = load_dataset("your-username/smollm3-stage1-pretrain")

# Access train split
train_data = dataset['train']
print(f"Train samples: {len(train_data)}")

# Iterate over samples
for sample in train_data:
    text = sample['text']
    source = sample['source']
    weight = sample['weight']
    # ... use for training
```

### Streaming Mode

For large datasets, use streaming:

```python
dataset = load_dataset(
    "your-username/smollm3-stage1-pretrain",
    split="train",
    streaming=True
)

for sample in dataset:
    print(sample['text'][:100])  # First 100 chars
```

### Filter by Source

```python
# Load only specific sources
dataset = load_dataset("your-username/smollm3-stage1-pretrain", split="train")

# Filter for fineweb-edu only
fineweb = dataset.filter(lambda x: "fineweb-edu" in x['source'])
print(f"FineWeb-edu samples: {len(fineweb)}")
```

### Filter by Language/Config

```python
# Vietnamese documents only
vietnamese = dataset.filter(lambda x: x['config'] == 'vie_Latn')

# English documents only
english = dataset.filter(lambda x: x['config'] in ['default', 'eng_Latn'])
```

## Troubleshooting

### Issue: "No JSONL files found"

**Check**:
```bash
# Verify files exist
ls -R datatrove_output/stage1/

# Check file extensions
find datatrove_output/stage1/ -name "*.jsonl.gz"
```

**Solution**: Make sure DataTrove pipeline completed successfully.

### Issue: "Authentication failed"

**Solutions**:
1. Login again: `huggingface-cli login`
2. Check token permissions: Must have "write" access
3. Provide token explicitly: `--token YOUR_TOKEN`

### Issue: "Repository not found"

**Solutions**:
1. Create repository first on HuggingFace
2. Check repository name format: `username/dataset-name`
3. Verify you have permissions to write to the repository

### Issue: "Out of memory"

**Solutions**:
1. Process in smaller batches: `--max-documents 100000`
2. Use sampling: `--sample-ratio 0.5`
3. Save locally first: `--output-dir ./local --no-push`
4. Then push the saved dataset:
   ```python
   from datasets import load_from_disk
   dataset = load_from_disk("./local")
   dataset.push_to_hub("username/dataset-name")
   ```

### Issue: "Upload too slow"

**Solutions**:
1. Use faster internet connection
2. Push during off-peak hours
3. Split into multiple smaller datasets
4. Use HuggingFace's large file upload tools

## Best Practices

### 1. Test First

Always test with a small sample before pushing the full dataset:

```bash
# Test with 1,000 documents
python push_datatrove_to_hub.py \
  --input-folder ./datatrove_output/stage1 \
  --repo-name your-username/test \
  --max-documents 1000
```

### 2. Keep Local Backup

Save a local copy before pushing:

```bash
python push_datatrove_to_hub.py \
  --input-folder ./datatrove_output/stage1 \
  --repo-name your-username/dataset \
  --output-dir ./backup  # Local backup
```

### 3. Use Meaningful Names

Good repository names:
- `username/smollm3-stage1-pretrain`
- `username/vietnamese-web-pretrain`
- `username/code-math-pretrain-mix`

Bad names:
- `username/data`
- `username/test123`
- `username/mydataset`

### 4. Add Dataset Card

After pushing, add a README.md to your dataset repository:

```markdown
# SmolLM3 Stage 1 Pretraining Dataset

This dataset contains preprocessed text data for LLM pretraining, following SmolLM3's approach.

## Dataset Details

- **Total documents**: 1.2M
- **Source**: 41 datasets from SmolLM3 collection
- **Processing**: DataTrove with Gopher quality filters
- **Languages**: English (99.675%) + Vietnamese (0.325%)

## Data Composition

- 37% General Web (dclm-baseline)
- 33.3% Educational Web (fineweb-edu)
- 11.9% Code (stack-edu, github)
- 2.7% Math (finemath)
- 9% Multilingual content

## Usage

\`\`\`python
from datasets import load_dataset

dataset = load_dataset("your-username/smollm3-stage1-pretrain")
\`\`\`

## Citation

Based on SmolLM3: https://huggingface.co/blog/smollm3
```

### 5. Version Your Datasets

For multiple stages or versions:

```bash
# Stage 1
python push_datatrove_to_hub.py \
  --input-folder ./datatrove_output/stage1 \
  --repo-name your-username/pretrain-stage1

# Stage 2
python push_datatrove_to_hub.py \
  --input-folder ./datatrove_output/stage2 \
  --repo-name your-username/pretrain-stage2

# Stage 3
python push_datatrove_to_hub.py \
  --input-folder ./datatrove_output/stage3 \
  --repo-name your-username/pretrain-stage3
```

## Performance Tips

### Speed Up Upload

1. **Compress data** (already done with .jsonl.gz)
2. **Use fast internet** (fiber/cable > wifi)
3. **Upload during off-peak hours** (night/weekend)

### Reduce Memory Usage

1. **Process in batches**:
   ```bash
   # Process first 100k documents
   python push_datatrove_to_hub.py \
     --input-folder ./datatrove_output/stage1 \
     --repo-name username/dataset-part1 \
     --max-documents 100000
   ```

2. **Use sampling**:
   ```bash
   # Use 50% of data
   python push_datatrove_to_hub.py \
     --input-folder ./datatrove_output/stage1 \
     --repo-name username/dataset-half \
     --sample-ratio 0.5
   ```

## Monitoring Upload

Monitor the upload progress:

```bash
# In terminal 1: Run push script
python push_datatrove_to_hub.py --input-folder ./data --repo-name user/dataset

# In terminal 2: Monitor network usage
watch -n 1 'nethogs'  # Or use your system's network monitor

# Check upload on HuggingFace
# Go to https://huggingface.co/datasets/username/dataset-name
# Refresh to see upload progress
```

## Advanced: Programmatic Upload

You can also use the script's functions in your own code:

```python
from push_datatrove_to_hub import (
    load_documents_from_folder,
    create_dataset_from_documents,
    push_to_hub
)

# Load documents
documents = load_documents_from_folder(
    input_folder="./datatrove_output/stage1",
    max_documents=10000
)

# Create dataset
dataset = create_dataset_from_documents(
    documents=documents,
    train_test_split=0.95
)

# Custom processing here...

# Push to hub
push_to_hub(
    dataset=dataset,
    repo_name="username/custom-dataset",
    private=False
)
```

## Summary

### Quick Reference

```bash
# 1. Process with DataTrove
python smollm3_stage1_datatrove_pipeline.py --max-datasets 5

# 2. Test push
python push_datatrove_to_hub.py \
  --input-folder ./datatrove_output/stage1 \
  --repo-name user/test \
  --max-documents 1000

# 3. Full push
python push_datatrove_to_hub.py \
  --input-folder ./datatrove_output/stage1 \
  --repo-name user/smollm3-stage1-pretrain \
  --output-dir ./backup
```

### Checklist

- [ ] DataTrove pipeline completed successfully
- [ ] Verified JSONL.gz files exist
- [ ] Logged in to HuggingFace (`huggingface-cli login`)
- [ ] Tested with small sample first
- [ ] Chose meaningful repository name
- [ ] Decided on public vs private
- [ ] Kept local backup (optional but recommended)
- [ ] Added dataset card/README after push

---

**You're now ready to push your processed data to HuggingFace Hub! 🚀**

Start with a small test, verify the output, then push the full dataset.
