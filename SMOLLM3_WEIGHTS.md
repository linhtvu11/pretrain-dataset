# SmolLM3 Dataset Weights and Training Mix

**Source**: https://huggingface.co/datasets/HuggingFaceTB/smollm3-configs/resolve/main/stage1_8T.yaml

This document explains the exact dataset weights used by HuggingFace to train SmolLM3 on 8 trillion tokens (Stage 1).

## Overview

SmolLM3 uses a carefully balanced mix of datasets with specific weights. The configuration in `smollm3_weighted_config.yaml` implements these exact weights for reproducible training.

## Dataset Weight Distribution

### Category Breakdown

| Category | Total Weight | Percentage |
|----------|-------------|------------|
| **General Web** | 73.2% | ~5.86T tokens |
| **Code** | 11.9% | ~0.95T tokens |
| **Math** | 2.7% | ~0.22T tokens |
| **Other** | 12.2% | ~0.98T tokens (stage 2/3) |
| **TOTAL** | 100% | 8T tokens |

### Detailed Breakdown

#### 1. General Web Content (73.2%)

**Primary English Datasets (70.3%)**:
- **dclm** (mlfoundations/dclm-baseline-1.0): 37.0% - Largest component
- **fineweb-edu** (HuggingFaceFW/fineweb-edu): 33.3% - Educational web content
- **pes2o** (allenai/peS2o): 2.0% - Semantic Scholar research papers

**Multilingual FineWeb-2 (9.0%)**:
- French (fra_Latn): 1.6%
- German (deu_Latn): 2.2%
- Spanish (spa_Latn): 2.0%
- Italian (ita_Latn): 1.05%
- Portuguese (por_Latn): 1.0%
- Chinese (cmn_Hans): 1.0%
- Russian (rus_Cyrl): 1.0%
- Persian (fas_Arab): 0.3%
- Japanese (jpn_Jpan): 0.325%
- Korean (kor_Hang): 0.325%
- Hindi (hin_Deva): 0.325%
- Thai (tha_Thai): 0.325%
- **Vietnamese (vie_Latn): 0.325%** 🇻🇳
- Greek (ell_Grek): 0.225%

**Other Web Sources (0.5%)**:
- Wikipedia: 0.1%
- StackExchange: 0.4%

#### 2. Code Content (11.9%)

**Stack-Edu by Language (9.61%)**:
- Python: 2.5% (largest code component)
- C++: 1.8%
- Java: 1.3%
- JavaScript: 1.3%
- C: 0.7%
- C#: 0.6%
- PHP: 0.6%
- HTML: 0.6%
- Markdown: 0.5%
- SQL: 0.4%
- TypeScript: 0.3%
- Swift: 0.1%
- Ruby: 0.08%
- Rust: 0.08%
- Shell: 0.07%
- Go: 0.05%

**GitHub & Kaggle (2.29%)**:
- Pull Requests: 0.6%
- Jupyter Scripts: 0.55%
- GitHub Issues: 0.32%
- Kaggle Notebooks: 0.05%

#### 3. Math Content (2.7%)

- **finemath** (HuggingFaceTB/finemath): 1.7%
- **infiwebmath**: 1.0%

## Vietnamese Representation

Vietnamese content has **0.325%** weight in the training mix, equal to:
- ~26 billion tokens in 8T training
- Same weight as Japanese, Korean, Hindi, and Thai
- Part of the multilingual FineWeb-2 dataset (vie_Latn config)

## Usage

### Option 1: Using the Weighted Config

```bash
# Use the SmolLM3 weighted configuration
python merge_datasets.py \
  --config smollm3_weighted_config.yaml \
  --output-dir ./smollm3_dataset \
  --use-weights
```

### Option 2: Custom Training Mix

You can adjust weights in `smollm3_weighted_config.yaml`:

```yaml
datasets:
  - name: "HuggingFaceFW/fineweb-edu"
    weight: 0.333  # Adjust this value
    # ... other config

  - name: "HuggingFaceFW/fineweb-2"
    config: "vie_Latn"
    weight: 0.00325  # Vietnamese weight
```

### Option 3: Increase Vietnamese Content

To increase Vietnamese representation:

```yaml
  - name: "HuggingFaceFW/fineweb-2"
    config: "vie_Latn"
    weight: 0.01  # Increase from 0.00325 to 1%
```

**Note**: Remember to normalize all weights to sum to 1.0!

## Dataset Mapping

### HuggingFace Dataset Paths

| SmolLM3 Name | HuggingFace Path | Config |
|--------------|------------------|--------|
| dclm | mlfoundations/dclm-baseline-1.0 | default |
| fineweb-edu | HuggingFaceFW/fineweb-edu | default |
| pes2o | allenai/peS2o | default |
| fw2-vie | HuggingFaceFW/fineweb-2 | vie_Latn |
| fw2-fra | HuggingFaceFW/fineweb-2 | fra_Latn |
| fw2-spa | HuggingFaceFW/fineweb-2 | spa_Latn |
| stack-edu-Python | HuggingFaceTB/stack-edu | python |
| finemath | HuggingFaceTB/finemath | default |
| wiki | wikimedia/wikipedia | 20231101.en |
| stackexchange | HuggingFaceTB/stackexchange_2025_md | default |

### Config Parameters

Many datasets require specific config parameters:

```python
# FineWeb-2 Vietnamese
load_dataset("HuggingFaceFW/fineweb-2", "vie_Latn")

# Stack-Edu Python
load_dataset("HuggingFaceTB/stack-edu", "python")

# Wikipedia English
load_dataset("wikimedia/wikipedia", "20231101.en")
```

## Sampling Strategy

When using weights, implement **weighted random sampling**:

1. **Probability-based sampling**: Sample from each dataset with probability proportional to its weight
2. **Token-based mixing**: Mix datasets to achieve target token counts
3. **Epoch-based**: Cycle through datasets multiple times based on weights

### Example Calculation

For 8T tokens total:
- dclm (37%): Sample 2.96T tokens
- fineweb-edu (33.3%): Sample 2.664T tokens
- Vietnamese (0.325%): Sample 26B tokens

## Training Stages

SmolLM3 training has 3 stages:

### Stage 1 (8T tokens) - This Config
- Focus: General knowledge, code, math
- Duration: ~87.8% of total training
- Mix: As specified above

### Stage 2 - Educational Code
- Focus: High-quality educational code (stack-edu)
- Not in this config (separate training phase)

### Stage 3 - Reasoning
- Focus: Reasoning datasets (OpenMathReasoning, OpenCodeReasoning, etc.)
- Not in this config (separate training phase)

## Recommendations

### For Balanced Multilingual Model
Use the exact SmolLM3 weights - proven to work well.

### For Vietnamese-Focused Model
Increase weights:
```yaml
# Increase Vietnamese from 0.325% to 5-10%
- name: "HuggingFaceFW/fineweb-2"
  config: "vie_Latn"
  weight: 0.05  # 5%

# Add Vietnamese-specific datasets
- name: "vietgpt/binhvq_news_vi"
  weight: 0.05  # 5%
```

### For Code-Focused Model
Increase code weights:
```yaml
# Increase Python from 2.5% to 10%
- name: "HuggingFaceTB/stack-edu"
  config: "python"
  weight: 0.10  # 10%
```

## Weight Normalization

Always ensure weights sum to 1.0:

```python
import yaml

with open('smollm3_weighted_config.yaml') as f:
    config = yaml.safe_load(f)

total_weight = sum(d['weight'] for d in config['datasets'])
print(f"Total weight: {total_weight}")  # Should be ~1.0

# Normalize if needed
for dataset in config['datasets']:
    dataset['weight'] /= total_weight
```

## Implementation in merge_datasets.py

To use weights in the merger:

```python
# Weighted sampling
from datasets import load_dataset, concatenate_datasets
import numpy as np

def sample_with_weights(datasets_config):
    """Sample from datasets according to weights"""
    datasets = []
    weights = []

    for ds_config in datasets_config:
        ds = load_dataset(
            ds_config['name'],
            ds_config.get('config'),
            split='train',
            streaming=True
        )
        datasets.append(ds)
        weights.append(ds_config['weight'])

    # Normalize weights
    weights = np.array(weights) / sum(weights)

    # Sample from datasets
    # Implementation depends on your sampling strategy
    return interleave_datasets(datasets, probabilities=weights)
```

## Verification

To verify your implementation matches SmolLM3:

1. **Check weight sum**: Should equal 1.0
2. **Check Vietnamese weight**: Should be 0.00325 (0.325%)
3. **Check top datasets**: dclm (37%), fineweb-edu (33.3%)
4. **Check code total**: Should be ~11.9%
5. **Check math total**: Should be ~2.7%

## References

- SmolLM3 Config: https://huggingface.co/datasets/HuggingFaceTB/smollm3-configs/resolve/main/stage1_8T.yaml
- SmolLM3 Blog Post: https://huggingface.co/blog/smollm3
- Training Details: https://huggingface.co/collections/HuggingFaceTB/smollm3-pretraining-datasets

## Notes

1. **DCLM dataset**: May need verification - check if mlfoundations/dclm-baseline-1.0 is correct path
2. **InfiWebMath**: Listed as nvidia/OpenMathInstruct-1 - verify this mapping
3. **Subsets**: Some datasets like issues-kaggle-notebooks have multiple subsets
4. **Configs**: Language-specific configs use ISO codes + script (e.g., vie_Latn, jpn_Jpan)

## License Compatibility

Ensure all datasets have compatible licenses for your use case:
- Most are Apache 2.0, MIT, or CC-BY
- Check each dataset's license on HuggingFace before commercial use
- Vietnamese fineweb-2 (vie_Latn) inherits FineWeb-2's license
