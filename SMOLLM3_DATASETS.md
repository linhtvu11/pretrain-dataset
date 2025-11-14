# SmolLM3 Collection Datasets

**Source**: https://huggingface.co/collections/HuggingFaceTB/smollm3-pretraining-datasets

This document lists all datasets from the HuggingFaceTB SmolLM3 pretraining collection that have been added to our configuration.

## Overview

The SmolLM3 collection contains **high-quality, curated datasets** used for training the SmolLM3 model. These datasets are:
- ✓ Production-tested by HuggingFace team
- ✓ Well-documented and maintained
- ✓ Optimized for LLM pretraining
- ✓ Include educational, code, math, and general web content
- ✓ **Include multilingual support (Vietnamese!)**

## New Datasets Added

### English Datasets (11 new datasets)

#### Web & General Content (5 datasets)

1. **HuggingFaceFW/fineweb-edu** ✓
   - Size: 3.5B rows
   - Description: Educational web content filtered from FineWeb
   - Text field: `text`
   - Categories: general, education
   - Usage in SmolLM3: Core educational content

2. **HuggingFaceFW/fineweb-2** (English: eng_Latn) ✓
   - Size: Massive (1868 languages supported)
   - Description: Multilingual web dataset, English subset
   - Text field: `text`
   - Config: `eng_Latn`
   - Categories: general, tech
   - Usage in SmolLM3: Stage 1 (85% of mix)

3. **epfml/FineWeb2-HQ** (English: eng_Latn) ✓
   - Size: 380M rows (top 10% of FineWeb2)
   - Description: High-quality filtered FineWeb2 using ML classifier
   - Text field: `text`
   - Config: `eng_Latn`
   - Categories: general, tech
   - Quality: Top 10% filtered by deep learning model

4. **HuggingFaceTB/smollm-corpus** (cosmopedia-v2) ✓
   - Size: 39M rows
   - Description: Synthetic educational content
   - Text field: `text`
   - Config: `cosmopedia-v2`
   - Categories: general, education, synthetic
   - Special: Synthetically generated educational content

5. **HuggingFaceTB/smollm-corpus** (fineweb-edu-dedup) ✓
   - Size: 190M rows
   - Description: Deduplicated educational web content
   - Text field: `text`
   - Config: `fineweb-edu-dedup`
   - Categories: general, education

#### Code Datasets (4 datasets)

6. **bigcode/the-stack-v2** ✓
   - Size: 67.5TB (3.28B files, 658 languages)
   - Description: Massive upgrade from v1 (6.4TB → 67.5TB)
   - Text field: `text` (uses blob_id for download)
   - Categories: code
   - Note: Requires special handling for blob_id download
   - Usage in SmolLM3: Stage 1 (12% of mix)

7. **HuggingFaceTB/stack-edu** ✓
   - Size: 125B tokens, 167M rows, 17.5GB
   - Description: Educational code filtered from Stack v2
   - Text field: `text`
   - Categories: code, education
   - Languages: 15 programming languages
   - Usage in SmolLM3: Stage 2 educational focus

8. **HuggingFaceTB/smollm-corpus** (python-edu) ✓
   - Size: 7.68M rows
   - Description: Python educational code
   - Text field: `text`
   - Config: `python-edu`
   - Categories: code, education

9. **HuggingFaceTB/issues-kaggle-notebooks** ✓
   - Size: 16.1M rows (15.5M issues + 580k notebooks)
   - Description: GitHub issues and Kaggle notebooks
   - Text field: `text`
   - Categories: code, documentation
   - Contains: Discussion, Q&A, code notebooks

#### Math & Science (2 datasets)

10. **HuggingFaceTB/finemath** ✓
    - Size: 48.3M rows
    - Description: Mathematical educational content
    - Text field: `text`
    - Categories: math, education
    - Usage in SmolLM3: Stage 1 (3% of mix)

11. **Retained existing**: armanc/scientific_papers
    - For ArXiv and PubMed papers

### Vietnamese Datasets (2 new datasets) 🎉

12. **HuggingFaceFW/fineweb-2** (Vietnamese: vie_Latn) ✓
    - Size: Large (part of 1868 language dataset)
    - Description: Multilingual web dataset, Vietnamese subset
    - Text field: `text`
    - Config: `vie_Latn` (Vietnamese in Latin script)
    - Categories: general, tech
    - **First high-quality web dataset for Vietnamese from SmolLM3!**

13. **epfml/FineWeb2-HQ** (Vietnamese: vie_Latn) ✓
    - Size: 4M rows (top 10% quality)
    - Description: High-quality filtered Vietnamese web content
    - Text field: `text`
    - Config: `vie_Latn`
    - Categories: general, tech
    - Quality: ML-filtered top 10% quality
    - **High-quality Vietnamese web content!**

## Dataset Statistics Summary

### Total Counts
- **Total new datasets**: 13
- **English datasets**: 11 new (now 21 total)
- **Vietnamese datasets**: 2 new (now 8 total)
- **From SmolLM3 collection**: 13 datasets

### By Category
**Code**: 7 datasets (4 new from SmolLM3)
- the-stack-v2, stack-edu, smollm-corpus (python-edu), issues-kaggle-notebooks
- the-stack (v1), github-code, stack-edu (existing)

**Education**: 8 datasets (6 new from SmolLM3)
- fineweb-edu, smollm-corpus (cosmopedia-v2, fineweb-edu-dedup, python-edu)
- stack-edu, finemath

**Math**: 1 dataset (1 new from SmolLM3)
- finemath

**Web/General**: 11 datasets (5 new from SmolLM3)
- fineweb-2 (eng_Latn, vie_Latn), fineweb2-hq (eng_Latn, vie_Latn)
- fineweb-edu, smollm-corpus (cosmopedia-v2, fineweb-edu-dedup)

**Science**: 1 dataset
- scientific_papers (existing)

**QA**: 5 datasets
- squad, cosmos_qa, wiki_qa (existing English)
- vietnamese_students_feedback, vi-alpaca (existing Vietnamese)

## SmolLM3 Training Mix

The SmolLM3 model was trained using a specific mix:

**Stage 1 (85% + 12% + 3%)**:
- 85%: fineweb-2 (web content)
- 12%: the-stack-v2 (code)
- 3%: finemath (math)

**Stage 2**:
- stack-edu (educational code)

**Stage 3 (Reasoning)**:
- Additional reasoning datasets (not all included in our config)

## Key Features of SmolLM3 Datasets

1. **High Quality**: All datasets filtered and curated
2. **Educational Focus**: Many datasets specifically filtered for educational content
3. **Multilingual**: fineweb-2 supports 1868 languages including Vietnamese!
4. **Large Scale**: Some of the largest open datasets (67.5TB for stack-v2)
5. **Diverse**: Code, math, science, general web, QA
6. **Production-Tested**: Used in actual LLM training by HuggingFace

## Usage Examples

### Load FineWeb-Edu
```python
from datasets import load_dataset
ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
```

### Load Vietnamese FineWeb2-HQ
```python
ds = load_dataset("epfml/FineWeb2-HQ", "vie_Latn", split="train", streaming=True)
```

### Load SmolLM Corpus (Python Educational Code)
```python
ds = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split="train", streaming=True)
```

### Load FineMath
```python
ds = load_dataset("HuggingFaceTB/finemath", split="train", streaming=True)
```

### Load Stack-Edu (Educational Code)
```python
ds = load_dataset("HuggingFaceTB/stack-edu", split="train", streaming=True)
```

## Special Handling

### The Stack v2
- Uses `blob_id` instead of direct text
- Requires downloading content from Software Heritage S3
- Our config attempts to use `text` field if available
- May require custom loader

### SmolLM Corpus
- Has 3 subsets: `cosmopedia-v2`, `fineweb-edu-dedup`, `python-edu`
- Must specify config when loading
- All use `text` field

### FineWeb-2 & FineWeb2-HQ
- Must specify language config (e.g., `eng_Latn`, `vie_Latn`)
- Supports 1868 language-script pairs
- Vietnamese: `vie_Latn`

## Configuration Updates

All SmolLM3 datasets have been added to `datasets_config.yaml` with:
- ✓ Correct dataset paths
- ✓ Proper text field names
- ✓ Required config parameters
- ✓ Categories and metadata
- ✓ `source: "SmolLM3"` tag for identification

## Benefits for Our Toolkit

1. **Vietnamese Support**: Now we have high-quality web data for Vietnamese!
2. **Educational Focus**: Many educational datasets perfect for quality pretraining
3. **Code Quality**: Educational code datasets (stack-edu, python-edu)
4. **Math Content**: Specialized math dataset (finemath)
5. **Production Quality**: Proven datasets used by HuggingFace
6. **Large Scale**: Access to massive datasets (67.5TB stack-v2)

## Next Steps

Users can now:
1. ✓ Use high-quality Vietnamese web data (fineweb-2, fineweb2-hq)
2. ✓ Access educational content (fineweb-edu, stack-edu, python-edu)
3. ✓ Include math-focused content (finemath)
4. ✓ Use the largest code dataset (stack-v2)
5. ✓ Customize mix based on SmolLM3 proven recipes

## References

- SmolLM3 Collection: https://huggingface.co/collections/HuggingFaceTB/smollm3-pretraining-datasets
- FineWeb-2: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
- FineWeb2-HQ: https://huggingface.co/datasets/epfml/FineWeb2-HQ
- Stack v2: https://huggingface.co/datasets/bigcode/the-stack-v2
- Stack-Edu: https://huggingface.co/datasets/HuggingFaceTB/stack-edu
- FineMath: https://huggingface.co/datasets/HuggingFaceTB/finemath
- SmolLM Corpus: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus
