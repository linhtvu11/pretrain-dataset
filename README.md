# LLM Pretraining Dataset Collection and Preparation

A comprehensive toolkit for collecting, cleaning, filtering, and merging HuggingFace datasets suitable for Large Language Model (LLM) pretraining. This project focuses on high-quality text data related to code, technology, science, biology, and question-answering.

## Features

- **Multi-language Support**: Includes both English and Vietnamese datasets
- **Weighted Sampling**: Production-tested dataset weights from SmolLM3 (exact 3-stage training mix)
- **Bilingual Filtering**: Complete English and Vietnamese spam/junk detection (75 keep keywords, 54 exclude keywords)
- **Smart Filtering**: Excludes junk/spam content while preserving technical, scientific, and code-related text
- **Automatic Cleaning**: Removes HTML tags, normalizes Unicode, cleans URLs and whitespace
- **Deduplication**: Prevents duplicate content using hash-based matching
- **Flexible Configuration**: Easy-to-modify YAML configuration for dataset selection
- **HuggingFace Integration**: Direct push to HuggingFace Hub
- **Comprehensive Logging**: Detailed statistics and progress tracking
- **Tested**: 100% test pass rate (21/21 tests - English and Vietnamese)

## Dataset Collection

**✓ VERIFIED**: All datasets confirmed to exist on HuggingFace (November 2025)

**🎉 NEW**: Added 13 datasets from **SmolLM3 Collection** (including Vietnamese support!)
**🔥 NEW**: Added SmolLM3 production weights - exact dataset mix used to train SmolLM3 on 8T tokens!

### English Datasets (21 verified, +11 new from SmolLM3)

**From SmolLM3 Collection (11 new):**
1. **HuggingFaceFW/fineweb-edu** - 3.5B educational web content
2. **HuggingFaceFW/fineweb-2** (config: eng_Latn) - Multilingual web (1868 languages)
3. **epfml/FineWeb2-HQ** (config: eng_Latn) - Top 10% quality filtered (380M rows)
4. **HuggingFaceTB/smollm-corpus** (cosmopedia-v2) - 39M synthetic educational
5. **HuggingFaceTB/smollm-corpus** (fineweb-edu-dedup) - 190M deduplicated educational
6. **HuggingFaceTB/smollm-corpus** (python-edu) - 7.68M Python educational code
7. **bigcode/the-stack-v2** - 67.5TB code, 658 languages (v1: 6.4TB → v2: 67.5TB!)
8. **HuggingFaceTB/stack-edu** - 125B tokens educational code
9. **HuggingFaceTB/finemath** - 48.3M mathematical content
10. **HuggingFaceTB/issues-kaggle-notebooks** - 16M GitHub issues + Kaggle notebooks

**Existing Datasets (10):**
11. **HuggingFaceFW/fineweb** - 15T tokens high-quality web
12. **allenai/c4** (config: en) - 750GB clean web text
13. **Skylion007/openwebtext** - Reddit-sourced quality content
14. **wikimedia/wikipedia** (config: 20231101.en) - English Wikipedia
15. **bigcode/the-stack** - 6TB code (v1)
16. **codeparrot/github-code** - 115M code files from GitHub
17. **armanc/scientific_papers** (config: arxiv) - ArXiv and PubMed papers
18. **rajpurkar/squad** - Stanford Question Answering Dataset
19. **allenai/cosmos_qa** - 35.6K commonsense QA
20. **microsoft/wiki_qa** - Wikipedia-based QA

### Vietnamese Datasets (8 verified, +2 new from SmolLM3) 🇻🇳

**From SmolLM3 Collection (2 new):**
1. **HuggingFaceFW/fineweb-2** (config: vie_Latn) - 🎉 High-quality Vietnamese web data
2. **epfml/FineWeb2-HQ** (config: vie_Latn) - 🎉 4M top 10% quality Vietnamese web

**Existing Datasets (6):**
3. **vietgpt/binhvq_news_vi** - 19.4M Vietnamese articles, 4.78GB
4. **bkai-foundation-models/NewsCategory** - 596K categorized news
5. **wikimedia/wikipedia** (config: 20231101.vi) - Vietnamese Wikipedia
6. **opendatalab/WanJuan-Vietnamese** - 280GB+ multi-category corpus
7. **uitnlp/vietnamese_students_feedback** - 16K+ annotated sentences
8. **bkai-foundation-models/vi-alpaca** - 50K instruction-following examples

**See DATASETS.md, VERIFIED_DATASETS.md, and SMOLLM3_DATASETS.md for complete details.**

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd pretrain-dataset

# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace (required for pushing datasets)
huggingface-cli login
```

## Configuration

Edit `datasets_config.yaml` to customize:

- Dataset selection (enable/disable specific datasets)
- Text field names for each dataset
- Filtering criteria (length, keywords, patterns)
- Deduplication settings
- Output configuration

### Key Configuration Options

```yaml
filtering:
  min_length: 100          # Minimum text length in characters
  max_length: 1000000      # Maximum text length

  keep_keywords:           # Content with these keywords is prioritized (75 total)
    # English (32): code, programming, science, research, algorithm...
    - code
    - programming
    - science
    - research

    # Vietnamese (43): lập trình, khoa học, công nghệ, dữ liệu...
    - "lập trình"
    - "khoa học"
    - "công nghệ"
    - "dữ liệu"

  exclude_keywords:        # Content with these is filtered out (54 total)
    # English spam (24): subscribe now, buy now, click here...
    - "subscribe now"
    - "click here"
    - spam

    # Vietnamese spam (30): đăng ký ngay, mua ngay, lừa đảo...
    - "đăng ký ngay"
    - "mua ngay"
    - "lừa đảo"

  junk_patterns:           # Regex patterns for spam detection (9 patterns)
    - "!!!!!+"             # Excessive punctuation
    - "(?i)(mua|đăng\\s*ký).*ngay.*[!]{2,}"  # Vietnamese spam

  deduplication:
    enabled: true
    similarity_threshold: 0.85
```

**See FILTERING.md for complete bilingual filtering documentation.**

## Usage

### Test Cleaning Functionality

```bash
python test_cleaning.py
```

This runs test cases to verify the cleaning and filtering logic works correctly.

### 🔥 Use SmolLM3 Production Weights (Recommended)

Replicate SmolLM3's exact training mix with production-tested dataset weights:

```bash
# Stage 1: Foundation (0-8T tokens) - Balanced mix
python merge_datasets.py \
  --config smollm3_weighted_config.yaml \
  --use-weights \
  --output-dir ./smollm3_stage1_dataset

# Stage 2: Educational Focus (8T-9T tokens) - Enhanced math & code
python merge_datasets.py \
  --config smollm3_stage2_config.yaml \
  --use-weights \
  --output-dir ./smollm3_stage2_dataset

# Stage 3: Reasoning & Advanced Code (9T-11T tokens) - Expert level
python merge_datasets.py \
  --config smollm3_stage3_config.yaml \
  --use-weights \
  --output-dir ./smollm3_stage3_dataset
```

**Stage 1 Dataset weights**:
- 37% dclm (general web)
- 33.3% fineweb-edu (educational content)
- 11.9% code (Python 2.5%, C++ 1.8%, Java 1.3%, etc.)
- 2.7% math (finemath 1.7%, infiwebmath 1.0%)
- 9% multilingual (including 0.325% Vietnamese)

**Stage 2 & 3**: Increased math (up to 15.6%) and code (up to 20.2%) with reasoning datasets.

**See `SMOLLM3_TRAINING_STAGES.md` for complete three-stage training guide.**
**See `SMOLLM3_WEIGHTS.md` and `DATASET_WEIGHTS_SUMMARY.md` for weight details.**
**See `WEIGHTED_SAMPLING_USAGE.md` for detailed weighted sampling documentation.**

### Merge Datasets (Quick Test)

For testing with limited samples:

```bash
python merge_datasets.py \
  --max-samples 1000 \
  --output-dir ./test_output
```

### Merge All Datasets

Full dataset processing:

```bash
python merge_datasets.py \
  --output-dir ./merged_dataset
```

### Merge English Only

```bash
python merge_datasets.py \
  --english-only \
  --output-dir ./english_dataset
```

### Merge Vietnamese Only

```bash
python merge_datasets.py \
  --vietnamese-only \
  --output-dir ./vietnamese_dataset
```

### Push to HuggingFace Hub

```bash
python merge_datasets.py \
  --push-to-hub \
  --hub-repo your-username/merged-llm-pretrain \
  --output-dir ./merged_dataset
```

For private datasets:

```bash
python merge_datasets.py \
  --push-to-hub \
  --hub-repo your-username/merged-llm-pretrain \
  --private
```

### Advanced Usage

Combine multiple options:

```bash
python merge_datasets.py \
  --config custom_config.yaml \
  --max-samples 10000 \
  --english-only \
  --push-to-hub \
  --hub-repo your-username/english-pretrain \
  --private
```

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────┐
│                  datasets_config.yaml                    │
│  (Dataset sources and filtering configuration)          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                 clean_and_filter.py                      │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐ │
│  │ TextCleaner  │  │  TextFilter   │  │ Deduplicator │ │
│  │              │  │               │  │              │ │
│  │ - Remove HTML│  │ - Length check│  │ - Hash-based │ │
│  │ - Clean URLs │  │ - Keywords    │  │ - Fuzzy match│ │
│  │ - Normalize  │  │ - Patterns    │  │              │ │
│  └──────────────┘  └───────────────┘  └──────────────┘ │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                 merge_datasets.py                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │           DatasetMerger                          │   │
│  │                                                  │   │
│  │  1. Load datasets from HuggingFace              │   │
│  │  2. Apply cleaning and filtering                │   │
│  │  3. Merge into single dataset                   │   │
│  │  4. Create train/test split                     │   │
│  │  5. Save locally or push to Hub                 │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Text Processing Pipeline

```
Raw Text
  │
  ├─> Unicode Normalization
  │
  ├─> HTML Tag Removal
  │
  ├─> URL Cleaning
  │
  ├─> Whitespace Normalization
  │
  ├─> Length Filtering
  │
  ├─> Junk Pattern Detection
  │
  ├─> Keyword Filtering
  │
  ├─> Relevance Scoring
  │
  ├─> Code Detection
  │
  └─> Deduplication
      │
      ▼
  Cleaned Text
```

## Filtering Logic

### Content is KEPT if:

1. Length is between min_length and max_length
2. No junk patterns detected (excessive punctuation, spam indicators)
3. No exclude keywords found
4. **Either**:
   - Contains relevant keywords (code, science, tech, etc.)
   - **OR** Contains code indicators (function definitions, imports, etc.)
5. Not a duplicate of previously seen content

### Code Detection

The system automatically detects code-related content even without explicit keywords by looking for:

- Function definitions (`def`, `function`, `public`)
- Class definitions
- Import statements
- Variable declarations
- Markdown code blocks
- Common programming patterns

## Output Format

The merged dataset is saved in HuggingFace Dataset format with:

- **train** split (95% of data)
- **test** split (5% of data)
- Single `text` field containing cleaned content

## Statistics and Monitoring

The scripts provide detailed statistics:

- Total texts processed
- Number kept vs. filtered
- Breakdown of filtering reasons:
  - Empty texts
  - Length violations
  - Junk patterns
  - Exclude keywords
  - Low relevance
  - Duplicates
- Special tracking for code content

Example output:

```
=== Processing Statistics ===
Total texts processed: 10000
Kept: 7543 (75.43%)
  - With code: 2341
Filtered out: 2457
  - Empty: 23
  - Empty after cleaning: 45
  - Length: 156
  - Junk patterns: 892
  - Exclude keywords: 234
  - Low relevance: 1056
  - Duplicates: 51
========================================
```

## Customization

### Adding New Datasets

Edit `datasets_config.yaml`:

```yaml
english_datasets:
  - name: "your-org/your-dataset"
    description: "Description of dataset"
    text_field: "text"  # Field containing the text
    split: "train"
    streaming: true
    categories: ["code", "tech"]
```

### Modifying Filtering Rules

Adjust in `datasets_config.yaml`:

```yaml
filtering:
  keep_keywords:
    - your
    - custom
    - keywords

  exclude_keywords:
    - unwanted
    - content

  junk_patterns:
    - "your_regex_pattern"
```

### Custom Processing Logic

Modify `clean_and_filter.py`:

- `TextCleaner`: Add new cleaning methods
- `TextFilter`: Add custom filtering logic
- `TextDeduplicator`: Modify deduplication strategy

## Performance Considerations

- **Streaming Mode**: Used for large datasets to avoid loading all data into memory
- **Batch Processing**: Process in chunks for better memory management
- **Deduplication Limits**: Fuzzy deduplication limited to 10K texts for performance
- **Max Samples**: Use `--max-samples` for testing before full processing

## Troubleshooting

### Issue: Dataset not loading

```bash
# Check if dataset exists
huggingface-cli repo info dataset_name

# Try without streaming
# Edit config: streaming: false
```

### Issue: Out of memory

```bash
# Use max-samples for testing
python merge_datasets.py --max-samples 1000

# Or enable streaming in config
streaming: true
```

### Issue: Permission denied on push

```bash
# Login again
huggingface-cli login

# Check token has write permissions
huggingface-cli whoami
```

### Issue: Text field not found

Check the dataset structure:

```python
from datasets import load_dataset
ds = load_dataset("dataset_name", split="train", streaming=True)
print(next(iter(ds)))  # See available fields
```

Update `text_field` in config accordingly.

## Best Practices

1. **Test First**: Always run with `--max-samples` first
2. **Check Output**: Verify a few samples from the merged dataset
3. **Monitor Stats**: Review filtering statistics to tune parameters
4. **Backup Config**: Keep different configs for different use cases
5. **Version Control**: Commit config changes before major runs
6. **Resource Planning**: Large datasets may take hours to process

## License

This project is provided as-is for research and educational purposes. Individual datasets may have their own licenses - please check each dataset's license before use.

## Contributing

Contributions welcome! Please:

1. Add new datasets to the config with proper attribution
2. Improve filtering logic with clear comments
3. Add tests for new features
4. Update documentation

## Citation

If you use this toolkit, please cite:

```bibtex
@software{llm_pretrain_dataset,
  title = {LLM Pretraining Dataset Collection and Preparation},
  author = {Your Name},
  year = {2025},
  url = {repository-url}
}
```

## Acknowledgments

- HuggingFace for the datasets library and hosting
- Dataset creators for providing open-access data
- BigCode, Allen AI, Microsoft, and other organizations for quality datasets

## Contact

For questions or issues, please open an issue on the repository.
