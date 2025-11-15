# DataTrove Implementation Summary

## What Was Implemented

I've added a **production-grade data processing pipeline using HuggingFace's DataTrove** - the exact same tool used to create SmolLM3's training data.

## Why DataTrove?

DataTrove is **not just another data processing library** - it's the production system behind:

- ✅ **SmolLM3** - 11.2T tokens processed across 3 stages
- ✅ **FineWeb** - One of the highest quality web datasets ever created
- ✅ **The Stack v2** - 67.5TB of code data

### Key Advantages Over Custom Scripts

| Aspect | Custom Scripts | DataTrove |
|--------|---------------|-----------|
| **Scale** | Limited by RAM | **Unlimited (streaming)** |
| **Quality Filters** | Basic keyword matching | **Production-tested (Gopher, C4, FineWeb)** |
| **Deduplication** | Simple hash-based | **Advanced (MinHash, sentence-level, exact substring)** |
| **Parallelization** | Manual threading | **Automatic multi-core + cluster support** |
| **Resumability** | Manual checkpoints | **Automatic task completion tracking** |
| **Production Use** | Experimental | **Used by HuggingFace production** |

## Files Created

### 1. **smollm3_stage1_datatrove_pipeline.py** (Main Pipeline)

Complete production pipeline for SmolLM3 Stage 1 processing.

**Features**:
- Processes all 41 datasets from `smollm3_weighted_config.yaml`
- Weighted sampling based on exact SmolLM3 proportions
- Production quality filters:
  - **LanguageFilter**: English + Vietnamese detection
  - **GopherRepetitionFilter**: Removes repetitive content
  - **GopherQualityFilter**: 8 quality metrics from DeepMind's Gopher
  - **Custom spam filter**: Based on our bilingual keyword lists
- Parallel processing with automatic checkpointing
- Optional deduplication (sentence-level or MinHash)

**Usage**:
```bash
# Test with 5 datasets (~10-30 minutes)
python smollm3_stage1_datatrove_pipeline.py --max-datasets 5

# Full pipeline with all 41 datasets (~4-48 hours depending on hardware)
python smollm3_stage1_datatrove_pipeline.py \
  --config smollm3_weighted_config.yaml \
  --output-folder ./datatrove_output/stage1 \
  --tasks 10 \
  --workers 4
```

### 2. **datatrove_simple_example.py** (Learning Examples)

Simple examples to understand DataTrove before running the full pipeline.

**Three examples included**:

1. **Simple**: English dataset (Skylion007/openwebtext)
   ```bash
   python datatrove_simple_example.py simple
   ```

2. **Vietnamese**: Vietnamese news dataset
   ```bash
   python datatrove_simple_example.py vietnamese
   ```

3. **Multilingual**: FineWeb-2 with multiple languages
   ```bash
   python datatrove_simple_example.py multilingual
   ```

Each example takes ~2-5 minutes and helps you understand the pipeline structure.

### 3. **requirements_datatrove.txt** (Dependencies)

All DataTrove dependencies in one file.

**Installation**:
```bash
pip install -r requirements_datatrove.txt
```

**Key packages**:
- `datatrove[all]` - Core library with all features
- `trafilatura` - HTML text extraction
- `fasttext-wheel` - Language detection
- `simhash` - Deduplication
- `pyarrow` - Efficient data handling

### 4. **DATATROVE_GUIDE.md** (Complete Documentation)

Comprehensive 500+ line guide covering everything you need to know.

**Sections**:
- What is DataTrove and why use it
- Installation and verification
- Quick start examples
- SmolLM3 Stage 1 pipeline explained
- Understanding quality filters
- Advanced features (deduplication, parallelization)
- Production deployment guide
- Resource requirements and optimization
- Troubleshooting
- Comparison with custom scripts

### 5. **README.md** (Updated)

Added DataTrove section at the top with quick start instructions and comparison table.

## Quality Filters Explained

### 1. LanguageFilter
```python
LanguageFilter(
    languages=["en", "vi"],  # English and Vietnamese
    language_threshold=0.65  # 65% confidence required
)
```
- Detects document language using fastText
- Removes documents that aren't clearly English or Vietnamese
- Configurable confidence threshold

### 2. GopherRepetitionFilter
```python
GopherRepetitionFilter(
    dup_line_frac=0.3,      # Max 30% duplicate lines
    dup_para_frac=0.3,       # Max 30% duplicate paragraphs
    top_n_grams=(2, 3, 4),   # Check 2,3,4-grams
    dup_n_grams=(0.25, 0.25, 0.25),  # Max 25% duplicate n-grams
)
```
- Removes documents with excessive repetition
- Catches common low-quality patterns (spam, boilerplate)
- Based on DeepMind's Gopher paper

### 3. GopherQualityFilter
```python
GopherQualityFilter(
    min_doc_words=20,        # At least 20 words
    max_doc_words=200000,    # Max 200k words
    min_avg_word_length=3,   # Average word at least 3 chars
    max_avg_word_length=10,  # Average word max 10 chars
    max_symbol_word_ratio=0.1,  # Max 10% symbols
    max_bullet_lines_ratio=0.9,  # Max 90% bullet points
    max_ellipsis_lines_ratio=0.3,  # Max 30% lines with "..."
    max_non_alpha_words_ratio=0.8,  # Max 80% non-alphabetic
)
```
- 8 different quality metrics
- Removes low-quality documents
- Production-tested on billions of documents

### 4. Custom Spam Filter
```python
LambdaFilter(
    lambda doc: not any(kw in doc.text.lower() for kw in spam_keywords)
)
```
- Uses our bilingual spam keyword list (55 keywords)
- English: "subscribe now", "buy now", "click here", etc.
- Vietnamese: "đăng ký ngay", "mua ngay", "lừa đảo", etc.

## Output Format

Processed data is saved in compressed JSONL format:

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
    ├── vie_Latn/  # Vietnamese
    │   └── 0.jsonl.gz
    └── eng_Latn/  # English
        └── 0.jsonl.gz
```

Each JSONL file contains:
```json
{
  "text": "The actual document text...",
  "id": "unique-document-id",
  "metadata": {
    "source": "HuggingFaceFW/fineweb-edu",
    "config": "default",
    "weight": 0.333
  }
}
```

## How It Works

### Step-by-Step Process

1. **Load Configuration**
   - Reads `smollm3_weighted_config.yaml`
   - Gets 41 datasets with weights

2. **For Each Dataset**:
   ```
   HuggingFaceReader → Sampler → Language Filter →
   Repetition Filter → Quality Filter → Spam Filter →
   JSONL Writer
   ```

3. **Parallel Processing**:
   - Multiple tasks run in parallel
   - Each task has multiple workers
   - Automatic load balancing

4. **Checkpointing**:
   - Tracks completed tasks
   - Resume from last checkpoint on crash
   - No redundant processing

5. **Output**:
   - Compressed JSONL files (gzip)
   - Organized by dataset and config
   - Ready for LLM training

## Usage Examples

### Quick Test (Recommended First Step)

```bash
# Install dependencies
pip install -r requirements_datatrove.txt

# Test with simple example (2-5 minutes)
python datatrove_simple_example.py simple

# Check output
ls -lh test_output/
```

### Small-Scale Test (5 datasets)

```bash
# Test pipeline with 5 datasets (~10-30 minutes)
python smollm3_stage1_datatrove_pipeline.py \
  --max-datasets 5 \
  --tasks 2 \
  --workers 2 \
  --output-folder ./test_datatrove

# Check output size
du -sh test_datatrove/
```

### Full SmolLM3 Stage 1 (Production)

```bash
# Process all 41 datasets (~4-48 hours)
python smollm3_stage1_datatrove_pipeline.py \
  --config smollm3_weighted_config.yaml \
  --output-folder ./datatrove_output/stage1 \
  --tasks 10 \
  --workers 4 \
  --logging-dir ./datatrove_logs

# Monitor progress
tail -f datatrove_logs/main.log

# Check output
du -sh datatrove_output/stage1/
```

### With Deduplication

```bash
# Add deduplication step
python smollm3_stage1_datatrove_pipeline.py \
  --config smollm3_weighted_config.yaml \
  --output-folder ./datatrove_output/stage1 \
  --tasks 10 \
  --workers 4 \
  --run-dedup
```

## Resource Requirements

### For Quick Testing (5 datasets)
- **CPU**: 4 cores
- **RAM**: 8 GB
- **Disk**: 50 GB
- **Time**: ~10-30 minutes

### For Full Pipeline (41 datasets)

**Minimal**:
- CPU: 8 cores
- RAM: 16 GB
- Disk: 200 GB
- Time: ~24-48 hours

**Recommended**:
- CPU: 16 cores
- RAM: 32 GB
- Disk: 500 GB (SSD preferred)
- Time: ~8-12 hours

**High-Performance**:
- CPU: 32+ cores
- RAM: 64 GB+
- Disk: 1 TB SSD
- Time: ~4-6 hours

## Next Steps

### 1. Start with Examples

```bash
# Run all three simple examples
python datatrove_simple_example.py simple
python datatrove_simple_example.py vietnamese
python datatrove_simple_example.py multilingual
```

### 2. Test with Small Dataset

```bash
# Test with 5 datasets
python smollm3_stage1_datatrove_pipeline.py --max-datasets 5
```

### 3. Run Full Pipeline

```bash
# Full SmolLM3 Stage 1
python smollm3_stage1_datatrove_pipeline.py
```

### 4. Process Stage 2 and Stage 3

Modify the script to use:
- `smollm3_stage2_config.yaml` for Stage 2 (8T-9T tokens)
- `smollm3_stage3_config.yaml` for Stage 3 (9T-11T tokens)

### 5. Prepare for Training

Convert JSONL to your training framework's format (Parquet, TFRecord, etc.)

## Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce workers
```bash
--workers 1  # Use only 1 worker per task
```

### Issue: Too Slow
**Solutions**:
1. Increase parallelization: `--tasks 20 --workers 4`
2. Use SSD storage
3. Reduce sample size: `--max-datasets 5`

### Issue: Dataset Not Loading
**Check**:
1. Internet connection
2. HuggingFace Hub access
3. Dataset name is correct

### Issue: Pipeline Crashes
**Solutions**:
1. Check logs: `tail -f datatrove_logs/main.log`
2. Restart (auto-resume): Same command
3. Reduce parallelization to isolate issue

## Comparison with Custom Scripts

### When to Use Custom Scripts (merge_datasets.py)

✅ Experimentation and prototyping
✅ Small datasets (<1M documents)
✅ Learning and understanding the basics
✅ Quick testing of filtering ideas

### When to Use DataTrove

✅ Production training (>1M documents)
✅ Replicating SmolLM3's exact approach
✅ Processing at scale (billions of documents)
✅ Advanced deduplication needed
✅ Multi-stage processing pipelines
✅ Want production-tested quality filters

## Key Takeaways

1. **DataTrove is production-grade** - This is what HuggingFace actually uses
2. **Same filters as SmolLM3** - Gopher quality filters, language detection
3. **Unlimited scale** - Streaming mode handles any dataset size
4. **Automatic everything** - Parallelization, checkpointing, resume
5. **Start small, scale up** - Test with examples, then run full pipeline

## References

- **DataTrove GitHub**: https://github.com/huggingface/datatrove
- **SmolLM3 Blog**: https://huggingface.co/blog/smollm3
- **Gopher Paper** (quality filters): https://arxiv.org/abs/2112.11446
- **Our Complete Guide**: [DATATROVE_GUIDE.md](DATATROVE_GUIDE.md)

---

**You now have production-grade data processing like SmolLM3! 🚀**

Start with the simple examples, then scale up to the full pipeline when ready.

Questions? See [DATATROVE_GUIDE.md](DATATROVE_GUIDE.md) for detailed documentation.
