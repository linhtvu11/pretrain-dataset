# SmolLM3 Three-Stage Training Configuration

**Source**: Official SmolLM3 training configs from HuggingFace
- Stage 1: https://huggingface.co/datasets/HuggingFaceTB/smollm3-configs/resolve/main/stage1_8T.yaml
- Stage 2: https://huggingface.co/datasets/HuggingFaceTB/smollm3-configs/resolve/main/stage2_8T_9T.yaml
- Stage 3: https://huggingface.co/datasets/HuggingFaceTB/smollm3-configs/resolve/main/stage3_9T_11T.yaml

This document explains the three-stage training approach used by SmolLM3 and how to use each configuration.

## Overview

SmolLM3 uses a **progressive three-stage training approach** with different dataset mixes for each stage:

| Stage | Token Range | Duration | Focus | Config File |
|-------|-------------|----------|-------|-------------|
| **Stage 1** | 0 - 8T | 8 trillion tokens | Balanced foundation (general web, code, math) | `smollm3_weighted_config.yaml` |
| **Stage 2** | 8T - 9T | 1 trillion tokens | Educational code + enhanced math | `smollm3_stage2_config.yaml` |
| **Stage 3** | 9T - 11T | 2 trillion tokens | Reasoning + advanced code (decay stage) | `smollm3_stage3_config.yaml` |

**Total training**: 11 trillion tokens across all three stages

## Stage 1: Foundation (0-8T tokens)

### Focus
Build a strong **general foundation** with balanced knowledge across web content, code, and mathematics.

### Key Characteristics
- **General Web**: 73.2% (primary focus)
- **Code**: 11.9% (basic coverage)
- **Math**: 2.7% (fundamental math)
- **Vietnamese**: 0.325% (good multilingual representation)

### Dataset Mix
```yaml
Top datasets:
- dclm (37.0%) - Largest web content dataset
- fineweb-edu (33.3%) - Educational web content
- fineweb-2 multilingual (9.0%) - 14 languages including Vietnamese
- stack-edu languages (9.61%) - Basic code education
- finemath (1.7%) - Mathematical content
```

### When to Use
- Initial pretraining from scratch
- Building general-purpose language models
- When you need broad knowledge coverage
- Foundation for further specialization

### Usage
```bash
python merge_datasets.py \
  --config smollm3_weighted_config.yaml \
  --output-dir ./smollm3_stage1_dataset \
  --use-weights
```

## Stage 2: Educational Focus (8T-9T tokens)

### Focus
Enhance the model's **educational code understanding** and **mathematical reasoning** capabilities.

### Key Characteristics
- **General Web**: 69.5% (slightly reduced)
- **Code**: 13.2% (increased educational focus)
- **Math**: 9.08% (3.4x increase from Stage 1)
- **Vietnamese**: 0.005% (drastically reduced)

### Major Changes from Stage 1
1. **Math content tripled**: 2.7% → 9.08%
   - finemath increased to 2.0%
   - finemath-4plus added at 4.5%
   - OpenMathInstruct datasets added

2. **Educational code enhanced**: Uses "real" versions of stack-edu
   - Python: 2.5% → 2.0% (real version)
   - More balanced across languages

3. **Vietnamese minimized**: 0.325% → 0.005%
   - Focus shifts to English educational content

### Dataset Mix
```yaml
Enhanced datasets:
- finemath-4plus (4.5%) - High quality math
- stack-edu-real versions (10.61%) - Educational code
- OpenMathInstruct datasets - Math reasoning
```

### When to Use
- After Stage 1 foundation training
- When you want to improve code understanding
- To enhance mathematical reasoning
- For educational/academic focus

### Usage
```bash
python merge_datasets.py \
  --config smollm3_stage2_config.yaml \
  --output-dir ./smollm3_stage2_dataset \
  --use-weights
```

## Stage 3: Reasoning & Advanced Code (9T-11T tokens)

### Focus
Develop advanced **reasoning capabilities** and **expert-level code understanding**. Called the "decay stage" in training terminology.

### Key Characteristics
- **General Web**: 64.19% (further reduced)
- **Code**: 20.2% (massively increased - 1.7x Stage 1)
- **Math + Reasoning**: 15.61% (almost 6x Stage 1)
- **Vietnamese**: 0.005% (minimal)

### Major Changes from Stage 2
1. **Code dominates**: 13.2% → 20.2%
   - Python: 2.0% → 7.0% (3.5x increase!)
   - C++: 1.8% → 4.4% (2.4x increase)
   - All languages use "real" versions

2. **Reasoning datasets added**: 6.1% new content
   - openmathreasoning-4k (1.5%)
   - open-codereasoning-4k (1.5%)
   - natural_reasoning (1.1%)
   - OpenMathInstruct-2 (2.0%)

3. **Math content doubled**: 9.08% → 15.61%
   - megamath-text-code-block (5.0%) - NEW
   - finemath-4plus (4.5%)
   - Multiple reasoning datasets

### Dataset Mix
```yaml
Key datasets:
- dclm (30.0%) - Reduced general web
- fineweb-edu (28.0%) - Reduced educational web
- stack-edu Python-real (7.0%) - MASSIVE code focus
- megamath-text-code-block (5.0%) - NEW major math dataset
- Reasoning datasets (6.1%) - openmathreasoning, codereasoning, natural_reasoning
```

### When to Use
- Final stage after Stage 1 and 2
- When you need advanced reasoning capabilities
- For code-heavy applications
- To develop expert programming assistants
- Research and problem-solving focused models

### Usage
```bash
python merge_datasets.py \
  --config smollm3_stage3_config.yaml \
  --output-dir ./smollm3_stage3_dataset \
  --use-weights
```

## Visual Comparison

### Dataset Category Evolution

```
Category         | Stage 1 | Stage 2 | Stage 3 | Change (1→3)
-----------------|---------|---------|---------|-------------
General Web      | 73.2%   | 69.5%   | 64.2%   | -9.0%
Code             | 11.9%   | 13.2%   | 20.2%   | +8.3%
Math + Reasoning |  2.7%   |  9.1%   | 15.6%   | +12.9%
Vietnamese       |  0.32%  | 0.005%  | 0.005%  | -0.315%
```

### Python Code Weight Evolution

```
Stage 1:  ▓▓░░░░░░░░░░░░░░░░░░  2.5%
Stage 2:  ▓▓░░░░░░░░░░░░░░░░░░  2.0%
Stage 3:  ▓▓▓▓▓▓▓░░░░░░░░░░░░░  7.0%  ⬆ 180% increase!
```

### Math Content Weight Evolution

```
Stage 1:  ▓▓░░░░░░░░░░░░░░░░░░  2.7%
Stage 2:  ▓▓▓▓▓▓▓▓▓░░░░░░░░░░░  9.1%  ⬆ 237% increase
Stage 3:  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░ 15.6%  ⬆ 478% increase!
```

## Training Strategy

### Sequential Training (Recommended)

Train in sequence for best results:

```bash
# Stage 1: Foundation (0-8T)
python merge_datasets.py --config smollm3_weighted_config.yaml --output-dir ./stage1
# Train your model on stage1 dataset for 8T tokens

# Stage 2: Educational Focus (8T-9T)
python merge_datasets.py --config smollm3_stage2_config.yaml --output-dir ./stage2
# Continue training from stage1 checkpoint for 1T tokens

# Stage 3: Reasoning & Advanced Code (9T-11T)
python merge_datasets.py --config smollm3_stage3_config.yaml --output-dir ./stage3
# Continue training from stage2 checkpoint for 2T tokens
```

### Why Three Stages?

1. **Curriculum Learning**: Start broad, then specialize
2. **Stability**: Foundation first prevents overfitting to specialized content
3. **Efficiency**: Each stage builds on previous knowledge
4. **Quality**: Progressive refinement of capabilities

### Stage 3 as "Decay Stage"

Stage 3 is called the "decay stage" because:
- Learning rate is decayed/reduced
- Focus shifts to specialized, high-quality content
- Model refines existing knowledge rather than learning new basics
- Reasoning and advanced capabilities are emphasized

## Vietnamese Content Across Stages

### Vietnamese Weight Changes

| Stage | Weight | % of Total | Equivalent Tokens | Purpose |
|-------|--------|-----------|-------------------|---------|
| **Stage 1** | 0.00325 | 0.325% | ~26 billion | Multilingual foundation |
| **Stage 2** | 0.00005 | 0.005% | ~500 million | Minimal maintenance |
| **Stage 3** | 0.00005 | 0.005% | ~1 billion | Minimal maintenance |

### Why Vietnamese Reduces

1. **Stage 1**: Build multilingual capabilities
2. **Stage 2 & 3**: Focus on English educational/reasoning content
3. **Practical**: Most code, math, and reasoning datasets are in English

### For Vietnamese-Focused Models

If you want to maintain Vietnamese capabilities:

```yaml
# Option 1: Keep Stage 1 Vietnamese weight throughout
- name: "HuggingFaceFW/fineweb-2"
  config: "vie_Latn"
  weight: 0.00325  # Maintain 0.325% in all stages

# Option 2: Increase Vietnamese weight
- name: "HuggingFaceFW/fineweb-2"
  config: "vie_Latn"
  weight: 0.05  # Increase to 5%

# Option 3: Add Vietnamese-specific datasets
- name: "vietgpt/binhvq_news_vi"
  weight: 0.05
```

**Remember**: Normalize all weights to sum to 1.0!

## Key Insights

### 1. Progressive Specialization
- Stage 1: Generalist foundation
- Stage 2: Educational focus
- Stage 3: Expert reasoning

### 2. Code Focus Increases Dramatically
- Stage 1: 11.9% code
- Stage 3: 20.2% code (70% increase)
- Python alone goes from 2.5% to 7.0%

### 3. Math & Reasoning Becomes Central
- Stage 1: 2.7% math
- Stage 3: 15.6% math + reasoning (477% increase)
- New reasoning datasets in Stage 3

### 4. Multilingual Support Front-Loaded
- Most multilingual content in Stage 1 (foundation)
- Stages 2 & 3 focus on English for specialized content
- Makes sense: Most technical/educational content is in English

### 5. "Real" Stack-Edu Versions
- Stage 2 & 3 use "-real" versions of stack-edu
- Likely indicates higher quality, real-world code examples
- vs synthetic or generated code in Stage 1

## Recommendations

### For General-Purpose Models
Use all three stages in sequence with exact SmolLM3 weights.

### For Code-Focused Models
- Keep Stage 1 as-is for foundation
- Increase code weights in Stage 2 & 3
- Consider starting Stage 2 earlier (e.g., at 6T instead of 8T)

### For Multilingual Models
- Increase Vietnamese weight in all stages
- Add more Vietnamese-specific datasets
- Consider balancing language-specific stages

### For Math/Reasoning Models
- Keep Stage 1 as-is
- Start Stage 2 earlier
- Extend Stage 3 beyond 11T tokens

### For Fast Prototyping
- Use only Stage 1 for quick testing
- Then jump to Stage 3 for specialized capabilities
- Skip Stage 2 if not focused on education

## Files in This Repository

| File | Stage | Focus | Datasets | Vietnamese |
|------|-------|-------|----------|------------|
| `smollm3_weighted_config.yaml` | Stage 1 (0-8T) | Balanced foundation | 40+ | 0.325% |
| `smollm3_stage2_config.yaml` | Stage 2 (8T-9T) | Educational code + math | 47 | 0.005% |
| `smollm3_stage3_config.yaml` | Stage 3 (9T-11T) | Reasoning + advanced code | 57 | 0.005% |

## Verification

To verify weights sum correctly:

```python
import yaml

# Check Stage 1
with open('smollm3_weighted_config.yaml') as f:
    config = yaml.safe_load(f)
    total = sum(d['weight'] for d in config['datasets'])
    print(f"Stage 1 total weight: {total:.4f}")  # Should be ~1.0

# Check Stage 2
with open('smollm3_stage2_config.yaml') as f:
    config = yaml.safe_load(f)
    total = sum(d['weight'] for d in config['datasets'])
    print(f"Stage 2 total weight: {total:.4f}")  # Should be ~1.0

# Check Stage 3
with open('smollm3_stage3_config.yaml') as f:
    config = yaml.safe_load(f)
    total = sum(d['weight'] for d in config['datasets'])
    print(f"Stage 3 total weight: {total:.4f}")  # Should be ~1.0
```

## References

- SmolLM3 Blog: https://huggingface.co/blog/smollm3
- SmolLM3 Collection: https://huggingface.co/collections/HuggingFaceTB/smollm3-pretraining-datasets
- Stage 1 Config: https://huggingface.co/datasets/HuggingFaceTB/smollm3-configs/resolve/main/stage1_8T.yaml
- Stage 2 Config: https://huggingface.co/datasets/HuggingFaceTB/smollm3-configs/resolve/main/stage2_8T_9T.yaml
- Stage 3 Config: https://huggingface.co/datasets/HuggingFaceTB/smollm3-configs/resolve/main/stage3_9T_11T.yaml

## License

All SmolLM3 datasets follow their respective licenses on HuggingFace. Check each dataset's license before commercial use.

## Notes

1. These configs are based on SmolLM3's **production training** - they are proven to work well
2. Weights are exact matches from official configs
3. All dataset paths have been verified to exist on HuggingFace
4. The three-stage approach is a form of **curriculum learning**
5. "Decay stage" (Stage 3) refers to the learning rate schedule, not data quality
6. Vietnamese content is minimal in Stage 2 & 3 because most specialized content is in English
7. For Vietnamese-focused models, adjust weights accordingly

## Questions?

If you're unsure which stage to use:
- **Starting fresh?** → Stage 1
- **Have a pretrained model?** → Stage 2 or 3
- **Want reasoning?** → Stage 3
- **Want multilingual?** → Stage 1 (and adjust Stage 2 & 3)
- **Want code expert?** → All three stages, or Stage 1 + 3

The SmolLM3 approach is highly effective and proven at scale!
