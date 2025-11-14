# SmolLM3 Dataset Weights - Quick Reference

## Top 10 Datasets by Weight

| Rank | Dataset | Weight | Tokens (8T) | Category |
|------|---------|--------|-------------|----------|
| 1 | **dclm** | 37.0% | 2.96T | General Web |
| 2 | **fineweb-edu** | 33.3% | 2.66T | Education |
| 3 | **Python (stack-edu)** | 2.5% | 200B | Code |
| 4 | **German (fw2)** | 2.2% | 176B | Multilingual |
| 5 | **Spanish (fw2)** | 2.0% | 160B | Multilingual |
| 6 | **pes2o** | 2.0% | 160B | Science |
| 7 | **C++ (stack-edu)** | 1.8% | 144B | Code |
| 8 | **finemath** | 1.7% | 136B | Math |
| 9 | **French (fw2)** | 1.6% | 128B | Multilingual |
| 10 | **Java (stack-edu)** | 1.3% | 104B | Code |

## Vietnamese Dataset

| Dataset | Weight | Tokens (8T) | Notes |
|---------|--------|-------------|-------|
| **FineWeb-2 Vietnamese** (vie_Latn) | 0.325% | 26B | Same as JP, KR, HI, TH |

## All Datasets by Category

### General Web (73.2% = 5.86T tokens)

#### English (70.3%)
| Dataset | Weight | Tokens |
|---------|--------|--------|
| dclm | 37.0% | 2.96T |
| fineweb-edu | 33.3% | 2.66T |
| pes2o | 2.0% | 160B |

#### Multilingual FineWeb-2 (9.0%)
| Language | Config | Weight | Tokens |
|----------|--------|--------|--------|
| German | deu_Latn | 2.2% | 176B |
| Spanish | spa_Latn | 2.0% | 160B |
| French | fra_Latn | 1.6% | 128B |
| Italian | ita_Latn | 1.05% | 84B |
| Chinese | cmn_Hans | 1.0% | 80B |
| Portuguese | por_Latn | 1.0% | 80B |
| Russian | rus_Cyrl | 1.0% | 80B |
| Japanese | jpn_Jpan | 0.325% | 26B |
| Korean | kor_Hang | 0.325% | 26B |
| Hindi | hin_Deva | 0.325% | 26B |
| Thai | tha_Thai | 0.325% | 26B |
| **Vietnamese** | **vie_Latn** | **0.325%** | **26B** |
| Persian | fas_Arab | 0.3% | 24B |
| Greek | ell_Grek | 0.225% | 18B |

#### Other (0.5%)
| Dataset | Weight | Tokens |
|---------|--------|--------|
| StackExchange | 0.4% | 32B |
| Wikipedia | 0.1% | 8B |

### Code (11.9% = 952B tokens)

#### Stack-Edu by Language (9.61%)
| Language | Weight | Tokens |
|----------|--------|--------|
| Python | 2.5% | 200B |
| C++ | 1.8% | 144B |
| Java | 1.3% | 104B |
| JavaScript | 1.3% | 104B |
| C | 0.7% | 56B |
| C# | 0.6% | 48B |
| PHP | 0.6% | 48B |
| HTML | 0.6% | 48B |
| Markdown | 0.5% | 40B |
| SQL | 0.4% | 32B |
| TypeScript | 0.3% | 24B |
| Swift | 0.1% | 8B |
| Ruby | 0.08% | 6.4B |
| Rust | 0.08% | 6.4B |
| Shell | 0.07% | 5.6B |
| Go | 0.05% | 4B |

#### GitHub & Kaggle (2.29%)
| Dataset | Weight | Tokens |
|---------|--------|--------|
| Pull Requests | 0.6% | 48B |
| Jupyter Scripts | 0.55% | 44B |
| GitHub Issues | 0.32% | 25.6B |
| Kaggle Notebooks | 0.05% | 4B |

### Math (2.7% = 216B tokens)

| Dataset | Weight | Tokens |
|---------|--------|--------|
| finemath | 1.7% | 136B |
| infiwebmath | 1.0% | 80B |

## Weight Distribution Charts

### By Category (Pie Chart)
```
General Web:     ███████████████████████████████████████ 73.2%
Code:            ██████████ 11.9%
Math:            ██ 2.7%
Other (Stage2/3):███████ 12.2%
```

### Top Languages in Multilingual Mix
```
German:    ██████ 2.2%
Spanish:   █████ 2.0%
French:    ████ 1.6%
Italian:   ███ 1.05%
Chinese:   ██ 1.0%
Russian:   ██ 1.0%
Portuguese:██ 1.0%
Vietnamese:█ 0.325%
Others:    ███ 1.775%
```

### Programming Languages
```
Python:      ██████████████ 2.5%
C++:         ██████████ 1.8%
Java:        ███████ 1.3%
JavaScript:  ███████ 1.3%
C:           ████ 0.7%
Others:      ████████ 3.11%
```

## Usage Examples

### 1. Exact SmolLM3 Replication
```bash
python merge_datasets.py \
  --config smollm3_weighted_config.yaml \
  --target-tokens 8000000000000 \  # 8T tokens
  --use-weights
```

### 2. Vietnamese-Enhanced (10x Vietnamese)
```yaml
# Multiply Vietnamese weight by 10
- name: "HuggingFaceFW/fineweb-2"
  config: "vie_Latn"
  weight: 0.0325  # 3.25% instead of 0.325%
```

### 3. Code-Heavy Mix (2x code)
```yaml
# Double all code weights
- name: "HuggingFaceTB/stack-edu"
  config: "python"
  weight: 0.05  # 5% instead of 2.5%
```

## Quick Facts

- **Total datasets**: 40+ individual dataset configs
- **Languages**: 14 natural languages + 16 programming languages
- **Vietnamese tokens**: 26 billion (0.325% of 8T)
- **Code tokens**: 952 billion (11.9% of 8T)
- **Math tokens**: 216 billion (2.7% of 8T)
- **Largest dataset**: dclm at 37% (2.96T tokens)
- **Smallest dataset**: Go at 0.05% (4B tokens)

## Weight Tiers

### Tier 1: Major (>10%)
- dclm: 37%
- fineweb-edu: 33.3%

### Tier 2: Large (1-10%)
- German, Spanish, Python, C++, etc.

### Tier 3: Medium (0.1-1%)
- Most programming languages, multilingual subsets

### Tier 4: Small (<0.1%)
- Go, Ruby, Rust, Shell, Kaggle

## Notes

1. Weights are from SmolLM3 Stage 1 (8T tokens)
2. Stage 2 and Stage 3 have different mixes (not included here)
3. All weights sum to ~88% (remainder is Stage 2/3)
4. Vietnamese has equal weight to Japanese, Korean, Hindi, Thai
5. Token counts are approximate based on 8T total

## See Also

- `SMOLLM3_WEIGHTS.md` - Detailed explanation
- `smollm3_weighted_config.yaml` - Full configuration file
- `SMOLLM3_DATASETS.md` - Dataset descriptions
