# Complete Dataset List for LLM Pretraining

This document provides detailed information about all datasets included in this project, organized by language and category.

## English Datasets (10 datasets)

### 1. HuggingFaceFW/fineweb
- **Size**: 15 trillion tokens
- **Category**: General, Technology
- **Description**: High-quality web dataset, extensively deduplicated and filtered
- **License**: Various (check individual sources)
- **Best For**: General language understanding, web content
- **Streaming**: Yes
- **Text Field**: `text`

### 2. allenai/c4
- **Size**: ~750GB
- **Category**: General
- **Description**: Colossal Clean Crawled Corpus - cleaned web text from Common Crawl
- **License**: ODC-BY
- **Best For**: General pretraining, broad domain coverage
- **Streaming**: Yes
- **Text Field**: `text`

### 3. openwebtext
- **Size**: ~40GB
- **Category**: General, QA
- **Description**: Open source recreation of OpenAI's WebText dataset from Reddit
- **License**: Various (from Reddit sources)
- **Best For**: High-quality general text, conversational content
- **Streaming**: Yes
- **Text Field**: `text`

### 4. bigcode/the-stack
- **Size**: 6TB+
- **Category**: Code
- **Description**: Permissively-licensed source code in 358 programming languages
- **License**: Multiple open source licenses
- **Best For**: Code generation, programming understanding
- **Streaming**: Yes
- **Text Field**: `content`

### 5. codeparrot/github-code
- **Size**: Several GB
- **Category**: Code
- **Description**: Large dataset of code from GitHub repositories
- **License**: Open source licenses
- **Best For**: Code understanding, software engineering
- **Streaming**: Yes
- **Text Field**: `code`

### 6. scientific_papers
- **Size**: Multiple GB
- **Category**: Science, Biology
- **Description**: Scientific papers from ArXiv and PubMed
- **License**: Various (mostly permissive)
- **Best For**: Scientific reasoning, academic writing
- **Streaming**: Yes
- **Text Field**: `article`

### 7. taesiri/arxiv_qa
- **Size**: Variable
- **Category**: Science, QA
- **Description**: ArXiv papers formatted as question-answer pairs
- **License**: ArXiv license
- **Best For**: Scientific QA, technical understanding
- **Streaming**: Yes
- **Text Field**: `text`

### 8. allenai/cosmos_qa
- **Size**: 35,600 problems
- **Category**: QA
- **Description**: Commonsense-based reading comprehension as multiple-choice questions
- **License**: CC-BY-4.0
- **Best For**: Commonsense reasoning, reading comprehension
- **Streaming**: No
- **Text Field**: `context`

### 9. microsoft/wiki_qa
- **Size**: Several thousand QA pairs
- **Category**: QA, General
- **Description**: Wikipedia-based question answering dataset
- **License**: Microsoft Research License
- **Best For**: Factual QA, general knowledge
- **Streaming**: No
- **Text Field**: `answer`

### 10. squad
- **Size**: 100,000+ QA pairs
- **Category**: QA
- **Description**: Stanford Question Answering Dataset - extractive QA
- **License**: CC BY-SA 4.0
- **Best For**: Reading comprehension, extractive QA
- **Streaming**: No
- **Text Field**: `context`

## Vietnamese Datasets (8 datasets)

### 1. bkai-foundation-models/BKAINewsCorpus
- **Size**: 32 million articles, 53GB
- **Category**: General, News
- **Description**: Massive Vietnamese news corpus, clean and deduplicated
- **License**: Check dataset page
- **Best For**: Vietnamese language understanding, news domain
- **Streaming**: Yes
- **Text Field**: `text`

### 2. bkai-foundation-models/NewsCategory
- **Size**: Large
- **Category**: General, News
- **Description**: Categorized Vietnamese news dataset
- **License**: Check dataset page
- **Best For**: Vietnamese classification, news understanding
- **Streaming**: Yes
- **Text Field**: `text`

### 3. vietgpt/wikipedia_vi
- **Size**: Variable (Wikipedia dump)
- **Category**: General
- **Description**: Vietnamese Wikipedia (2025 version)
- **License**: CC BY-SA
- **Best For**: Encyclopedic knowledge in Vietnamese
- **Streaming**: Yes
- **Text Field**: `text`

### 4. uitnlp/vietnamese_students_feedback
- **Size**: 16,000+ sentences
- **Category**: QA, Feedback
- **Description**: Vietnamese sentences with sentiment and topic annotations
- **License**: Check dataset page
- **Best For**: Vietnamese sentiment, educational domain
- **Streaming**: No
- **Text Field**: `sentence`

### 5. PhoGPT/vi_instructions
- **Size**: Variable
- **Category**: QA, Instructions
- **Description**: Vietnamese instruction-following dataset
- **License**: Check dataset page
- **Best For**: Instruction following in Vietnamese
- **Streaming**: No
- **Text Field**: `text`

### 6. uitnlp/vietnamese_qa
- **Size**: Variable
- **Category**: QA
- **Description**: Vietnamese question answering dataset
- **License**: Check dataset page
- **Best For**: Vietnamese QA tasks
- **Streaming**: No
- **Text Field**: `text`

### 7. vietgpt/vn_news_corpus
- **Size**: Large
- **Category**: Technology, News
- **Description**: Large Vietnamese news corpus with technology coverage
- **License**: Check dataset page
- **Best For**: Vietnamese tech content, news
- **Streaming**: Yes
- **Text Field**: `content`

### 8. uitnlp/vi_tech_corpus
- **Size**: Variable
- **Category**: Technology, Code
- **Description**: Vietnamese technology and programming content
- **License**: Check dataset page
- **Best For**: Vietnamese tech/code understanding
- **Streaming**: Yes
- **Text Field**: `text`

## Category Breakdown

### By Primary Category

#### Code (2 English datasets)
- bigcode/the-stack
- codeparrot/github-code

#### Science & Biology (2 English datasets)
- scientific_papers
- taesiri/arxiv_qa

#### Question Answering (6 total: 3 English, 3 Vietnamese)
**English:**
- allenai/cosmos_qa
- microsoft/wiki_qa
- squad

**Vietnamese:**
- uitnlp/vietnamese_students_feedback
- PhoGPT/vi_instructions
- uitnlp/vietnamese_qa

#### General/News (8 total: 3 English, 5 Vietnamese)
**English:**
- HuggingFaceFW/fineweb
- allenai/c4
- openwebtext

**Vietnamese:**
- bkai-foundation-models/BKAINewsCorpus
- bkai-foundation-models/NewsCategory
- vietgpt/wikipedia_vi
- vietgpt/vn_news_corpus
- uitnlp/vi_tech_corpus

#### Technology (2 Vietnamese datasets)
- vietgpt/vn_news_corpus
- uitnlp/vi_tech_corpus

## Dataset Selection Criteria

All datasets were selected based on:

1. **Quality**: Well-curated, cleaned data
2. **Size**: Sufficient size for pretraining
3. **Relevance**: Focus on code, tech, science, bio, QA domains
4. **Accessibility**: Publicly available on HuggingFace
5. **License**: Permissive licenses suitable for model training
6. **Diversity**: Coverage of multiple domains and formats
7. **Language**: English and Vietnamese focus

## Total Dataset Statistics

- **Total Datasets**: 18
- **English**: 10
- **Vietnamese**: 8
- **Streaming Capable**: 13
- **Non-Streaming**: 5
- **Code Datasets**: 2
- **Science Datasets**: 2
- **QA Datasets**: 6
- **General/News**: 8
- **Technology**: 4

## Dataset Access Notes

### Authentication
Some datasets may require HuggingFace authentication:
```bash
huggingface-cli login
```

### Streaming vs Non-Streaming
- **Streaming**: Memory efficient, processes data on-the-fly
- **Non-Streaming**: Loads entire dataset, faster iteration but needs more RAM

### Rate Limits
HuggingFace has rate limits for dataset downloads. For large-scale processing:
- Use streaming mode
- Add delays between requests if needed
- Consider downloading datasets first for offline processing

## Recommended Combinations

### For General LLM Pretraining
```yaml
english_datasets:
  - HuggingFaceFW/fineweb
  - allenai/c4
  - openwebtext
  - squad

vietnamese_datasets:
  - bkai-foundation-models/BKAINewsCorpus
  - vietgpt/wikipedia_vi
```

### For Code-Focused Model
```yaml
english_datasets:
  - bigcode/the-stack
  - codeparrot/github-code
  - allenai/c4

vietnamese_datasets:
  - uitnlp/vi_tech_corpus
```

### For Science/Tech Model
```yaml
english_datasets:
  - scientific_papers
  - taesiri/arxiv_qa
  - HuggingFaceFW/fineweb
  - allenai/cosmos_qa

vietnamese_datasets:
  - vietgpt/vn_news_corpus
  - uitnlp/vi_tech_corpus
```

### For QA-Focused Model
```yaml
english_datasets:
  - squad
  - allenai/cosmos_qa
  - microsoft/wiki_qa

vietnamese_datasets:
  - uitnlp/vietnamese_qa
  - PhoGPT/vi_instructions
```

## Notes on Vietnamese Datasets

Vietnamese datasets are particularly valuable because:
1. Less common than English datasets
2. High-quality curated sources (BKAI, UIT, VietGPT)
3. Cover important domains (news, tech, QA)
4. Include both formal and informal text

## Updates and Maintenance

This list was compiled in November 2025. Datasets may be updated or new ones added. Always:
- Check dataset pages for latest information
- Verify licenses before use
- Review dataset cards for updates
- Test datasets before large-scale processing

## References

- HuggingFace Datasets: https://huggingface.co/datasets
- BigCode Project: https://www.bigcode-project.org/
- Allen Institute for AI: https://allenai.org/
- BKAI Foundation: https://huggingface.co/bkai-foundation-models
- VietGPT: https://huggingface.co/vietgpt
- UIT NLP: https://huggingface.co/uitnlp
