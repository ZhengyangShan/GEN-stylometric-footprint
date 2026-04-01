# AI Writers Have a Consistent Stylometric Footprint, but AI Editors Do Not

## Pipeline Overview

```
Step 1                Step 2              Step 3             Step 4
Data Processing  -->  LLM Generation -->  Cleaning &    -->  PCA &
(5 domains,           (8 models via       Feature            Feature
 5K prompts)           vLLM / HF)         Extraction         Analysis

     Step 5                Step 6                 Step 7
-->  Classification   -->  Feature Importance -->  Visualization
     (Logistic Reg.        (Multi-level:           (Publication
      + Transfer)           Global/Domain/LLM)      Figures)
```

## Features Extracted

14 stylometric features spanning 4 categories:

| Category | Features |
|----------|----------|
| **Lexical** | Lexical Diversity, Lexical Density, % Long Words |
| **Character** | Entropy, % Vowels, % Consonants, % Punctuation |
| **Structure** | Word Count, Sentence Count, Avg Sentence Length, Burstiness |
| **Readability** | Gunning Fog Index, Linsear Write Formula, Parse Tree Depth |

## LLMs Used for Generation

| Model | Size | Where to use |
|-------|------|-------|
| Qwen-2.5-7B-Instruct | 7B | AI-Generation |
| LLaMA-3.1-8B-Instruct | 8B | AI-Generation |
| Gemma-3-12B-IT | 12B | AI-Generation |
| GPT-OSS-20B | 20B | AI-Generation |
| Gemma-3-27B-IT | 27B | AI-Generation |
| LLaMA-3.3-70B-Instruct | 70B | AI-Generation and Ai-Editing |
| Qwen-2.5-72B-Instruct | 72B | AI-Generation and Ai-Editing |
| GPT-OSS-120B | 120B | AI-Generation and Ai-Editing |

## Quick Start

### Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk spacy textstat datasets statsmodels
python -m spacy download en_core_web_sm

# For generation (GPU required):
pip install torch transformers vllm
```

### Step-by-step

```bash
# 1. Build the prompt dataset
python 1_data_processing.py \
    --m4_csv /path/to/processed_M4.csv \
    --output data/five_domain_prompts.csv

# 2. Generate AI text (choose one method)
# vLLM (recommended for multi-model, high throughput)
python 2_generate_vllm.py \
    --input data/five_domain_prompts.csv \
    --output_dir data/generations/ \
    --models "Qwen/Qwen2.5-7B-Instruct,google/gemma-3-12b-it"

# 3. Clean LLM artifacts
python 3_clean_llm_artifacts.py \
    --input data/combined_results.csv \
    --output data/combined_results_cleaned.csv

# 4. Feature extraction (used as a library)
python -c "
from feature_extraction import extract_all_features
features = extract_all_features('Your sample text here.')
print(features)

# 5. Classification & transfer learning
python 5_classification.py

# 6. Feature importance analysis
python 6_feature_importance.py

# 7. Generate paper figures
python 7_feature_importance_plots.py
```

## Generation Parameters

All models use the following default generation parameters:

| Parameter | Value |
|-----------|-------|
| `max_new_tokens` | 1024 |
| `temperature` | 0.7 |
| `top_p` | 0.9 |
| `top_k` | 50 |
| `repetition_penalty` | 1.2 |
| `do_sample` | True |

## Data Domains

| Domain | Prompts | Task |
|--------|---------|------|
| Wikipedia |  1,000 | Write article with given title |
| WikiHow | 1,000 | Write how-to article with given title |
| ArXiv | 1,000 | Rephrase abstract with given title |
| Reddit | 1,000 | Answer question in specified voice style |
| Story Generation |  1,000 | Generate story from scenario |

## Key Analysis Outputs

- **Transfer matrices**: Cross-domain and cross-model generalization accuracy
- **Feature importance at 4 levels**: Global (L0), Per-Domain (L1), Per-LLM (L2), Per-Pair (L3)
- **Feature dynamics**: AI-edited text analysis
- **Feature robustness ranking**: Which features transfer well across conditions
