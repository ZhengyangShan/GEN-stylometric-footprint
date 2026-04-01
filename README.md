# AI-Generated Text Detection via Stylometric Features

A complete pipeline for detecting AI-generated text using stylometric (writing style) features. We collect human-written texts from 5 domains, generate parallel AI texts using 8 LLMs, extract 14 linguistic features, and train logistic regression classifiers to distinguish human from machine writing. The analysis includes cross-domain/cross-model transfer learning and multi-level feature importance analysis.

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

## File Structure

| File | Description |
|------|-------------|
| `1_data_processing.py` | Builds multi-domain prompt dataset (Wikipedia, WikiHow, ArXiv, Reddit, Story Generation) with stratified sampling |
| `2_generate_hf_transformers.py` | Single-model text generation via HuggingFace Transformers (e.g., Gemma-3-12B) |
| `2_generate_vllm.py` | High-throughput multi-model generation via vLLM with automatic GPU memory management |
| `3_clean_llm_artifacts.py` | Cleans model-specific artifacts (prompt echoes, markdown, chain-of-thought, chatty openers, disclaimers) from all 8 LLMs |
| `3c_feature_extraction.py` | Extracts 14 stylometric features from text |
| `4_pca_analysis.py` | PCA, correlation, VIF, and factor analysis on the feature space |
| `5_classification.py` | Logistic regression classifiers with cross-domain and cross-model transfer learning analysis |
| `6_feature_importance.py` | Multi-level feature importance (Global / Per-Domain / Per-LLM / Per-Pair) |
| `7_feature_importance_plots.py` | Publication-quality figures for feature importance results |

## Features Extracted

14 stylometric features spanning 4 categories:

| Category | Features |
|----------|----------|
| **Lexical** | Lexical Diversity, Lexical Density, % Long Words |
| **Character** | Entropy, % Vowels, % Consonants, % Punctuation |
| **Structure** | Word Count, Sentence Count, Avg Sentence Length, Burstiness |
| **Readability** | Gunning Fog Index, Linsear Write Formula, Parse Tree Depth |

## LLMs Used for Generation

| Model | Size | Class |
|-------|------|-------|
| Qwen-2.5-7B-Instruct | 7B | Small |
| LLaMA-3.1-8B-Instruct | 8B | Small |
| Gemma-3-12B-IT | 12B | Small |
| GPT-OSS-20B | 20B | Mid |
| Gemma-3-27B-IT | 27B | Mid |
| LLaMA-3.3-70B-Instruct | 70B | Large |
| Qwen-2.5-72B-Instruct | 72B | Large |
| GPT-OSS-120B | 120B | Large |

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
# Option A: vLLM (recommended for multi-model, high throughput)
python 2_generate_vllm.py \
    --input data/five_domain_prompts.csv \
    --output_dir data/generations/ \
    --models "Qwen/Qwen2.5-7B-Instruct,google/gemma-3-12b-it"

# Option B: HuggingFace Transformers (single model)
python 2_generate_hf_transformers.py \
    --model google/gemma-3-12b-it \
    --input data/five_domain_prompts.csv \
    --output data/generations/gemma3_12b.jsonl

# 3. Clean LLM artifacts
python 3_clean_llm_artifacts.py \
    --input data/combined_results.csv \
    --output data/combined_results_cleaned.csv

# 4. Feature extraction (used as a library)
python -c "
from feature_extraction import extract_all_features
features = extract_all_features('Your sample text here.')
print(features)
"

# 5. PCA analysis
python 4_pca_analysis.py \
    --input data/models_generations_with_features.csv \
    --output_dir results/pca/

# 6. Classification & transfer learning
python 5_classification.py

# 7. Feature importance analysis
python 6_feature_importance.py

# 8. Generate paper figures
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

| Domain | Source | Prompts | Task |
|--------|--------|---------|------|
| Wikipedia | M4 Dataset | 1,000 | Write article with given title |
| WikiHow | M4 Dataset | 1,000 | Write how-to article with given title |
| ArXiv | M4 Dataset | 1,000 | Rephrase abstract with given title |
| Reddit | M4 Dataset | 1,000 | Answer question in specified voice style |
| Story Generation | WritingPrompts (HF) | 1,000 | Generate story from scenario |

## Key Analysis Outputs

- **Transfer matrices**: Cross-domain and cross-model generalization accuracy
- **Feature importance at 4 levels**: Global (L0), Per-Domain (L1), Per-LLM (L2), Per-Pair (L3)
- **Feature robustness ranking**: Which features transfer well across conditions
- **PCA visualization**: Human vs LLM separation in stylistic space
