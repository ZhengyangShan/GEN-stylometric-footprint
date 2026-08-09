# AI Writers Have a Consistent Stylometric Footprint, but AI Editors Do Not

Research code for a study on stylometric detection of machine-written text.

Two experiments live here:

1. **Writing.** 8 open-weight LLMs write text for the same 5,000 prompts that humans already answered, across 5 domains. Every document is scored on 14 interpretable stylometric features, and a logistic regression is trained to separate human from machine. The interesting question is not the accuracy number but how far a detector trained on one domain or one model carries over to the others.
2. **Editing.** The same features are measured on human documents after an LLM revises them, using the 303 editing prompts from EditLens (Thai et al., 2025) at revision ratios from 1% to 100%.

---

## Repository layout

```
stylometric-footprint/
├── code/
│   ├── data_collection/
│   │   ├── 1_data_processing.py          build the 5,000-prompt dataset
│   │   ├── 2_generate_vllm.py            generate with vLLM (multi-model)
│   │   └── 2_generate_hf_transformers.py generate with HF Transformers (single model)
│   ├── feature/
│   │   ├── 3_clean_llm_artifacts.py      strip per-model boilerplate and CoT leakage
│   │   ├── 3_feature_extraction.py       the 14 features (imported as a library)
│   │   └── 4_pca_analysis.py             PCA, correlation, VIF
│   ├── model/
│   │   ├── 5_classification.py           detection + cross-domain / cross-model transfer
│   │   └── 6_feature_importance.py       importance at 4 levels (L0-L3)
│   └── editing/
│       ├── run_editing.sh                entry point for the editing experiment
│       ├── run_editing.py                sampling, prompting, metrics
│       ├── editor.py                     revision-ratio logic
│       ├── edit_distance.py              char- and word-level Levenshtein
│       ├── llm_interface.py              vLLM wrapper
│       └── EditLens/                     submodule: pangramlabs/EditLens
└── data/                                 8 JSONL files, one per generating model
```

---

## Setup

```bash
git clone --recurse-submodules https://github.com/ZhengyangShan/stylometric-footprint.git
cd stylometric-footprint
```

If you already cloned without the flag: `git submodule update --init --recursive`

**Analysis only** (everything except generation):

```bash
pip install pandas numpy scipy scikit-learn statsmodels matplotlib seaborn \
            nltk spacy textstat datasets joblib tqdm
python -m spacy download en_core_web_sm
```

**Generation and editing** (GPU required):

```bash
pip install torch transformers vllm rapidfuzz
```

NLTK's `punkt` and `stopwords` download themselves on first import of the feature module. One function in `5_classification.py` also imports `networkx`, install it if you want the feature-correlation graph.

---

## Data

`data/` holds the raw generations, one JSONL file per model, roughly 5,000 rows each:

| File | Model |
|---|---|
| `Qwen_Qwen2.5-7B-Instruct_text.jsonl` | Qwen-2.5-7B-Instruct |
| `meta-llama_Llama-3.1-8B-Instruct_text.jsonl` | LLaMA-3.1-8B-Instruct |
| `Gemma3_12B_text.jsonl` | Gemma-3-12B-IT |
| `gpt-oss20B_text.jsonl` | GPT-OSS-20B |
| `Gemma3_27B_text.jsonl` | Gemma-3-27B-IT |
| `meta-llama_Llama-3.3-70B-Instruct_text.jsonl` | LLaMA-3.3-70B-Instruct |
| `Qwen_Qwen2.5-72B-Instruct_text.jsonl` | Qwen-2.5-72B-Instruct |
| `gpt-oss120B_text.jsonl` | GPT-OSS-120B |

Each line:

```json
{"prompt": "...", "human_text": "...", "source": "wikipedia", "machine_text": "..."}
```

Files written by the vLLM script carry an extra `"model"` key. `human_text` is the reference document from the source corpus, so every row is a matched human/machine pair on the same prompt.

Prompts come from the M4 corpus (Wikipedia, WikiHow, ArXiv, Reddit) plus WritingPrompts (`euclaise/writingprompts` on HuggingFace) for story generation.

| Domain | Prompts | Task |
|---|---|---|
| Wikipedia | 1,000 | Write an article for a given title |
| WikiHow | 1,000 | Write a how-to article for a given title |
| ArXiv | 1,000 | Rephrase an abstract for a given title |
| Reddit | 1,000 | Answer a question in a specified voice |
| Story generation | 1,000 | Write a story from a scenario |

Wikipedia is sampled stratified by prompt length, Reddit is balanced across voice styles, WikiHow drops prompts over 2,000 characters. Seed is 42 throughout.

---

## Pipeline

```
1. build prompts  →  2. generate  →  3. clean  →  3b. extract features
                                                        ↓
                    6. importance  ←  5. classify  ←  4. PCA / VIF
```

### 1. Build the prompt dataset

```bash
python code/data_collection/1_data_processing.py \
    --m4_csv /path/to/processed_M4.csv \
    --output data/five_domain_prompts.csv
```

Output columns: `prompt`, `human_text`, `source`.

### 2. Generate

vLLM handles several models in one run and resumes from whatever is already in the output file:

```bash
python code/data_collection/2_generate_vllm.py \
    --input data/five_domain_prompts.csv \
    --output_dir data/ \
    --models "Qwen/Qwen2.5-7B-Instruct,google/gemma-3-12b-it"
```

Batch size, `max_model_len`, and GPU memory fraction are set per model inside the script, with a fallback for anything unlisted. The Transformers path is simpler and takes one model at a time:

```bash
python code/data_collection/2_generate_hf_transformers.py \
    --model google/gemma-3-12b-it \
    --input data/five_domain_prompts.csv \
    --output data/Gemma3_12B_text.jsonl \
    --batch_size 32
```

### 3. Clean generation artifacts

Every model leaves its own residue: Gemma echoes the `user/model` chat scaffold, GPT-OSS emits chain-of-thought before an `assistantfinal` token, LLaMA-70B repeats the prompt, several models append disclaimers. Left in place, a classifier would learn the boilerplate instead of the writing style.

```bash
python code/feature/3_clean_llm_artifacts.py \
    --input  data/combined_results.csv \
    --output data/combined_results_cleaned.csv \
    --verify_gpt_oss20b
```

This step reads a wide CSV with `prompt` plus one text column per model, named `qwen-7B_text`, `gemma3-12B_text`, `gemma3-27B_text`, `qwen-72B_text`, `llama31-8B_text`, `llama33-70B_text`, `gpt-oss20B_text`, `gpt-oss120B_text`. Missing columns are skipped with a warning. `--verify_gpt_oss20b` prints how many rows still start with a known artifact, which is the fastest way to check whether the regex set is keeping up.

### 3b. Extract features

`3_feature_extraction.py` is a library, not a script. The leading digit blocks a normal import, so load it by name:

```python
import sys, importlib
sys.path.append("code/feature")
fx = importlib.import_module("3_feature_extraction")

fx.extract_all_features("Your sample text here.")
# {'lexical_diversity': 0.83, 'entropy': 4.12, 'num_words': 24, ...}
```

Run it over the cleaned text to build the long-format table the analysis scripts expect: one row per document with `text`, `label` (0 human, 1 machine), `source` (domain), `generator` (`human` or a model column name such as `gemma3-27B_text`), and the 14 feature columns.

### 4. PCA, correlation, VIF

```bash
python code/feature/4_pca_analysis.py \
    --input data/models_generations_with_features.csv \
    --output_dir pca_results/
```

Writes explained-variance and 2D/3D scatter plots, a correlation matrix, `loadings.csv`, and `vif_table.csv`.

### 5. Classification and transfer

```bash
python code/model/5_classification.py \
    --data data/models_generations_with_features.csv \
    --output figures/ \
    --analysis both \
    --top_k 5 \
    --n_runs 5
```

`--analysis` picks `source` (cross-domain), `generator` (cross-model), or `both`. Classes are downsampled to the minority size, and every cell is averaged over `--n_runs` splits. The script drops the `summarization` source and the `phi4` and `mistral` generators by default, which keeps runs consistent with the paper.

### 6. Feature importance at four levels

```bash
python code/model/6_feature_importance.py \
    --data data/models_generations_with_features.csv \
    --output feature_importance_results/ \
    --n_runs 5 --n_perm 10
```

| Level | Model trained on | Question it answers |
|---|---|---|
| L0 global | all data | Which features separate machine from human overall? |
| L1 domain | one domain | Which features matter inside this domain? |
| L2 LLM | one model vs human | What gives this particular model away? |
| L3 pair | one domain × one model | What does a specialist pick up on? |

Each level retrains from scratch, so L1/L2/L3 are not the global model applied to subsets. Two signals are reported per feature: mean absolute logistic coefficient, and permutation importance (accuracy drop when the feature is shuffled). Results land in `L0_global/`, `L1_domain/`, `L2_llm/`, `L3_pair/`, and `comparison/`, including `feature_robustness_ranking.csv`, which ranks features by how often they survive in the top 5 across conditions.

---

## Feature set

14 features, four groups:

| Group | Features |
|---|---|
| **Lexical** | lexical diversity (type/token), lexical density (content words), % words over 6 characters |
| **Character** | token entropy, % vowels, % consonants, % punctuation |
| **Structure** | word count, sentence count, average sentence length, burstiness (std/mean of word frequencies) |
| **Readability** | Gunning Fog, Linsear Write, dependency parse tree depth |

Tokenization is NLTK, parse depth comes from spaCy `en_core_web_sm`, and the two readability scores come from `textstat`.

---

## Models

| Model | Params | Used for |
|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | 7B | writing |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | writing |
| `google/gemma-3-12b-it` | 12B | writing |
| `openai/gpt-oss-20b` | 20B | writing |
| `google/gemma-3-27b-it` | 27B | writing |
| `meta-llama/Llama-3.3-70B-Instruct` | 70B | writing and editing |
| `Qwen/Qwen2.5-72B-Instruct` | 72B | writing and editing |
| `openai/gpt-oss-120b` | 120B | writing and editing |

Sampling settings, shared by both generation scripts:

| Parameter | Value |
|---|---|
| `max_new_tokens` | 1024 |
| `temperature` | 0.7 |
| `top_p` | 0.9 |
| `top_k` | 50 |
| `repetition_penalty` | 1.2 |

---

## AI-editing experiment

Applies the 303 EditLens editing prompts to human-written documents at a random revision ratio per document, then measures character- and word-level edit distance alongside the stylometric shift. Run from `code/`, needs 4 GPUs (`TENSOR_PARALLEL_SIZE=4`).

```bash
cd code
bash editing/run_editing.sh                                    # all 303 prompts
bash editing/run_editing.sh --max-prompts 10                   # quick test
bash editing/run_editing.sh --prompt-split test --samples-per-domain 200 \
                           --min-revision-pct 50 --max-revision-pct 50
```

Set `MODEL_NAME` at the top of `run_editing.sh` to switch between `Qwen/Qwen2.5-72B-Instruct` (default), `meta-llama/Llama-3.3-70B-Instruct`, and `openai/gpt-oss-120b`.

Useful flags: `--prompt-split` (train/val/test), `--prompt-category` (for example `tone_and_style`), `--samples-per-domain`, `--min-revision-pct` / `--max-revision-pct`, `--filter-label` (0 edits human text, 1 edits AI text).

Results go to `editing/results/editlens_<timestamp>/`:

```
sampled_documents.csv        the documents chosen for editing
experiment_results_raw.csv   edits, written before metrics so a crash costs nothing
experiment_results.csv       edits plus edit-distance metrics
experiment_results.json      same, as JSON
```

---

## Credits

- **M4** for the Wikipedia, WikiHow, ArXiv, and Reddit prompts. Wang et al., *M4: Multi-generator, Multi-domain, and Multi-lingual Black-Box Machine-Generated Text Detection*, EACL 2024. [aclanthology.org/2024.eacl-long.83](https://aclanthology.org/2024.eacl-long.83/)
- **WritingPrompts** for story generation. [huggingface.co/datasets/euclaise/writingprompts](https://huggingface.co/datasets/euclaise/writingprompts)
- **EditLens** for the 303 editing prompts. Thai et al., ICLR 2026. [arXiv:2510.03154](https://arxiv.org/abs/2510.03154), code included here as a submodule from [pangramlabs/EditLens](https://github.com/pangramlabs/EditLens).
