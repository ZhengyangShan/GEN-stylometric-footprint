"""
EditLens experiment: Apply 303 editing prompts from Thai et al. (2025)
to sampled documents with random revision ratios.

This script:
1. Randomly samples N documents per domain
2. For each EditLens prompt, assigns a random revision ratio (1-100%)
3. Applies each prompt to every sampled document
4. Computes edit distance metrics (char-level, word-level)
5. Saves detailed results for analysis
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from editing.editor import Editor, RevisionConfig
from editing.edit_distance import calculate_edit_distance, levenshtein_distance
from editing.llm_interface import generate_chat_batch, get_llm
from editing.prompts import (
    get_editlens_prompt,
    get_editlens_prompts,
    EDITLENS_CATEGORIES,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run EditLens 303-prompt experiment with random revision ratios"
    )

    # Input/Output
    p.add_argument("--csv-path", type=str, required=True, help="Input CSV path")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory (default: editing/results/editlens_<timestamp>)")
    p.add_argument("--text-col", type=str, default="text")
    p.add_argument("--domain-col", type=str, default="source")
    p.add_argument("--label-col", type=str, default="label")
    p.add_argument("--filter-label", type=int, default=0,
                   help="Only edit rows with this label (0=human, 1=AI)")

    # Model
    p.add_argument("--model-name", type=str,
                   default="meta-llama/Llama-3.3-70B-Instruct")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.9)

    # Sampling config
    p.add_argument("--samples-per-domain", type=int, default=5,
                   help="Documents to sample per domain")
    p.add_argument("--prompt-split", type=str, default=None,
                   choices=["train", "val", "test"],
                   help="EditLens split to use (default: all 303)")
    p.add_argument("--prompt-category", type=str, default=None,
                   choices=EDITLENS_CATEGORIES,
                   help="Filter prompts to a single category")
    p.add_argument("--max-prompts", type=int, default=None,
                   help="Cap number of prompts (randomly sampled)")
    p.add_argument("--min-revision-pct", type=int, default=1)
    p.add_argument("--max-revision-pct", type=int, default=100)

    # Processing
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sample_documents(
    df: pd.DataFrame, domain_col: str, text_col: str,
    n_per_domain: int, seed: int, max_doc_chars: int = 0,
) -> pd.DataFrame:
    sampled_dfs = []
    for domain in sorted(df[domain_col].unique()):
        domain_df = df[df[domain_col] == domain]
        if max_doc_chars > 0:
            before = len(domain_df)
            domain_df = domain_df[domain_df[text_col].str.len() <= max_doc_chars]
            skipped = before - len(domain_df)
            if skipped > 0:
                print(f"  [{domain}] Filtered out {skipped} docs exceeding {max_doc_chars} chars")
        n = min(n_per_domain, len(domain_df))
        sampled_dfs.append(domain_df.sample(n=n, random_state=seed))
        print(f"  Sampled {n} docs from {domain}")
    return pd.concat(sampled_dfs, ignore_index=True)


def compute_edit_metrics(original: str, edited: str) -> Dict[str, Any]:
    """Compute comprehensive edit distance metrics."""
    edit_result = calculate_edit_distance(original, edited)

    orig_words = original.split()
    edit_words = edited.split()

    char_lev = levenshtein_distance(original, edited)
    max_char = max(len(original), len(edited))
    char_ratio = char_lev / max_char if max_char > 0 else 0.0

    word_lev = levenshtein_distance(" ".join(orig_words), " ".join(edit_words))

    return {
        "actual_edit_ratio_pct": round(edit_result.char_ratio * 100, 2),
        "char_levenshtein_distance": char_lev,
        "char_edit_ratio_pct": round(char_ratio * 100, 2),
        "word_levenshtein_distance": word_lev,
        "original_char_count": len(original),
        "edited_char_count": len(edited),
        "original_word_count": len(orig_words),
        "edited_word_count": len(edit_words),
        "word_count_diff": len(edit_words) - len(orig_words),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")

    # ── Load & filter data ──────────────────────────────────
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    if args.label_col in df.columns:
        df = df[df[args.label_col] == args.filter_label].copy()
        print(f"Filtered to {len(df)} rows (label=={args.filter_label})")

    # ── Sample documents ────────────────────────────────────
    print(f"\nSampling {args.samples_per_domain} documents per domain:")
    # Auto-compute max doc length from MAX_MODEL_LEN (leave room for prompt overhead + output)
    max_model_len = int(os.environ.get("MAX_MODEL_LEN", 4096))
    # ~4 chars/token, reserve max_new_tokens for output + ~300 tokens for prompt template
    max_doc_chars = (max_model_len - args.max_new_tokens - 300) * 4
    print(f"  MAX_MODEL_LEN={max_model_len}, max_new_tokens={args.max_new_tokens} -> max_doc_chars={max_doc_chars}")

    sampled_df = sample_documents(
        df, args.domain_col, args.text_col,
        args.samples_per_domain, args.seed, max_doc_chars=max_doc_chars,
    )
    print(f"Total sampled: {len(sampled_df)} documents")

    # ── Select EditLens prompts ─────────────────────────────
    prompts = get_editlens_prompts(split=args.prompt_split, category=args.prompt_category)
    if args.max_prompts and args.max_prompts < len(prompts):
        prompts = random.sample(prompts, args.max_prompts)
    print(f"\nEditLens prompts: {len(prompts)}")
    if args.prompt_split:
        print(f"  Split filter: {args.prompt_split}")
    if args.prompt_category:
        print(f"  Category filter: {args.prompt_category}")

    # ── Assign random revision % to each prompt ─────────────
    revision_pcts = {
        i: random.randint(args.min_revision_pct, args.max_revision_pct)
        for i in range(len(prompts))
    }

    # ── Output directory ────────────────────────────────────
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = Path("editing/results") / f"editlens_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # ── Save config ─────────────────────────────────────────
    config = vars(args)
    config["timestamp"] = datetime.now().isoformat()
    config["n_prompts"] = len(prompts)
    config["n_documents"] = len(sampled_df)
    config["total_runs"] = len(prompts) * len(sampled_df)
    config["revision_pcts_per_prompt"] = revision_pcts
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    sampled_df.to_csv(output_dir / "sampled_documents.csv", index=False)

    # ── Save prompt list for reference ──────────────────────
    with open(output_dir / "prompts_used.json", "w") as f:
        prompts_with_pct = []
        for i, p in enumerate(prompts):
            prompts_with_pct.append({**p, "prompt_idx": i, "revision_pct": revision_pcts[i]})
        json.dump(prompts_with_pct, f, indent=2, ensure_ascii=False)

    # ── Warm up vLLM model ───────────────────────────────────
    print(f"\nLoading model {args.model_name}...")
    get_llm(args.model_name)

    # ── Build ALL conversations upfront ─────────────────────
    doc_rows = []
    for idx, row in sampled_df.iterrows():
        doc_rows.append({
            "idx": int(idx),
            "text": str(row[args.text_col]),
            "domain": row[args.domain_col],
            "row": row,
        })

    total_runs = len(prompts) * len(doc_rows)
    print(f"\n{'='*60}")
    print(f"EditLens Experiment")
    print(f"  Prompts : {len(prompts)}")
    print(f"  Docs    : {len(doc_rows)}")
    print(f"  Total   : {total_runs} edits (single batch to vLLM)")
    print(f"  Rev %   : random [{args.min_revision_pct}, {args.max_revision_pct}]")
    print(f"{'='*60}\n")

    job_meta = []
    conversations = []

    for prompt_idx, prompt_entry in enumerate(prompts):
        prompt_text = prompt_entry["prompt"]
        rev_pct = revision_pcts[prompt_idx]

        for doc in doc_rows:
            sys_msg, usr_msg = get_editlens_prompt(prompt_text, doc["text"], rev_pct)
            conversations.append([
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": usr_msg},
            ])
            job_meta.append({
                "prompt_idx": prompt_idx,
                "prompt_text": prompt_text,
                "category": prompt_entry["category"],
                "contributor": prompt_entry["contributor"],
                "prompt_split": prompt_entry["split"],
                "target_revision_pct": rev_pct,
                "doc_idx": doc["idx"],
                "domain": doc["domain"],
                "original_text": doc["text"],
                "row": doc["row"],
            })

    print(f"Built {len(conversations)} conversations. Sending to vLLM...")
    start_time = time.time()

    # ── Single batch call — vLLM handles internal scheduling ─
    try:
        edited_texts = generate_chat_batch(
            conversations=conversations,
            model_name=args.model_name,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        global_status = "success"
    except Exception as e:
        print(f"ERROR: {e}")
        edited_texts = [m["original_text"] for m in job_meta]
        global_status = f"error: {str(e)[:200]}"

    gen_time = time.time() - start_time
    print(f"Generation done in {gen_time/60:.1f} min")

    # ── Save raw results IMMEDIATELY (no metrics yet) ────────
    print("Saving raw results (before metrics)...")
    raw_results: List[Dict[str, Any]] = []

    for i, meta in enumerate(job_meta):
        original_text = meta["original_text"]
        edited_text = edited_texts[i] if i < len(edited_texts) else original_text

        result_entry = {
            "prompt_idx": meta["prompt_idx"],
            "prompt_text": meta["prompt_text"],
            "category": meta["category"],
            "contributor": meta["contributor"],
            "prompt_split": meta["prompt_split"],
            "target_revision_pct": meta["target_revision_pct"],
            "doc_id": meta["doc_idx"],
            "domain": meta["domain"],
            "original_text": original_text,
            "edited_text": edited_text,
            "model_name": args.model_name,
            "status": global_status,
        }

        for col in ["generator", "label"]:
            if col in meta["row"]:
                result_entry[f"original_{col}"] = meta["row"][col]

        raw_results.append(result_entry)

    raw_df = pd.DataFrame(raw_results)
    raw_df.to_csv(output_dir / "experiment_results_raw.csv", index=False)
    print(f"  Saved {len(raw_results)} raw results to experiment_results_raw.csv")

    # ── Compute metrics ──────────────────────────────────────
    print("Computing edit metrics...")
    all_results = []
    for i, entry in enumerate(raw_results):
        metrics = compute_edit_metrics(entry["original_text"], entry["edited_text"])
        all_results.append({**entry, **metrics})
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(raw_results)} metrics computed...")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Completed {len(all_results)} edits in {total_time/60:.1f} minutes")
    print(f"{'='*60}\n")

    # ── Save results with metrics ────────────────────────────
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "experiment_results.csv", index=False)

    with open(output_dir / "experiment_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # ── Summary statistics ──────────────────────────────────
    print("\n=== Summary by Category ===")
    for cat in sorted(results_df["category"].unique()):
        cat_df = results_df[results_df["category"] == cat]
        mean_target = cat_df["target_revision_pct"].mean()
        mean_actual = cat_df["actual_edit_ratio_pct"].mean()
        print(f"  {cat:30s}  n={len(cat_df):4d}  target={mean_target:.1f}%  actual={mean_actual:.1f}%")

    print(f"\n=== Summary by Domain ===")
    for dom in sorted(results_df["domain"].unique()):
        dom_df = results_df[results_df["domain"] == dom]
        mean_actual = dom_df["actual_edit_ratio_pct"].mean()
        print(f"  {dom:20s}  n={len(dom_df):4d}  actual={mean_actual:.1f}%")

    print(f"\nResults saved to: {output_dir}")
    print(f"  - experiment_results.csv")
    print(f"  - experiment_results.json")
    print(f"  - prompts_used.json")
    print(f"  - sampled_documents.csv")
    print(f"  - config.json")


if __name__ == "__main__":
    main()
