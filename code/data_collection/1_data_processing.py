"""
Step 1: Data Processing
=======================
Builds the multi-domain prompt dataset for AI-generated text detection.

Samples 1,000 prompts from each of 5 domains (Wikipedia, WikiHow, ArXiv,
Reddit, Story Generation) with stratified/balanced sampling, producing a
final CSV of 5,000 (prompt, human_text, source) rows.

Usage:
    python 1_data_processing.py \
        --m4_csv  /path/to/processed_M4.csv \
        --stories_dir /path/to/stories/ \
        --output /path/to/five_domain_prompts.csv

Inputs:
    - processed_M4.csv: The M4 dataset containing human-written texts from
      Wikipedia, WikiHow, ArXiv, and Reddit domains.
    - stories/ directory: Folder of story_*.json files for the summarization
      domain (used only if summarization is enabled).
    - HuggingFace "euclaise/writingprompts" dataset (downloaded automatically).

Output:
    - five_domain_prompts.csv: 5,000 rows with columns [prompt, human_text, source].
"""

import argparse
import json
import os
import random
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset

warnings.filterwarnings("ignore")
random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def prompt_stats(df: pd.DataFrame, source_name: str) -> None:
    """Print prompt-length statistics and optionally show a histogram."""
    temp = df[df["source"] == source_name].copy()
    temp["prompt_length"] = temp["prompt"].apply(len)
    print(f"\n--- {source_name} ---")
    print(f"  Count:  {len(temp)}")
    print(f"  Avg:    {temp['prompt_length'].mean():.0f} chars")
    print(f"  Min:    {temp['prompt_length'].min()}")
    print(f"  Max:    {temp['prompt_length'].max()}")


# ─────────────────────────────────────────────────────────────────────────────
# Domain samplers
# ─────────────────────────────────────────────────────────────────────────────

def sample_wikipedia(df: pd.DataFrame, n: int = 1000) -> pd.DataFrame:
    """Stratified sample from Wikipedia by prompt length (short/long)."""
    df["prompt_length"] = df["prompt"].apply(len)
    wiki = df[df["source"] == "wikipedia"].copy()
    median_len = wiki["prompt_length"].median()
    bins = [0, median_len, float("inf")]
    labels = ["short", "long"]
    wiki["length_bin"] = pd.cut(wiki["prompt_length"], bins=bins, labels=labels)
    sampled = wiki.groupby("length_bin", group_keys=False).apply(
        lambda x: x.sample(n=n // 2, random_state=42)
    )
    return sampled.drop(columns=["prompt_length", "length_bin"])


def sample_wikihow(df: pd.DataFrame, n: int = 1000, max_chars: int = 2000) -> pd.DataFrame:
    """Sample WikiHow prompts, filtering out very long ones."""
    wiki = df[df["source"] == "wikihow"].copy()
    wiki["prompt_length"] = wiki["prompt"].apply(len)
    wiki = wiki[wiki["prompt_length"] <= max_chars]
    sampled = wiki.sample(n=n, random_state=42)
    return sampled.drop(columns=["prompt_length"])


def sample_arxiv(df: pd.DataFrame, n: int = 1000) -> pd.DataFrame:
    """Random sample from ArXiv."""
    arxiv = df[df["source"] == "arxiv"].copy()
    return arxiv.sample(n=n, random_state=42)


def sample_reddit(df: pd.DataFrame, n: int = 1000) -> pd.DataFrame:
    """Balanced sample from Reddit across voice styles."""
    reddit = df[df["source"] == "reddit"].copy()

    def extract_voice(text):
        match = re.search(r"answer in an? (.+?) voice", text, re.IGNORECASE)
        return match.group(1).strip().lower() if match else None

    reddit["voice_style"] = reddit["prompt"].apply(extract_voice)
    styles = reddit["voice_style"].dropna().unique()

    expert_count = reddit[reddit["voice_style"] == "expert confident"].shape[0]
    remaining = n - expert_count
    per_style = remaining // (len(styles) - 1)

    parts = [reddit[reddit["voice_style"] == "expert confident"]]
    for style in styles:
        if style != "expert confident":
            subset = reddit[reddit["voice_style"] == style]
            parts.append(subset.sample(n=min(per_style, len(subset)), random_state=42))

    # Top up to exactly n if needed
    balanced = pd.concat(parts)
    if len(balanced) < n:
        extra = reddit[~reddit.index.isin(balanced.index)].sample(
            n=n - len(balanced), random_state=42
        )
        balanced = pd.concat([balanced, extra])

    return balanced.drop(columns=["voice_style"], errors="ignore")


def sample_story_generation(n: int = 1000, cache_dir: str = None) -> pd.DataFrame:
    """Sample from the WritingPrompts dataset on HuggingFace."""
    dataset = load_dataset("euclaise/writingprompts", cache_dir=cache_dir)
    wp = dataset["train"].to_pandas()
    wp = wp[wp["prompt"].str.startswith("[ WP ]")]

    min_char_length = 70
    wp = wp[wp["prompt"].str.len() >= min_char_length]
    sampled = wp.sample(n=n, random_state=42)

    sampled = sampled.rename(columns={"story": "human_text"})
    sampled["prompt"] = sampled["prompt"].str.replace(r"^\[\s*WP\s*\]\s*", "", regex=True)
    sampled["prompt"] = "Generate a story based on this scenario: " + sampled["prompt"]
    sampled["source"] = "story_generation"
    return sampled[["prompt", "human_text", "source"]]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build multi-domain prompt dataset")
    parser.add_argument("--m4_csv", required=True, help="Path to processed_M4.csv")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--hf_cache", default=None, help="HuggingFace cache directory")
    args = parser.parse_args()

    # Load M4 dataset
    print("Loading M4 dataset...")
    df = pd.read_csv(args.m4_csv)
    drop_cols = [c for c in ["davinci_text", "chatgpt_text", "cohere_text", "dolly_text"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Add prompt_length for sampling
    df["prompt_length"] = df["prompt"].apply(len)

    # Print domain stats
    for source in df["source"].unique():
        prompt_stats(df, source)

    # Sample each domain
    print("\nSampling domains...")
    wikipedia_df = sample_wikipedia(df, n=1000)
    wikihow_df = sample_wikihow(df, n=1000)
    arxiv_df = sample_arxiv(df, n=1000)
    reddit_df = sample_reddit(df, n=1000)
    story_df = sample_story_generation(n=1000, cache_dir=args.hf_cache)

    # Drop prompt_length from M4-derived frames
    for frame in [wikipedia_df, wikihow_df, arxiv_df, reddit_df]:
        if "prompt_length" in frame.columns:
            frame.drop(columns=["prompt_length"], inplace=True)

    # Combine
    final = pd.concat([wikipedia_df, wikihow_df, arxiv_df, reddit_df, story_df])
    final = final.reset_index(drop=True)

    print(f"\nFinal dataset: {final.shape}")
    print(final["source"].value_counts())

    # Save
    final.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
