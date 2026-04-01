"""
Step 2a: Text Generation via HuggingFace Transformers
=====================================================
Generates LLM text for each prompt using HuggingFace Transformers
(suitable for single-model runs, e.g., Gemma-3-12B).

Usage:
    python 2_generate_hf_transformers.py \
        --model google/gemma-3-12b-it \
        --input /path/to/five_domain_prompts.csv \
        --output /path/to/Gemma3_12B_text.jsonl \
        --batch_size 32 \
        --max_new_tokens 1024

Output:
    JSONL file with one JSON object per line:
    {"prompt": "...", "human_text": "...", "source": "...", "machine_text": "..."}
"""

import argparse
import json
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Generation helpers
# ─────────────────────────────────────────────────────────────────────────────

def format_prompt_gemma(prompt_text: str) -> str:
    """Format prompt for Gemma-style chat models."""
    return f"<start_of_turn>user\n{prompt_text}\n<end_of_turn>\n<start_of_turn>model\n"


def format_prompt_generic(prompt_text: str) -> str:
    """Generic prompt formatting (no chat template)."""
    return prompt_text


def generate_batch_responses(
    prompts: list,
    model,
    tokenizer,
    generation_params: dict,
    device: str,
    batch_size: int = 8,
) -> list:
    """Generate responses for a batch of prompts."""
    all_outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_params)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_outputs.extend(zip(batch, decoded))
    return all_outputs


# ─────────────────────────────────────────────────────────────────────────────
# Resume-safe JSONL writer
# ─────────────────────────────────────────────────────────────────────────────

def save_with_generation(
    df: pd.DataFrame,
    model,
    tokenizer,
    generation_params: dict,
    device: str,
    output_path: str,
    format_fn,
    batch_size: int = 8,
):
    """Generate and save results incrementally, skipping already-done prompts."""
    existing_prompts = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing_prompts.add(json.loads(line)["prompt"])
                except Exception:
                    continue

    print(f"Already completed: {len(existing_prompts)} / {len(df)}")

    with open(output_path, "a", encoding="utf-8") as f_out:
        for i in tqdm(range(0, len(df), batch_size)):
            batch_df = df.iloc[i : i + batch_size]
            new_df = batch_df[~batch_df["prompt"].isin(existing_prompts)]
            if new_df.empty:
                continue

            raw_prompts = new_df["prompt"].tolist()
            formatted = [format_fn(p) for p in raw_prompts]
            generated_pairs = generate_batch_responses(
                formatted, model, tokenizer, generation_params, device, batch_size
            )

            for original_prompt, full_output in generated_pairs:
                try:
                    match_idx = formatted.index(original_prompt)
                    row = new_df.iloc[match_idx]
                except (ValueError, IndexError):
                    continue

                entry = {
                    "prompt": row["prompt"],
                    "human_text": row["human_text"],
                    "source": row["source"],
                    "machine_text": full_output,
                }
                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f_out.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate LLM text via HF Transformers")
    parser.add_argument("--model", required=True, help="HuggingFace model name (e.g., google/gemma-3-12b-it)")
    parser.add_argument("--input", required=True, help="Input CSV with prompt, human_text, source columns")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--hf_cache", default=None, help="HuggingFace cache directory")
    parser.add_argument("--prompt_format", choices=["gemma", "generic"], default="gemma",
                        help="Prompt formatting style")
    args = parser.parse_args()

    # Set cache dir
    if args.hf_cache:
        os.environ["HF_HOME"] = args.hf_cache

    # Load data
    df = pd.read_csv(args.input)
    assert {"prompt", "human_text", "source"}.issubset(df.columns)
    print(f"Loaded {len(df)} prompts")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.hf_cache)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        cache_dir=args.hf_cache,
    )
    print(f"Model loaded: {args.model}")

    generation_params = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": True,
        "repetition_penalty": args.repetition_penalty,
        "num_return_sequences": 1,
        "eos_token_id": tokenizer.eos_token_id,
    }

    format_fn = format_prompt_gemma if args.prompt_format == "gemma" else format_prompt_generic

    save_with_generation(
        df=df,
        model=model,
        tokenizer=tokenizer,
        generation_params=generation_params,
        device=device,
        output_path=args.output,
        format_fn=format_fn,
        batch_size=args.batch_size,
    )
    print("Done.")


if __name__ == "__main__":
    main()
