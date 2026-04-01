"""
Step 2b: Text Generation via vLLM
==================================
Output:
    One JSONL file per model in output_dir:
    {model_tag}_text.jsonl with {"model", "prompt", "human_text", "source", "machine_text"}
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from typing import Dict, List, Set

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams


# ─────────────────────────────────────────────────────────────────────────────
# Per-model default configurations
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CONFIGS: Dict[str, Dict] = {
    # 70B class (tight on single GPU for KV cache)
    "mistralai/Mistral-Large-Instruct-2411": dict(batch_size=8, max_model_len=2048, gpu_mem=0.97),
    "meta-llama/Llama-3.3-70B-Instruct":    dict(batch_size=8, max_model_len=2048, gpu_mem=0.99),
    "Qwen/Qwen2.5-72B-Instruct":            dict(batch_size=8, max_model_len=2048, gpu_mem=0.99),
    # Mid-size
    "google/gemma-3-27b-it":                 dict(batch_size=32, max_model_len=4096, gpu_mem=0.90),
    "openai/gpt-oss-20b":                    dict(batch_size=32, max_model_len=4096, gpu_mem=0.90),
    # Small
    "meta-llama/Llama-3.1-8B-Instruct":     dict(batch_size=64, max_model_len=4096, gpu_mem=0.90),
    "google/gemma-3-12b-it":                 dict(batch_size=64, max_model_len=4096, gpu_mem=0.90),
    "Qwen/Qwen2.5-7B-Instruct":             dict(batch_size=64, max_model_len=4096, gpu_mem=0.90),
    # Fallback
    "default":                               dict(batch_size=16, max_model_len=4096, gpu_mem=0.90),
}


def get_model_cfg(model_name: str) -> Dict:
    return MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["default"])


def safe_model_id(model_name: str) -> str:
    """Turn 'org/model-name' into a filesystem-safe identifier."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)


# ─────────────────────────────────────────────────────────────────────────────
# GPU memory management
# ─────────────────────────────────────────────────────────────────────────────

_llm_cache: Dict[str, LLM] = {}


def hard_reset(sleep_sec: float = 2.0):
    """Best-effort GPU memory reset between models."""
    global _llm_cache
    for k, llm in list(_llm_cache.items()):
        try:
            if hasattr(llm, "llm_engine") and hasattr(llm.llm_engine, "shutdown"):
                llm.llm_engine.shutdown()
        except Exception:
            pass
    _llm_cache.clear()
    gc.collect()
    try:
        import torch
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass
    time.sleep(sleep_sec)


def get_llm(model_name: str, tensor_parallel_size: int, gpu_mem: float, max_model_len: int) -> LLM:
    if model_name in _llm_cache:
        return _llm_cache[model_name]
    print(f"[vLLM] Loading {model_name} | TP={tensor_parallel_size} | gpu_mem={gpu_mem} | max_len={max_model_len}")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
        enforce_eager=True,
    )
    _llm_cache[model_name] = llm
    return llm


# ─────────────────────────────────────────────────────────────────────────────
# JSONL helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_existing_prompts(jsonl_path: str) -> Set[str]:
    s = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    s.add(json.loads(line)["prompt"])
                except Exception:
                    continue
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_batch(llm: LLM, prompts: List[str], sampling: SamplingParams) -> List[str]:
    outs = llm.generate(prompts, sampling)
    return [o.outputs[0].text.strip() if o.outputs and o.outputs[0].text else "" for o in outs]


def run_one_model(
    df: pd.DataFrame,
    model_name: str,
    output_dir: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
    tensor_parallel_size: int = 1,
):
    cfg = get_model_cfg(model_name)
    batch_size = int(cfg["batch_size"])
    max_model_len = int(cfg["max_model_len"])
    gpu_mem = float(cfg["gpu_mem"])

    model_tag = safe_model_id(model_name)
    output_path = os.path.join(output_dir, f"{model_tag}_text.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    existing = load_existing_prompts(output_path)
    total = len(df)

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Config: batch={batch_size} max_len={max_model_len} gpu_mem={gpu_mem}")
    print(f"Output: {output_path}")
    print(f"Done: {len(existing)} / {total}")
    print(f"{'='*60}")

    sampling = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )

    llm = get_llm(model_name, tensor_parallel_size, gpu_mem, max_model_len)
    start_time = time.time()
    start_done = len(existing)

    with open(output_path, "a", encoding="utf-8") as f_out:
        for i in tqdm(range(0, total, batch_size), desc=model_tag):
            chunk = df.iloc[i : i + batch_size]
            chunk = chunk[~chunk["prompt"].isin(existing)]
            if chunk.empty:
                continue

            prompts = chunk["prompt"].tolist()
            try:
                gens = generate_batch(llm, prompts, sampling)
            except Exception as e:
                print(f"[Warn] Batch failed ({type(e).__name__}), falling back to single-prompt")
                gens = []
                for p in prompts:
                    try:
                        g = generate_batch(llm, [p], sampling)[0]
                    except Exception:
                        g = ""
                    gens.append(g)

            for prompt, gen_text in zip(prompts, gens):
                row = chunk[chunk["prompt"] == prompt].iloc[0]
                entry = {
                    "model": model_name,
                    "prompt": prompt,
                    "human_text": row["human_text"],
                    "source": row["source"],
                    "machine_text": gen_text,
                }
                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                existing.add(prompt)
            f_out.flush()

            elapsed = time.time() - start_time
            done = len(existing)
            rate = (done - start_done) / elapsed if elapsed > 0 else 0.0
            remaining = total - done
            eta_h = (remaining / rate) / 3600 if rate > 0 else float("inf")
            print(f"  done={done}/{total} rate={rate:.2f}/s ETA={eta_h:.1f}h")

    print(f"[Done] {model_name} -> {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate LLM text via vLLM")
    parser.add_argument("--input", required=True, help="Input CSV (prompt, human_text, source)")
    parser.add_argument("--output_dir", required=True, help="Output directory for JSONL files")
    parser.add_argument("--models", required=True, help="Comma-separated model names")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--rerun_missing", action="store_true", help="Only generate missing prompts")
    parser.add_argument("--hf_cache", default=None, help="HuggingFace cache directory")
    args = parser.parse_args()

    if args.hf_cache:
        os.environ["HF_HOME"] = args.hf_cache
        os.environ["HF_HUB_CACHE"] = args.hf_cache

    df = pd.read_csv(args.input)
    assert {"prompt", "human_text", "source"}.issubset(df.columns)
    print(f"Loaded {len(df)} prompts")

    models = [m.strip() for m in args.models.split(",")]

    for model_name in models:
        hard_reset()
        try:
            run_one_model(
                df=df,
                model_name=model_name,
                output_dir=args.output_dir,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )
        finally:
            hard_reset()

    print("\nAll models complete.")


if __name__ == "__main__":
    main()
