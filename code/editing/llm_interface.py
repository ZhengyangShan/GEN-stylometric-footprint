"""
LLM interface using vLLM for efficient inference.
Adapted from editing/llm_interface.py with support for free-form text generation.
"""
from __future__ import annotations

import os
from typing import Optional

# Set cache directories BEFORE vLLM import
HF_CACHE_DIR = os.getenv("HF_HOME", "/projectnb/tin-lab/yukyung/emnlp-rebuttal/Models")
VLLM_CACHE_DIR = os.getenv("VLLM_CACHE_DIR", "/projectnb/tin-lab/yukyung/emnlp-rebuttal/vllm_cache")

# Environment setup
os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_DIR)
os.environ.setdefault("VLLM_CACHE_DIR", VLLM_CACHE_DIR)
os.environ.setdefault("XDG_CACHE_HOME", VLLM_CACHE_DIR)
os.environ.setdefault("TORCH_COMPILE_CACHE_DIR", f"{VLLM_CACHE_DIR}/torch_compile")

from vllm import LLM, SamplingParams


# Global LLM instance cache to avoid reloading models
_llm_cache = {}

# Model-specific batch sizes (optimized for 48GB × 4 GPUs)
MODEL_BATCH_SIZES = {
    # Very large models (100B+): Very small batch
    "openai/gpt-oss-120b": 4,
    "mistralai/Mistral-Large-Instruct-2411": 4,
    
    # Large models (70B): batch 32 with max_model_len=4096, max_new_tokens=1024
    "meta-llama/Llama-3.1-70B-Instruct": 32,
    "meta-llama/Llama-3.3-70B-Instruct": 32,
    "Qwen/Qwen2.5-72B-Instruct": 32,
    
    # Medium models (20-27B): Medium batch
    "openai/gpt-oss-20b": 16,
    "google/gemma-3-27b-it": 16,
    
    # Small models (7-12B): Large batch
    "google/gemma-3-12b-it": 32,
    "meta-llama/Llama-3.1-8B-Instruct": 32,
    "Qwen/Qwen2.5-7B-Instruct": 32,
    "mistralai/Mistral-7B-Instruct-v0.3": 32,
    
    # Default fallback
    "default": 8
}


def get_batch_size_for_model(model_name: str) -> int:
    """Get optimal batch size for a given model."""
    return MODEL_BATCH_SIZES.get(model_name, MODEL_BATCH_SIZES["default"])


def get_llm(model_name: str, max_model_len: int = 4096) -> LLM:
    """Get or create LLM instance (cached to avoid reloading)."""
    global _llm_cache
    
    if model_name not in _llm_cache:
        # Get settings from environment
        tensor_parallel_size = int(os.getenv("TENSOR_PARALLEL_SIZE", "4"))
        gpu_memory_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.85"))
        max_model_len = int(os.getenv("MAX_MODEL_LEN", "4096"))  # Limit context to save memory
        
        # Some models cannot use high TP (KV heads not divisible)
        incompatible_models = ['phi-4']
        if any(m in model_name for m in incompatible_models) and tensor_parallel_size > 1:
            print(f"  Warning: {model_name} incompatible with TP={tensor_parallel_size}, forcing TP=1")
            tensor_parallel_size = 1
        
        print(f"  Loading model {model_name} with TP={tensor_parallel_size}, max_model_len={max_model_len}...")
        _llm_cache[model_name] = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
        )
    else:
        print(f"  Reusing cached model {model_name}")
    
    return _llm_cache[model_name]


def generate_text(
    prompt: str,
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
    max_new_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate text using vLLM (single prompt, raw string)."""
    results = generate_text_batch(
        prompts=[prompt],
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return results[0]


def generate_text_batch(
    prompts: list[str],
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
    max_new_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> list[str]:
    """Generate text using vLLM for batch inference (raw string prompts)."""
    
    llm = get_llm(model_name)
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )
    
    # Batch generate
    outputs = llm.generate(prompts, sampling_params)
    
    if not outputs or len(outputs) == 0:
        raise RuntimeError("vLLM returned no outputs")
    
    # Extract generated text
    results = []
    for output in outputs:
        gen_text = output.outputs[0].text.strip()
        results.append(gen_text)
    
    return results


def generate_chat(
    system_message: str,
    user_message: str,
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
    max_new_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate text using vLLM chat interface (system + user messages)."""
    results = generate_chat_batch(
        conversations=[[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]],
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return results[0]


def generate_chat_batch(
    conversations: list[list[dict]],
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
    max_new_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> list[str]:
    """
    Generate text using vLLM chat interface for batch inference.

    Args:
        conversations: List of conversations, each a list of
                       {"role": "system"|"user", "content": "..."} dicts.
    """
    llm = get_llm(model_name)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )

    outputs = llm.chat(conversations, sampling_params)

    if not outputs or len(outputs) == 0:
        raise RuntimeError("vLLM returned no outputs")

    results = []
    for output in outputs:
        gen_text = output.outputs[0].text.strip()
        results.append(gen_text)

    return results
