"""
Main editor module with random revision percentage generation and LLM-driven free editing.
Uses vLLM for efficient inference.
"""
from __future__ import annotations

import random
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .prompts import get_editing_prompt, list_features
from .edit_distance import calculate_edit_distance, EditDistanceResult
from .llm_interface import generate_text, generate_text_batch, generate_chat, get_batch_size_for_model


@dataclass
class RevisionConfig:
    """Configuration for revision percentage generation."""
    min_pct: int = 1      # Minimum revision percentage
    max_pct: int = 100    # Maximum revision percentage
    balanced: bool = True  # Use balanced distribution
    fixed_pct: Optional[int] = None  # If set, use this fixed percentage


def generate_revision_percentage(config: RevisionConfig, seed: Optional[int] = None) -> int:
    """
    Generate a random revision percentage with balanced distribution.
    
    If balanced=True, uses a distribution that provides good coverage across:
    - Minimal edits (1-10%)
    - Light edits (10-25%)
    - Moderate edits (25-50%)
    - Substantial edits (50-75%)
    - Heavy edits (75-100%)
    
    Args:
        config: RevisionConfig with bounds and options
        seed: Optional random seed for reproducibility
        
    Returns:
        Integer percentage (1-100 by default)
    """
    if config.fixed_pct is not None:
        return config.fixed_pct
        
    if seed is not None:
        random.seed(seed)
    
    if config.balanced:
        # Stratified sampling for balanced distribution across 1-100%
        buckets = [
            (1, 10),       # Minimal
            (10, 25),      # Light
            (25, 50),      # Moderate
            (50, 75),      # Substantial
            (75, 100),     # Heavy
        ]
        # Filter buckets to be within config bounds
        buckets = [(max(lo, config.min_pct), min(hi, config.max_pct)) 
                   for lo, hi in buckets if lo < config.max_pct and hi > config.min_pct]
        
        if buckets:
            lo, hi = random.choice(buckets)
            return random.randint(lo, hi)
    
    # Uniform distribution fallback
    return random.randint(config.min_pct, config.max_pct)


class Editor:
    """
    LLM-based text editor with free editing and post-hoc analysis.
    Uses vLLM for efficient inference.
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        Initialize the editor with a specific model.
        
        Args:
            model_name: HuggingFace model name
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        print(f"Editor initialized with model: {model_name}")
        print(f"  max_new_tokens={max_new_tokens}, temperature={temperature}, top_p={top_p}")
    
    def edit(
        self,
        text: str,
        feature: str,
        revision_pct: Optional[int] = None,
        revision_config: Optional[RevisionConfig] = None,
    ) -> Dict[str, Any]:
        """
        Edit text with LLM-driven free editing.
        
        Args:
            text: Original text to edit
            feature: Feature category to target
            revision_pct: Target revision percentage (overrides config if set)
            revision_config: Configuration for random percentage generation
            
        Returns:
            Dictionary with editing results and metrics
        """
        if not text or not text.strip():
            return {
                "original_text": text,
                "edited_text": text,
                "target_revision_pct": 0,
                "actual_edit_ratio": 0.0,
                "feature": feature,
                "status": "skipped_empty",
            }
        
        # Determine revision percentage
        if revision_pct is not None:
            target_pct = revision_pct
        else:
            config = revision_config or RevisionConfig()
            target_pct = generate_revision_percentage(config)
        
        # Get prompt
        prompt = get_editing_prompt(feature, text, target_pct)
        
        # Generate edited text
        try:
            edited_text = self._generate(prompt)
            status = "success"
        except Exception as e:
            edited_text = text
            status = f"error: {str(e)[:100]}"
        
        # Calculate actual edit distance
        edit_result = calculate_edit_distance(text, edited_text)
        
        return {
            "original_text": text,
            "edited_text": edited_text,
            "target_revision_pct": target_pct,
            "actual_edit_ratio": edit_result.char_ratio,
            "actual_edit_ratio_pct": round(edit_result.char_ratio * 100, 2),
            "edit_distance": {
                "char_distance": edit_result.char_distance,
                "char_ratio": edit_result.char_ratio,
                "word_distance": edit_result.word_distance,
                "word_ratio": edit_result.word_ratio,
                "original_chars": edit_result.original_chars,
                "edited_chars": edit_result.edited_chars,
            },
            "feature": feature,
            "model_name": self.model_name,
            "status": status,
        }
    
    def edit_batch(
        self,
        texts: List[str],
        feature: str,
        revision_pcts: Optional[List[int]] = None,
        revision_config: Optional[RevisionConfig] = None,
    ) -> List[Dict[str, Any]]:
        """
        Edit multiple texts using vLLM batch inference.
        
        Args:
            texts: List of texts to edit
            feature: Feature category to target
            revision_pcts: Optional list of revision percentages (one per text)
            revision_config: Configuration for random percentage generation
            
        Returns:
            List of result dictionaries
        """
        config = revision_config or RevisionConfig()
        
        # Generate revision percentages and prompts for all texts
        target_pcts = []
        prompts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                target_pcts.append(0)
                prompts.append(None)
            else:
                pct = revision_pcts[i] if revision_pcts else generate_revision_percentage(config)
                target_pcts.append(pct)
                prompts.append(get_editing_prompt(feature, text, pct))
                valid_indices.append(i)
        
        # Batch generate for valid prompts
        valid_prompts = [prompts[i] for i in valid_indices]
        
        if valid_prompts:
            try:
                edited_texts = generate_text_batch(
                    prompts=valid_prompts,
                    model_name=self.model_name,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
            except Exception as e:
                print(f"  Batch generation failed: {e}")
                edited_texts = [texts[i] for i in valid_indices]
        else:
            edited_texts = []
        
        # Build results
        results = []
        valid_idx = 0
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                results.append({
                    "original_text": text,
                    "edited_text": text,
                    "target_revision_pct": 0,
                    "actual_edit_ratio": 0.0,
                    "actual_edit_ratio_pct": 0.0,
                    "feature": feature,
                    "model_name": self.model_name,
                    "status": "skipped_empty",
                    "index": i,
                })
            else:
                edited = edited_texts[valid_idx] if valid_idx < len(edited_texts) else text
                valid_idx += 1
                
                edit_result = calculate_edit_distance(text, edited)
                results.append({
                    "original_text": text,
                    "edited_text": edited,
                    "target_revision_pct": target_pcts[i],
                    "actual_edit_ratio": edit_result.char_ratio,
                    "actual_edit_ratio_pct": round(edit_result.char_ratio * 100, 2),
                    "edit_distance": {
                        "char_distance": edit_result.char_distance,
                        "char_ratio": edit_result.char_ratio,
                        "word_distance": edit_result.word_distance,
                        "word_ratio": edit_result.word_ratio,
                    },
                    "feature": feature,
                    "model_name": self.model_name,
                    "status": "success",
                    "index": i,
                })
        
        return results
    
    def edit_raw(self, text: str, system_message: str, user_message: str) -> Dict[str, Any]:
        """
        Edit text using system + user messages (chat format, e.g. EditLens prompts).

        Args:
            text: Original text (for edit distance calculation)
            system_message: System prompt (role + revision % constraint)
            user_message: User prompt (EditLens instruction + original text)

        Returns:
            Dictionary with editing results and metrics
        """
        if not text or not text.strip():
            return {
                "original_text": text,
                "edited_text": text,
                "actual_edit_ratio": 0.0,
                "actual_edit_ratio_pct": 0.0,
                "status": "skipped_empty",
            }

        try:
            edited_text = generate_chat(
                system_message=system_message,
                user_message=user_message,
                model_name=self.model_name,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            status = "success"
        except Exception as e:
            edited_text = text
            status = f"error: {str(e)[:100]}"

        edit_result = calculate_edit_distance(text, edited_text)

        return {
            "original_text": text,
            "edited_text": edited_text,
            "actual_edit_ratio": edit_result.char_ratio,
            "actual_edit_ratio_pct": round(edit_result.char_ratio * 100, 2),
            "edit_distance": {
                "char_distance": edit_result.char_distance,
                "char_ratio": edit_result.char_ratio,
                "word_distance": edit_result.word_distance,
                "word_ratio": edit_result.word_ratio,
                "original_chars": edit_result.original_chars,
                "edited_chars": edit_result.edited_chars,
            },
            "model_name": self.model_name,
            "status": status,
        }

    def _generate(self, prompt: str) -> str:
        """Generate text from prompt using vLLM."""
        return generate_text(
            prompt=prompt,
            model_name=self.model_name,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )


def create_editor(
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
    **kwargs
) -> Editor:
    """Factory function to create an Editor instance."""
    return Editor(model_name=model_name, **kwargs)
