"""
Edit distance calculation and analysis for post-hoc filtering.
Uses character-level and word-level Levenshtein distance to measure actual revision amount.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np


@dataclass
class EditDistanceResult:
    """Results from edit distance calculation."""
    char_distance: int
    char_ratio: float  # distance / max(len(orig), len(edited))
    word_distance: int
    word_ratio: float  # distance / max(word_count_orig, word_count_edited)
    original_chars: int
    edited_chars: int
    original_words: int
    edited_words: int
    

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein distance between two strings.
    Uses rapidfuzz C implementation for speed.
    """
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.distance(s1, s2)


def tokenize_words(text: str) -> List[str]:
    """Simple word tokenization."""
    return re.findall(r'\b\w+\b', text.lower())


def calculate_edit_distance(original: str, edited: str) -> EditDistanceResult:
    """
    Calculate both character-level and word-level edit distances.
    
    Args:
        original: Original text
        edited: Edited text
        
    Returns:
        EditDistanceResult with all metrics
    """
    # Character-level
    char_dist = levenshtein_distance(original, edited)
    max_chars = max(len(original), len(edited))
    char_ratio = char_dist / max_chars if max_chars > 0 else 0.0
    
    # Word-level
    orig_words = tokenize_words(original)
    edit_words = tokenize_words(edited)
    word_dist = levenshtein_distance(' '.join(orig_words), ' '.join(edit_words))
    max_words = max(len(orig_words), len(edit_words))
    word_ratio = word_dist / max(len(' '.join(orig_words)), len(' '.join(edit_words))) if max_words > 0 else 0.0
    
    return EditDistanceResult(
        char_distance=char_dist,
        char_ratio=char_ratio,
        word_distance=word_dist,
        word_ratio=word_ratio,
        original_chars=len(original),
        edited_chars=len(edited),
        original_words=len(orig_words),
        edited_words=len(edit_words),
    )


def calculate_normalized_edit_ratio(original: str, edited: str) -> float:
    """
    Calculate a normalized edit ratio suitable for filtering.
    Returns value between 0 (no change) and 1 (completely different).
    """
    result = calculate_edit_distance(original, edited)
    # Use character ratio as primary metric
    return result.char_ratio


def filter_by_edit_distance(
    results: List[Dict],
    min_ratio: float = 0.05,
    max_ratio: float = 0.80,
    text_key: str = "original_text",
    edited_key: str = "edited_text",
) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter editing results by actual edit distance.
    
    Args:
        results: List of result dictionaries
        min_ratio: Minimum edit ratio (filter out if too little change)
        max_ratio: Maximum edit ratio (filter out if too much change)
        text_key: Key for original text in result dict
        edited_key: Key for edited text in result dict
        
    Returns:
        Tuple of (accepted_results, rejected_results)
    """
    accepted = []
    rejected = []
    
    for result in results:
        orig = result.get(text_key, "")
        edited = result.get(edited_key, "")
        
        if not orig or not edited:
            rejected.append({**result, "rejection_reason": "missing_text"})
            continue
            
        ratio = calculate_normalized_edit_ratio(orig, edited)
        result["edit_ratio"] = ratio
        
        if ratio < min_ratio:
            rejected.append({**result, "rejection_reason": f"too_little_change ({ratio:.3f} < {min_ratio})"})
        elif ratio > max_ratio:
            rejected.append({**result, "rejection_reason": f"too_much_change ({ratio:.3f} > {max_ratio})"})
        else:
            accepted.append(result)
    
    return accepted, rejected


def analyze_edit_distribution(
    results: List[Dict],
    text_key: str = "original_text",
    edited_key: str = "edited_text",
) -> Dict:
    """
    Analyze the distribution of edit distances in results.
    
    Returns statistics about the editing distribution.
    """
    ratios = []
    for result in results:
        orig = result.get(text_key, "")
        edited = result.get(edited_key, "")
        if orig and edited:
            ratios.append(calculate_normalized_edit_ratio(orig, edited))
    
    if not ratios:
        return {"error": "no_valid_pairs"}
    
    ratios = np.array(ratios)
    
    return {
        "count": len(ratios),
        "mean": float(np.mean(ratios)),
        "std": float(np.std(ratios)),
        "min": float(np.min(ratios)),
        "max": float(np.max(ratios)),
        "median": float(np.median(ratios)),
        "q25": float(np.percentile(ratios, 25)),
        "q75": float(np.percentile(ratios, 75)),
        "percentiles": {
            "p10": float(np.percentile(ratios, 10)),
            "p25": float(np.percentile(ratios, 25)),
            "p50": float(np.percentile(ratios, 50)),
            "p75": float(np.percentile(ratios, 75)),
            "p90": float(np.percentile(ratios, 90)),
        },
        "buckets": {
            "0-10%": int(np.sum((ratios >= 0) & (ratios < 0.1))),
            "10-20%": int(np.sum((ratios >= 0.1) & (ratios < 0.2))),
            "20-30%": int(np.sum((ratios >= 0.2) & (ratios < 0.3))),
            "30-40%": int(np.sum((ratios >= 0.3) & (ratios < 0.4))),
            "40-50%": int(np.sum((ratios >= 0.4) & (ratios < 0.5))),
            "50%+": int(np.sum(ratios >= 0.5)),
        }
    }


def bin_by_edit_distance(
    results: List[Dict],
    bins: List[Tuple[float, float]] = None,
    text_key: str = "original_text",
    edited_key: str = "edited_text",
) -> Dict[str, List[Dict]]:
    """
    Bin results by edit distance ranges.
    
    Args:
        results: List of result dictionaries
        bins: List of (min, max) tuples for bin ranges
        text_key: Key for original text
        edited_key: Key for edited text
        
    Returns:
        Dictionary mapping bin labels to lists of results
    """
    if bins is None:
        bins = [
            (0.0, 0.1),    # minimal edits
            (0.1, 0.2),    # light edits
            (0.2, 0.3),    # moderate edits
            (0.3, 0.5),    # substantial edits
            (0.5, 1.0),    # heavy edits
        ]
    
    binned = {f"{int(b[0]*100)}-{int(b[1]*100)}%": [] for b in bins}
    
    for result in results:
        orig = result.get(text_key, "")
        edited = result.get(edited_key, "")
        
        if not orig or not edited:
            continue
            
        ratio = calculate_normalized_edit_ratio(orig, edited)
        result["edit_ratio"] = ratio
        
        for (min_r, max_r) in bins:
            if min_r <= ratio < max_r:
                binned[f"{int(min_r*100)}-{int(max_r*100)}%"].append(result)
                break
    
    return binned
