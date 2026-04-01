"""
Step 3: Clean LLM Generation Artifacts
=======================================
Each LLM column has model-specific artifacts that could confound
human vs. LLM classification. This script removes them so the
classifier sees only the actual generated content.

Artifact types handled per model:
- qwen-7B:     prompt continuation at start (starts with `. ` / `, `), markdown, end notes
- gemma3-12B:  `user\\n<prompt>\\nmodel\\n` prefix, preamble ("Okay, here's..."),
               IMPORTANT DISCLAIMER blocks, placeholder citations, markdown
- gemma3-27B:  `user\\n<prompt>\\nmodel\\n` prefix, end notes, markdown
- qwen-72B:    hallucinated prompt extension at start, markdown
- llama31-8B:  prompt continuation at start, end disclaimers, markdown
- llama33-70B: full prompt echo + hallucinated instructions at start, end notes, markdown
- gpt-oss20B:  chain-of-thought reasoning, 'assistantfinal' token, chatty openers,
               "Here's a draft..." transitions, prompt-echo leakage, markdown
- gpt-oss120B: CoT + assistantfinal token, fake Wikipedia framing, fake TOC, markdown

Usage:
    python 3_clean_llm_artifacts.py \
        --input  /path/to/combined_results.csv \
        --output /path/to/combined_results_cleaned.csv

    # Verify GPT-OSS-20B cleaning quality:
    python 3_clean_llm_artifacts.py \
        --input  /path/to/combined_results.csv \
        --output /path/to/combined_results_cleaned.csv \
        --verify_gpt_oss20b
"""

import argparse
import pandas as pd
import re
import sys


# ===========================================================================
# Shared utilities
# ===========================================================================

def remove_markdown(text: str) -> str:
    """Strip markdown formatting while preserving the underlying text."""
    # Horizontal rules (allow leading/trailing whitespace)
    text = re.sub(r'^\s*-{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\*{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*_{3,}\s*$', '', text, flags=re.MULTILINE)

    # Headers: ## Title -> Title
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # Bold + italic: ***text*** or ___text___
    text = re.sub(r'\*{3}(.+?)\*{3}', r'\1', text)
    text = re.sub(r'_{3}(.+?)_{3}', r'\1', text)

    # Bold: **text** or __text__
    text = re.sub(r'\*{2}(.+?)\*{2}', r'\1', text)
    text = re.sub(r'_{2}(.+?)_{2}', r'\1', text)

    # Unpaired bold markers (no closing **): just remove the **
    text = re.sub(r'\*{2}', '', text)

    # Italic: *text* or _text_ (be careful not to strip underscores in words)
    text = re.sub(r'(?<!\w)\*([^\*\n]+?)\*(?!\w)', r'\1', text)
    text = re.sub(r'(?<!\w)_([^_\n]+?)_(?!\w)', r'\1', text)

    # Strikethrough: ~~text~~
    text = re.sub(r'~~(.+?)~~', r'\1', text)

    # Inline code: `code`
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # Bullet points: lines starting with *, -, or + followed by space
    text = re.sub(r'^[\*\-\+]\s+', '', text, flags=re.MULTILINE)

    # Numbered lists: 1. item -> item
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)

    # Markdown links: [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # Markdown images: ![alt](url) -> alt
    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'\1', text)

    # Block quotes: > text -> text
    text = re.sub(r'^>\s?', '', text, flags=re.MULTILINE)

    return text


def remove_end_notes(text: str) -> str:
    """Remove trailing disclaimers, notes, and meta-commentary."""
    end_patterns = [
        # "Note:" / "Please note:" / "Note that" blocks at end
        r'(?:\n\s*){1,3}(?:\*{0,2})(?:Please\s+)?[Nn]ote(?:\*{0,2})\s*[:.].*$',
        # "IMPORTANT DISCLAIMER" / "IMPORTANT NOTE" blocks
        r'(?:\n\s*){1,3}(?:\*{0,2})IMPORTANT\s+(?:DISCLAIMER|NOTE)S?(?:\*{0,2})\s*[:.]?.*$',
        # "Disclaimer:" blocks
        r'(?:\n\s*){1,3}(?:\*{0,2})Disclaimer(?:\*{0,2})\s*[:.].*$',
        # "(Note: ...)" parenthetical at the very end
        r'\s*\((?:Please\s+)?[Nn]ote\s*:.*?\)\s*$',
    ]
    for pat in end_patterns:
        text = re.sub(pat, '', text, flags=re.DOTALL)
    return text


def remove_wiki_framing(text: str) -> str:
    """Remove fake Wikipedia boilerplate like 'From Wikipedia' and fake TOC."""
    # "From Wikipedia, the free encyclopedia"
    text = re.sub(r'^\s*\*{0,2}From Wikipedia,?\s*the free encyclopedia\*{0,2}\s*\n*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n\s*\*{0,2}From Wikipedia,?\s*the free encyclopedia\*{0,2}\s*\n*', '\n', text, flags=re.IGNORECASE)
    # Fake "Contents" section with numbered links
    text = re.sub(
        r'(?:#{0,4}\s*)?Contents\s*\n(?:\s*\d+\s+\[?[^\]\n]+\]?\s*\n?)+',
        '', text, flags=re.IGNORECASE
    )
    # "See Also" / "References" / "External Links" sections at the end
    text = re.sub(
        r'\n\s*(?:#{0,4}\s*)?(?:See\s+Also|References|External\s+Links|Further\s+Reading|Categories)\s*\n.*$',
        '', text, flags=re.DOTALL | re.IGNORECASE
    )
    return text


def _remove_user_model_prefix(text: str, prompt: str) -> str:
    """Remove `user\\n<prompt>\\nmodel\\n` prefix from gemma-style outputs."""
    if text.startswith('user\n'):
        model_idx = text.find('\nmodel\n')
        if model_idx > 0:
            text = text[model_idx + len('\nmodel\n'):]
        else:
            model_idx = text.find('model\n')
            if model_idx > 0:
                text = text[model_idx + len('model\n'):]
    return text.strip()


def _normalize_whitespace(text: str) -> str:
    """Collapse excessive whitespace while keeping paragraph structure."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    text = text.strip()
    return text


def _strip_leading_continuation(text: str) -> str:
    """Remove prompt-continuation fragment at the start (`. `, `, `, etc.)."""
    if text and text[0] in '.,;:)':
        lines = text.split('\n')
        start_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if i == 0:
                start_idx = 1
                continue
            if stripped:
                start_idx = i
                break
        text = '\n'.join(lines[start_idx:])
    return text


# ===========================================================================
# Per-model cleaners
# ===========================================================================

def clean_qwen7b(text: str, prompt: str) -> str:
    """qwen-7B: starts with prompt continuation (`. ` / `, `)."""
    if pd.isna(text):
        return text
    text = text.strip()
    text = _strip_leading_continuation(text)
    text = remove_end_notes(text)
    text = remove_wiki_framing(text)
    text = remove_markdown(text)
    text = _normalize_whitespace(text)
    return text


def clean_gemma3_12b(text: str, prompt: str) -> str:
    """gemma3-12B: `user\\n<prompt>\\nmodel\\n<preamble><article>`."""
    if pd.isna(text):
        return text
    text = text.strip()
    text = _remove_user_model_prefix(text, prompt)

    # Remove conversational preamble
    preamble_patterns = [
        r"^(?:Okay|OK|Sure|Certainly|Here(?:'s| is| are)|I(?:'ve|'ll| will| have)).*?(?=\n\s*(?:#{1,3}\s|---|\*\*[A-Z]))",
        r"^(?:Okay|OK|Sure|Certainly),?\s+(?:here(?:'s| is| are)|let me|I(?:'ve|'ll| will)).*?\n+",
    ]
    for pat in preamble_patterns:
        text = re.sub(pat, '', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove conversational opener
    text = re.sub(
        r"^(?:Okay|OK|Sure|Certainly|Let me|Let's|Right),?\s+(?:let's\s+)?(?:tackle|delve|dive|break|look|explore|unpack|address|discuss|examine|consider|think about|get into|start with).*?[.!]\s+",
        '', text, flags=re.IGNORECASE
    )

    # Remove short exclamatory openers
    text = re.sub(
        r'^(?:Okay|OK)(?:[,!…\.]+\s*(?:okay|so|yeah|like|this|here|wow|hmm)*[,!…\.]*\s*)+\s*\n*',
        '', text, flags=re.IGNORECASE
    )

    # Remove IMPORTANT DISCLAIMER blocks
    text = re.sub(
        r'(?:\n\s*){0,3}(?:\*{0,2})IMPORTANT\s+(?:DISCLAIMER|NOTE)S?(?:\*{0,2})\s*[:.]?.*$',
        '', text, flags=re.DOTALL | re.IGNORECASE
    )

    # Remove placeholder citations
    text = re.sub(r'\[Source\s*:.*?(?:Placeholder|placeholder).*?\]', '', text)
    text = re.sub(r'\[citation\s+needed\]', '', text, flags=re.IGNORECASE)

    text = remove_end_notes(text)
    text = remove_wiki_framing(text)
    text = remove_markdown(text)
    text = _normalize_whitespace(text)
    return text


def clean_gemma3_27b(text: str, prompt: str) -> str:
    """gemma3-27B: `user\\n<prompt>\\nmodel\\n<article>`."""
    if pd.isna(text):
        return text
    text = text.strip()
    text = _remove_user_model_prefix(text, prompt)

    preamble_patterns = [
        r"^(?:Okay|OK|Sure|Certainly|Here(?:'s| is| are)|I(?:'ve|'ll| will| have)).*?(?=\n\s*(?:#{1,3}\s|---|\*\*[A-Z]))",
        r"^(?:Okay|OK|Sure|Certainly),?\s+(?:here(?:'s| is| are)|let me|I(?:'ve|'ll| will)).*?\n+",
    ]
    for pat in preamble_patterns:
        text = re.sub(pat, '', text, flags=re.DOTALL | re.IGNORECASE)

    text = re.sub(
        r"^(?:Okay|OK|Sure|Certainly|Let me|Let's|Right),?\s+(?:let's\s+)?(?:tackle|delve|dive|break|look|explore|unpack|address|discuss|examine|consider|think about|get into|start with).*?[.!]\s+",
        '', text, flags=re.IGNORECASE
    )
    text = re.sub(
        r'^(?:Okay|OK)(?:[,!…\.]+\s*(?:okay|so|yeah|like|this|here|wow|hmm)*[,!…\.]*\s*)+\s*\n*',
        '', text, flags=re.IGNORECASE
    )

    text = remove_end_notes(text)
    text = remove_wiki_framing(text)
    text = remove_markdown(text)
    text = _normalize_whitespace(text)
    return text


def clean_qwen72b(text: str, prompt: str) -> str:
    """qwen-72B: starts with hallucinated prompt extension, then article."""
    if pd.isna(text):
        return text
    text = text.strip()

    if text and text[0] in '.,;:)':
        lines = text.split('\n')
        start_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if i == 0:
                start_idx = 1
                continue
            if stripped and (stripped.startswith('#') or
                           (len(stripped) > 10 and stripped[0].isupper())):
                start_idx = i
                break
        text = '\n'.join(lines[start_idx:])

    if prompt and text.startswith(prompt[:50]):
        idx = text.find('\n', len(prompt) - 20)
        if idx > 0:
            text = text[idx:].strip()

    text = remove_end_notes(text)
    text = remove_wiki_framing(text)
    text = remove_markdown(text)
    text = _normalize_whitespace(text)
    return text


def clean_llama31_8b(text: str, prompt: str) -> str:
    """llama31-8B: prompt continuation at start, end disclaimers."""
    if pd.isna(text):
        return text
    text = text.strip()
    text = _strip_leading_continuation(text)

    # Remove "Any resemblance..." type disclaimers
    text = re.sub(
        r'\n\s*Any resemblance.*?(?:coincidental|fictional)\.?\s*$',
        '', text, flags=re.DOTALL | re.IGNORECASE
    )

    text = remove_end_notes(text)
    text = remove_wiki_framing(text)
    text = remove_markdown(text)
    text = _normalize_whitespace(text)
    return text


def clean_llama33_70b(text: str, prompt: str) -> str:
    """llama33-70B: echoes full prompt + hallucinated instructions at start."""
    if pd.isna(text):
        return text
    text = text.strip()

    if prompt:
        prompt_stripped = prompt.strip()
        if text.startswith(prompt_stripped[:min(60, len(prompt_stripped))]):
            search_start = max(0, len(prompt_stripped) - 20)
            remaining = text[search_start:]
            lines = remaining.split('\n')
            found = False
            for i, line in enumerate(lines):
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith('#'):
                    text = '\n'.join(lines[i:])
                    found = True
                    break
                if (len(stripped) > 30 and stripped[0].isupper() and
                        not stripped.lower().startswith(('write ', 'create ', 'the article'))):
                    text = '\n'.join(lines[i:])
                    found = True
                    break
            if not found:
                if text.startswith(prompt_stripped):
                    text = text[len(prompt_stripped):].strip()

    text = remove_end_notes(text)
    text = remove_wiki_framing(text)
    text = remove_markdown(text)
    text = _normalize_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# GPT-OSS shared
# ---------------------------------------------------------------------------

def _clean_gpt_oss_common(text: str) -> str:
    """Shared cleaning for gpt-oss models: remove CoT before 'assistantfinal' token."""
    if 'assistantfinal' in text:
        idx = text.index('assistantfinal')
        text = text[idx + len('assistantfinal'):]
    elif re.search(r'[.!]\s*assistant\s*(?:\*\*|#|\n)', text):
        m = re.search(r'[.!]\s*assistant\s*(?=\*\*|#|\n)', text)
        if m:
            text = text[m.end():]
    return text.strip()


# ---------------------------------------------------------------------------
# GPT-OSS-20B: thorough 15-step cleaning
# ---------------------------------------------------------------------------

# Meta-reasoning / chain-of-thought patterns that appear before the real content
_GPT_OSS20B_META_PATTERNS = [
    r'^We\s+(?:need|are|have|should|must|can|want|investigate|present|\'ll)\b.*?(?=\n\n|\n[#*]|$)',
    r'^(?:Ok|Okay|OK)[\s,]*I\s+need\s+to\s+.*?(?=\n\n|\n[#*]|$)',
    r'^The\s+user\s+(?:wants|asked|needs|request|requests|presumably|was\s+read|gave|says)\b.*?(?=\n\n|\n[#*]|$)',
    r'^(?:I\s+need\s+to|Let\s+me|I\'ll|Let\'s)\s+(?:write|create|think|start|produce|search|gather|recall|attempt|research|rewrite|craft|check|consider|proceed|look)\b.*?(?=\n\n|\n[#*]|$)',
    r'^The\s+(?:question|instructions?|content|response)\s+(?:says?|are|is|must)\b.*?(?=\n\n|\n[#*]|$)',
    r'^But\s+(?:we|it|the|I|let|then|maybe|perhaps)\b.*?(?=\n\n|\n[#*]|$)',
    r'^(?:that|and|in\s+this|which)\s+(?:look|also|is|are|we|the)\b.*?(?=\n\n|\n[#*]|$)',
    r'^Write\s+in\s+this\s+article\s+about\b.*?(?=\n\n|\n[#*]|$)',
    r'^Provide\s+(?:details|content|information)\b.*?(?=\n\n|\n[#*]|$)',
    r'^So\s+(?:we|produce|let|it)\b.*?(?=\n\n|\n[#*]|$)',
    r'^Also\s+note\s+.*?(?=\n\n|\n[#*]|$)',
    r'^This\s+seems\s+quite\b.*?(?=\n\n|\n[#*]|$)',
    r'^Assuming\s+that\s+the\b.*?(?=\n\n|\n[#*]|$)',
    r'^where\s+\S+\b.*?(?=\n\n|\n[#*]|$)',
    r'^First,\s+(?:what|let|we|I)\b.*?(?=\n\n|\n[#*]|$)',
    r'^Better\s+to\s+.*?(?=\n\n|\n[#*]|$)',
    r'^Wait\b.*?(?=\n\n|\n[#*]|$)',
    r'^Actually\b.*?(?=\n\n|\n[#*]|$)',
    r'^Alternatively\b.*?(?=\n\n|\n[#*]|$)',
    r'^I\s+(?:\'m\s+not\s+sure|know|can\s+think|recall|remember|will\s+try)\b.*?(?=\n\n|\n[#*]|$)',
    r'^There\s+(?:might|\'s\s+likely|could|was|is\s+a\s+place)\b.*?(?=\n\n|\n[#*]|$)',
    r'^Could\s+(?:be|refer)\b.*?(?=\n\n|\n[#*]|$)',
    r'^No,\s+I\b.*?(?=\n\n|\n[#*]|$)',
    r'^We\s+must\s+.*?(?=\n\n|\n[#*]|$)',
    r'^Hmm\.?\s*(?=\n|$)',
    r'^[.…?!\s]+$',
    r'^[.…?!\s]{2,}(?=\S)',
    r'^\d+\.{2,}\s*',
    r'^The\s+first\s+step\??\s+.*?(?=\n\n|\n[#*]|$)',
    r'^Let\'?s\s+open\s+search\..*?(?=\n\n|\n[#*]|$)',
    r'^assistantanalysis\b.*?(?=\n\n|\n[#*]|$)',
    r'^No\s+results\s+due\s+to\b.*?(?=\n\n|\n[#*]|$)',
    r'^Better\s+approach\b.*?(?=\n\n|\n[#*]|$)',
    r'^Hold\s+up\b.*?(?=\n\n|\n[#*]|$)',
    r'^Another\s+possibility\b.*?(?=\n\n|\n[#*]|$)',
    r'^Maybe\s+(?:it|there|he|she|the)\b.*?(?=\n\n|\n[#*]|$)',
    r'^Not\s*(?:exactly)?\.?\s*$',
    r'^I\s+remember\s+.*?(?=\n\n|\n[#*]|$)',
    r'^The\s+question\s*:.*?(?=\n\n|\n[#*]|$)',
]

_GPT_OSS20B_CHATTY_PATTERNS = [
    r'^Oh\s+(?:wow|boy|man|no|my)[\s!,]+(?:So\s+(?:like,?\s*)?)?(?:That\'s\s+like\s+.*?[.!]\s*)?',
    r'^(?:Okay|Ok|Hey|Alrighty|Alright)[\s!,]+(?:then[\s—\-]+)?(?:So\s+(?:like,?\s*)?)?(?:I\s+um[.…\s]*)?',
    r'^Hey!\s+I\s+um[.…\s]*(?:\*\*)?(?:We\'ve\s+been\s*\(\?\?\)\s*\xa0?the\s*[.…\s]*)?',
    r'^Sure\s+thing[\s!,]+(?:So\s+)?',
    r'^Well,?\s+let\'?s\s+(?:dive|get)\s+(?:in|into)[\s!.,]*',
    r'^Well,\s+',
    r'^(?:Sure|Absolutely|Certainly)[\s!,]+',
    r'^Alrighty\s+then,?\s+',
]

_GPT_OSS20B_HERES_PATTERNS = [
    r'^(?:Okay,?\s+)?[Hh]ere\'?s\s+(?:a|an|the|my)\s+(?:draft|article|example|summary|updated\s+version|short|Wikipedia|rephrased\s+version|concise|rewrite|detailed|thorough|attempt).*?(?::\s*\n?|\.?\s*\n)',
    r'^[Hh]ere\s+is\s+(?:a|an|the|my)\s*(?:draft|article|example|short|potential|fictional|rephrased|concise|Wikipedia).*?(?::\s*\n?|\.?\s*\n)',
    r'^[Hh]ere\s+is\s+(?:a|an)\s+(?:short|concise|brief).*?(?::\s*\n?|\.?\s*\n)',
    r'^I\'?m?\s+sorry\s+for\s+(?:any\s+)?confusion.*?(?::\s*\n?|\.?\s*\n)',
    r'^Sure[!,]?\s+[Hh]ere\'?s?\s+(?:a|an|the|my)?\s*(?:draft|article|example|short|rephrased|concise|rewrite).*?(?::\s*\n?|\.?\s*\n)',
    r'^Sure,?\s+here\s+is\s+(?:a|an|the)\s+(?:short|concise|brief|draft|article).*?(?::\s*\n?|\.?\s*\n)',
    r'^Let\'?s\s+(?:rewrite|craft\s+new\s+version)\s*:\s*\n?',
    r'^It\s+appears\s+you\s+want\s+a\s+rephrased\s+version.*?(?::\s*\n?|\.?\s*\n)',
    r'^[Hh]ere\'?s\s+a\s+rephrased\s+(?:abstract|version|summary)\s*:\s*\n?',
    r'^[Hh]ere\'?s\s+an?\s+attempt\s+to\s+respond\s*:\s*\n?',
    r'^[Hh]ere\'?s\s+how\s+you\s+could\s+structure.*?(?::\s*\n?|\.?\s*\n)',
    r'^[Hh]ere\s+is\s+one\s+possible\s+way\s+to\s+.*?(?::\s*\n?|\.?\s*\n)',
    r'^[Hh]ere\'?s\s+your\s+requested\s+response\s*:\s*\n?',
    r'^[Hh]ere\'?s\s*\n',
    r'^[Hh]ere\s+is\s+a\s+(?:whimsical|brief|short|quick)\s+.*?(?:\.\s*\n)',
]

_GPT_OSS20B_LEAKAGE_PATTERNS = [
    r'^The\s+content\s+is\s+to\s+.*?(?=\n\n|\n[#*]|$)',
    r'^The\s+response\s+must\s+.*?(?=\n\n|\n[#*]|$)',
    r'^and\s+that\s+is\s+not\s+so\s+much.*?(?=\n\n|\n[#*]|$)',
    r'<URL>',
    r'^,?\s*i\.\s*$',
    r'^[Ff]or\s+you\s+while\s+we\s+cannot.*?(?=\n\n|\n[#*]|$)',
    r'^and\s+(?:you\s+must|include|contain|it\s+must).*?(?=\n\n|\n[#*]|$)',
    r'^\d+\)\s+The\s+first\s+paragraph.*?(?=\n\n|\n[#*]|$)',
    r'^\[?\d+\]?\s*$',
    r'^The\s+article:\s*',
    r'^in\s+this\??\s*\[?\d*\]?\s*',
    r'^,?\s*that\s+is\s+\*?\*?\d+\*?\*?\.\s*',
]


def clean_gpt_oss20b(text: str, prompt: str) -> str:
    """
    GPT-OSS-20B: thorough 15-step cleaning.

    Handles: 'assistantfinal' token, leading bold markers, parenthetical
    meta-instructions, leading punctuation from prompt echo, multi-sentence
    chain-of-thought blocks, chatty openers, "Here's a draft..." transitions,
    prompt-echo leakage, academic labels, markdown, wiki templates, and
    whitespace normalization.
    """
    if pd.isna(text) or not isinstance(text, str) or text.strip() == '':
        return text

    # Step 1: Split on 'assistantfinal' — strongest boundary marker
    if 'assistantfinal' in text:
        text = text.split('assistantfinal', 1)[1]

    # Step 1.5: Strip leading/trailing bold markers (**) early
    text = re.sub(r'^\s*\*{2,}\s*', '', text.strip())
    text = re.sub(r'\s*\*{2,}\s*$', '', text.strip())

    # Step 2: Remove leading parenthetical meta-instructions
    text = re.sub(r'^\s*\([^)]*\)\s*\.?\s*', '', text, flags=re.DOTALL)

    # Step 3: Strip leading punctuation from prompt echo
    text = re.sub(r'^[\s]*[.?!,;:"\'\-]+\s*', '', text)
    text = re.sub(r'^\\cite\{[^}]*\}\s*', '', text.strip())
    text = re.sub(r'^\\\[.*?\\\]\s*', '', text.strip(), flags=re.DOTALL)

    # Step 4: Remove multi-sentence meta-reasoning (iterative)
    for _ in range(20):
        original = text
        for pattern in _GPT_OSS20B_META_PATTERNS:
            text = re.sub(pattern, '', text.strip(), flags=re.IGNORECASE | re.DOTALL).strip()
        text = re.sub(r'^\s*\*{2,}\s*', '', text.strip())
        text = re.sub(r'^[\s]*[.?!,;:"\'\-…]+\s*', '', text.strip())
        text = text.strip()
        if text == original:
            break

    # Step 5: Remove chatty/conversational openers (iterative)
    for _ in range(5):
        original = text
        for pattern in _GPT_OSS20B_CHATTY_PATTERNS:
            text = re.sub(pattern, '', text.strip(), flags=re.IGNORECASE).strip()
        text = re.sub(r'^\s*\*{2,}\s*', '', text.strip())
        if text == original:
            break

    # Step 6: Remove "Here's a draft..." transition phrases
    for pattern in _GPT_OSS20B_HERES_PATTERNS:
        text = re.sub(pattern, '', text.strip(), flags=re.IGNORECASE | re.DOTALL).strip()

    # Step 7: Remove remaining prompt-echo / instruction leakage
    for pattern in _GPT_OSS20B_LEAKAGE_PATTERNS:
        text = re.sub(pattern, '', text.strip(), flags=re.IGNORECASE | re.DOTALL).strip()

    # Step 8: Remove leading "Abstract:" label
    text = re.sub(r'^Abstract:\s*\n?', '', text.strip(), flags=re.IGNORECASE)

    # Step 9: Clean up "Title:" prefix
    text = re.sub(r'^Title:\s*', '', text.strip())

    # Step 10: Remove leading horizontal rules
    text = re.sub(r'^[\s]*-{3,}\s*', '', text.strip())

    # Step 11: Remove wiki-style templates {{Infobox ...}}
    text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)

    # Step 12: Remove ellipsis/garbage at start
    text = re.sub(r'^\.{2,}\s*', '', text.strip())

    # Step 13: Strip any remaining leading punctuation
    text = re.sub(r'^[\s]*[.?!,;:]+\s*', '', text.strip())

    # Steps 14-15: Shared cleanup
    text = remove_end_notes(text)
    text = remove_wiki_framing(text)
    text = remove_markdown(text)

    # Final whitespace normalization (includes literal \n -> space)
    text = re.sub(r'\\n', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = text.strip()

    return text


def clean_gpt_oss120b(text: str, prompt: str) -> str:
    """gpt-oss120B: CoT + assistantfinal token, fake Wikipedia framing, heavy markdown."""
    if pd.isna(text):
        return text
    text = text.strip()

    text = _clean_gpt_oss_common(text)

    # Remove any remaining token leaks
    text = re.sub(r'(?:assistantfinal|<\|(?:im_end|im_start|end)\|>)', '', text)

    text = remove_wiki_framing(text)
    text = remove_end_notes(text)
    text = remove_markdown(text)
    text = _normalize_whitespace(text)
    return text


# ===========================================================================
# Cleaner registry
# ===========================================================================

CLEANERS = {
    'qwen-7B_text':     clean_qwen7b,
    'gemma3-12B_text':  clean_gemma3_12b,
    'gemma3-27B_text':  clean_gemma3_27b,
    'qwen-72B_text':    clean_qwen72b,
    'llama31-8B_text':  clean_llama31_8b,
    'llama33-70B_text': clean_llama33_70b,
    'gpt-oss20B_text':  clean_gpt_oss20b,
    'gpt-oss120B_text': clean_gpt_oss120b,
}


# ===========================================================================
# GPT-OSS-20B verification
# ===========================================================================

def verify_gpt_oss20b(df: pd.DataFrame, raw_col: str = 'gpt-oss20B_text',
                      clean_col: str = 'gpt-oss20B_text') -> None:
    """Print artifact-removal diagnostics for GPT-OSS-20B."""

    def check_artifact(name, pattern):
        return df[clean_col].fillna('').apply(
            lambda x: bool(re.search(pattern, x[:300], re.IGNORECASE))
        ).sum()

    print("=== GPT-OSS-20B ARTIFACT REMOVAL VERIFICATION ===")
    checks = [
        ("'assistantfinal' remaining",   r'assistantfinal'),
        ("'We need to' at start",        r'^We need to'),
        ("'We are' at start",            r'^We are'),
        ("'Ok, I need to' at start",     r'^(?:Ok|Okay),? I need to'),
        ("'The user wants' at start",    r'^The user (?:wants|asked|needs|gave|says|request)'),
        ("'Okay!' at start",            r'^Okay!'),
        ("'Oh wow' at start",           r'^Oh wow'),
        ("'Sure!' at start",            r'^Sure[!,]'),
        ("'Here.s a draft' at start",   r'^(?:Okay,? )?Here.s a (?:draft|rephrased)'),
        ("Starts with punctuation .,?", r'^[.?,]'),
        ("'Let's' at start",           r'^Let.s (?:think|write|attempt)'),
        ("Leading '...' at start",      r'^\.{2,}'),
    ]
    for label, pat in checks:
        print(f"  {label:40s} {check_artifact(label, pat)}")

    # Summary stats
    clean_lengths = df[clean_col].fillna('').astype(str).str.len()
    empty_after = (clean_lengths == 0).sum()
    print(f"\n  Average cleaned length: {clean_lengths.mean():.0f} chars")
    print(f"  Entries that became empty: {empty_after}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Clean LLM generation artifacts")
    parser.add_argument("--input", required=True, help="Input CSV (combined_results.csv)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--verify_gpt_oss20b", action="store_true",
                        help="Run GPT-OSS-20B artifact verification after cleaning")
    args = parser.parse_args()

    print(f"Reading {args.input} ...")
    df = pd.read_csv(args.input)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    for col, cleaner in CLEANERS.items():
        if col not in df.columns:
            print(f"  WARNING: column '{col}' not found, skipping")
            continue
        print(f"  Cleaning {col} ...")
        df[col] = df.apply(lambda row: cleaner(row[col], row['prompt']), axis=1)

    print(f"Writing cleaned data to {args.output} ...")
    df.to_csv(args.output, index=False)
    print("Done.")

    if args.verify_gpt_oss20b and 'gpt-oss20B_text' in df.columns:
        print()
        verify_gpt_oss20b(df)

    # Print sample verification
    print("\n" + "=" * 70)
    print("SAMPLE VERIFICATION (first row, first 300 chars per LLM column)")
    print("=" * 70)
    for col in CLEANERS:
        if col in df.columns:
            sample = str(df[col].iloc[0])[:300]
            print(f"\n--- {col} ---")
            print(sample)
            print("...")


if __name__ == '__main__':
    main()
