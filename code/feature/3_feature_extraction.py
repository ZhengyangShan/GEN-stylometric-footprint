import nltk
import numpy as np
import math
import spacy
import string
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import textstat

# Download necessary NLTK resources (run once in a fresh env)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Initialize stopwords and punctuation
stop_words = set(stopwords.words("english"))
punctuation_set = set(string.punctuation)

VOWELS = set("aeiou")
CONSONANTS = set("bcdfghjklmnpqrstvwxyz")

# Load SpaCy for syntactic analysis (dependency parse)
nlp = spacy.load("en_core_web_sm")


# -----------------------------
# Core features
# -----------------------------

def lexical_diversity(text: str) -> float:
    # Unique Words / Total Words (use lowercased tokens for "word types")
    tokens = word_tokenize(text)
    if not tokens:
        return 0.0
    tokens_lower = [t.lower() for t in tokens]
    return len(set(tokens_lower)) / len(tokens_lower)

def lexical_density(text: str) -> float:
    # Content Words / Total Words
    # Proxy: alphabetic tokens that are NOT stopwords
    tokens = word_tokenize(text)
    if not tokens:
        return 0.0
    tokens_lower = [t.lower() for t in tokens]
    content = [w for w in tokens_lower if w.isalpha() and w not in stop_words]
    return len(content) / len(tokens_lower)

def percent_long_words(text: str) -> float:
    # Words with >6 characters / Total Words
    tokens = word_tokenize(text)
    if not tokens:
        return 0.0
    tokens_lower = [t.lower() for t in tokens]
    long_words = [w for w in tokens_lower if w.isalpha() and len(w) > 6]
    return len(long_words) / len(tokens_lower)

def entropy(text: str) -> float:
    # -sum p(w) log2 p(w) over token frequencies (use lowercased tokens)
    tokens = word_tokenize(text)
    if not tokens:
        return 0.0
    tokens_lower = [t.lower() for t in tokens]
    freq = Counter(tokens_lower)
    total = len(tokens_lower)
    return -sum((c/total) * math.log2(c/total) for c in freq.values())

def burstiness(text: str) -> float:
    # std(word_freqs) / mean(word_freqs)
    tokens = word_tokenize(text)
    if not tokens:
        return 0.0
    tokens_lower = [t.lower() for t in tokens]
    freq_vals = np.array(list(Counter(tokens_lower).values()), dtype=float)
    if freq_vals.size == 0:
        return 0.0
    mu = float(freq_vals.mean())
    sigma = float(freq_vals.std())
    return (sigma / mu) if mu > 0 else 0.0

def percent_vowels(text: str) -> float:
    # vowel characters / total characters
    if not text:
        return 0.0
    t = text.lower()
    return sum(1 for ch in t if ch in VOWELS) / len(t)

def percent_consonants(text: str) -> float:
    # consonant characters / total characters
    if not text:
        return 0.0
    t = text.lower()
    return sum(1 for ch in t if ch in CONSONANTS) / len(t)

def percent_punctuation(text: str) -> float:
    # punctuation characters / total characters
    if not text:
        return 0.0
    return sum(1 for ch in text if ch in punctuation_set) / len(text)

def num_words(text: str) -> int:
    # total word count (tokens)
    return len(word_tokenize(text))

def num_sentences(text: str) -> int:
    # total sentence count
    return len(sent_tokenize(text))

def avg_sentence_length(text: str) -> float:
    # total words / total sentences
    s = num_sentences(text)
    return (num_words(text) / s) if s > 0 else 0.0

def parse_tree_depth(text: str) -> float:
    # max # of ancestors in dependency tree.
    doc = nlp(text)
    if len(doc) == 0:
        return 0.0
    return float(max((len(list(tok.ancestors)) for tok in doc), default=0))

def gunning_fog(text: str) -> float:
    # Use textstat's implementation
    if not text:
        return 0.0
    try:
        return float(textstat.gunning_fog(text))
    except Exception:
        return 0.0

def linsear_write(text: str) -> float:
    # Use textstat's implementation
    if not text:
        return 0.0
    try:
        return float(textstat.linsear_write_formula(text))
    except Exception:
        return 0.0


# -----------------------------
# MAIN: 14 core features
# -----------------------------
def extract_all_features(text: str) -> dict:
    text = text or ""
    return {
        "lexical_diversity": lexical_diversity(text),
        "lexical_density": lexical_density(text),
        "percent_long_words": percent_long_words(text),
        "entropy": entropy(text),
        "burstiness": burstiness(text),
        "percent_vowels": percent_vowels(text),
        "percent_consonants": percent_consonants(text),
        "percent_punctuation": percent_punctuation(text),
        "num_words": int(num_words(text)),
        "num_sentences": int(num_sentences(text)),
        "avg_sentence_length": avg_sentence_length(text),
        "parse_tree_depth": parse_tree_depth(text),
        "gunning_fog": gunning_fog(text),
        "linsear_write": linsear_write(text),
    }
