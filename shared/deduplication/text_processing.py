"""
Text processing functions for deduplication.

Contains:
- Text normalization
- Signature computation
- Keyword extraction
- Concept extraction
- N-gram extraction
- Similarity calculations
"""

import hashlib
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import ContentItem, SimilarityScore


# Common stop words to ignore in keyword extraction
STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "also", "now", "and", "but", "or", "if", "because", "until", "while",
    "this", "that", "these", "those", "which", "who", "whom", "whose",
    "what", "it", "its", "itself", "they", "them", "their", "we", "us",
    "our", "you", "your", "he", "him", "his", "she", "her", "i", "me", "my",
    # Domain-specific common filler words
    "structure", "structures", "relationship", "relationships",
    "connection", "connections", "represents", "represent",
    "corresponds", "correspond", "maps", "map", "mapping",
    "shows", "show", "demonstrates", "demonstrate",
    "exactly", "precisely", "naturally", "inherently",
})


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    - Lowercase
    - Remove extra whitespace
    - Remove punctuation (except hyphens in words)
    - Collapse multiple spaces
    """
    text = text.lower()
    # Keep alphanumeric, spaces, and hyphens within words
    text = re.sub(r'[^\w\s-]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def compute_signature(text: str) -> str:
    """
    Compute a content signature for fast duplicate detection.

    Uses MD5 of the first 500 chars of normalized text.
    """
    normalized = normalize_text(text)[:500]
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()[:16]


def stem_word(word: str) -> str:
    """
    Simple stemming for keyword normalization.

    Handles common English suffixes and mathematical plural forms.
    """
    word = word.lower()

    # Special cases for mathematical terms
    if word == "vertices":
        return "vertex"
    if word == "indices":
        return "index"
    if word == "matrices":
        return "matrix"

    # Common suffix handling
    if word.endswith("ices") and len(word) > 5:
        return word[:-4] + "ex"
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 3:
        return word[:-1]  # Keep the 'e' (edges -> edge)
    if word.endswith("s") and len(word) > 3 and not word.endswith("ss"):
        return word[:-1]
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]

    return word


def extract_keywords(text: str) -> set[str]:
    """
    Extract meaningful keywords from text.

    Returns normalized, stemmed keywords excluding stop words.
    """
    text = text.lower()
    words = re.findall(r'\b[\w-]+\b', text)

    keywords = set()
    for word in words:
        if word in STOP_WORDS or len(word) <= 2:
            continue
        if word.isdigit():
            # Keep numbers as-is (7, 14, 168 are significant)
            keywords.add(word)
            continue

        stemmed = stem_word(word)
        if len(stemmed) > 2:
            keywords.add(stemmed)

    return keywords


def extract_concepts(text: str) -> set[str]:
    """
    Extract domain-specific mathematical/structural concepts.

    More targeted than keywords - focuses on significant terms.
    """
    # Patterns for important domain concepts
    patterns = [
        r'\b\d+\b',  # Numbers (7, 14, 168)
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns
        # Geometry terms
        r'\b(?:point|line|vertex|vertices|edge|face|graph|plane|'
        r'tiling|quartic|hyperbolic|projective|affine|euclidean)\w*\b',
        # Group theory
        r'\b(?:group|orbit|stabilizer|subgroup|automorphism|isomorphism|'
        r'symmetry|psl|order|cyclic|abelian|permutation)\w*\b',
        # Structure/relationship
        r'\b(?:incidence|duality|correspondence|bijection|homeomorphism|'
        r'isogeny|morphism|functor|embedding)\w*\b',
        # Sanskrit/music/yoga (for this specific project)
        r'\b(?:sanskrit|phonetic|grammar|sutra|sutras|swara|shruti|'
        r'raga|chakra|nadi|maheshwara)\w*\b',
        # Named mathematical objects
        r'\b(?:fano|heawood|klein|petersen|steiner|galois|mobius|'
        r'riemann|poincare|hilbert|octonion|quaternion)\b',
    ]

    concepts = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            normalized = stem_word(match.lower())
            if len(normalized) > 1:
                concepts.add(normalized)

    return concepts


def extract_ngrams(text: str, n: int = 4) -> set[str]:
    """
    Extract character n-grams for fuzzy matching.

    N-grams capture local text structure and help detect
    rearranged or slightly modified content.
    """
    normalized = normalize_text(text)
    # Remove spaces for character n-grams
    chars = normalized.replace(' ', '')

    if len(chars) < n:
        return {chars} if chars else set()

    return {chars[i:i+n] for i in range(len(chars) - n + 1)}


def jaccard_similarity(set1: set, set2: set) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def calculate_similarity(item1: "ContentItem", item2: "ContentItem") -> "SimilarityScore":
    """
    Calculate comprehensive similarity between two content items.

    Uses multiple signals for robustness.
    """
    from .models import SimilarityScore

    score = SimilarityScore()

    # 1. Signature match (exact normalized content)
    score.signature_match = item1.signature == item2.signature

    # 2. Keyword similarity
    score.keyword_similarity = jaccard_similarity(item1.keywords, item2.keywords)

    # 3. Concept similarity (more domain-specific)
    score.concept_similarity = jaccard_similarity(item1.concepts, item2.concepts)

    # 4. N-gram similarity (fuzzy match)
    score.ngram_similarity = jaccard_similarity(item1.ngrams, item2.ngrams)

    # 5. Combined weighted score
    # Weight concepts more heavily (domain-specific), n-grams less (noise)
    score.combined_score = (
        score.keyword_similarity * 0.25 +
        score.concept_similarity * 0.45 +
        score.ngram_similarity * 0.30
    )

    return score


def calculate_keyword_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity based on keyword overlap.

    Backward-compatible function for explorer module.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Jaccard similarity between 0.0 and 1.0
    """
    keywords1 = extract_keywords(text1)
    keywords2 = extract_keywords(text2)
    return jaccard_similarity(keywords1, keywords2)
