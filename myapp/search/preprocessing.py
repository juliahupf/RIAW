"""
Text processing used by the web search engine (Part 4).

It follows the pipeline of Part 1, with the fixes for the problems found in Parts 2 and 3:
spelling variants are joined before tokenizing, and negative product attributes are not indexed.
"""
import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)

STOP_WORDS = frozenset(stopwords.words("english"))
_STEMMER = PorterStemmer()

# Words the shop writes in several ways. They are joined before tokenizing, so that "T-Shirt", "t shirt"
# and "tshirt" become one term instead of "shirt" for the first one (Part 2, problem 4)
COMPOUNDS = (
    (re.compile(r"\bt[\s-]?shirts?\b"), "tshirt"),
    (re.compile(r"\bsweat[\s-]shirts?\b"), "sweatshirt"),
    (re.compile(r"\btrack[\s-]pants?\b"), "trackpants"),
)

# product_details keys that identify the item but do not describe it
SKIPPED_DETAILS = frozenset({"style code", "sku", "model number", "model name", "pack of"})
# "Hooded: No" would add the term "hood" to jackets without a hood (Part 2, problem 3)
NEGATIVE_VALUES = frozenset({"no", "na", "n/a", "none", "nil", "-"})


@lru_cache(maxsize=None)
def stem(token):
    """The vocabulary is small and words repeat a lot, so caching the stemmer makes indexing much faster"""
    return _STEMMER.stem(token)


def build_terms(text):
    """Lowercase, join spelling variants, drop punctuation, stop words and single letters, and stem"""
    if not isinstance(text, str) or not text.strip():
        return []
    text = re.sub(r"https?://\S+", " ", text.lower())
    for pattern, replacement in COMPOUNDS:
        text = pattern.sub(replacement, text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [stem(t) for t in text.split() if t not in STOP_WORDS and (len(t) > 1 or t.isdigit())]


def details_text(details):
    """product_details as "key value" text, without identifiers and without attributes the product does not have"""
    return " ".join(f"{key} {value}" for key, value in (details or {}).items()
                    if key.lower() not in SKIPPED_DETAILS and str(value).strip().lower() not in NEGATIVE_VALUES)
