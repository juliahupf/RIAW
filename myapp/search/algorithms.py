import math
import re
from collections import Counter, defaultdict

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


def build_terms(text):
    """Same pipeline used for the documents in Part 1, so queries and documents share their terms"""
    if not isinstance(text, str) or not text.strip():
        return []
    text = re.sub(r"https?://\S+", " ", text.lower())
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOP_WORDS]
    tokens = [t for t in tokens if len(t) > 1 or t.isdigit()]
    return [STEMMER.stem(t) for t in tokens]


def create_inverted_index(docs, field="all_terms"):
    """
    Build the inverted index from the processed documents
    :param docs: list of processed documents (Part 1 output)
    :param field: field holding the terms to index
    :return: {term: {pid: term frequency in the document}}
    """
    index = defaultdict(dict)
    for doc in docs:
        for term, tf in Counter(doc[field]).items():
            index[term][doc["pid"]] = tf
    return dict(index)


def tf_weight(tf):
    return 1 + math.log10(tf)


def create_tfidf_index(docs, field="all_terms"):
    """
    Precompute everything needed to score a query with TF-IDF and cosine similarity
    :return: (inverted index, idf of each term, length of each document vector)
    """
    index = create_inverted_index(docs, field)
    num_docs = len(docs)
    idf = {term: math.log10(num_docs / len(postings)) for term, postings in index.items()}

    squared_norms = defaultdict(float)
    for term, postings in index.items():
        for pid, tf in postings.items():
            squared_norms[pid] += (tf_weight(tf) * idf[term]) ** 2
    doc_norms = {pid: math.sqrt(value) for pid, value in squared_norms.items()}
    return index, idf, doc_norms


def conjunctive_match(query_terms, index):
    """Documents containing all the query terms (AND). A term outside the vocabulary matches nothing"""
    if not query_terms or any(term not in index for term in query_terms):
        return set()
    postings = sorted((index[term] for term in set(query_terms)), key=len)  # shortest first
    matches = set(postings[0])
    for posting in postings[1:]:
        matches &= posting.keys()
    return matches


def score_tfidf(query_terms, pids, index, idf, doc_norms):
    """Cosine similarity between the TF-IDF vector of the query and each document in pids"""
    query_weights = {t: tf_weight(tf) * idf.get(t, 0) for t, tf in Counter(query_terms).items()}
    query_norm = math.sqrt(sum(w ** 2 for w in query_weights.values()))
    scores = {}
    for pid in pids:
        dot = sum(w * tf_weight(index[t][pid]) * idf[t] for t, w in query_weights.items() if pid in index.get(t, {}))
        scores[pid] = dot / (query_norm * doc_norms[pid]) if query_norm and doc_norms[pid] else 0.0
    return scores


def search_in_corpus(query, index, idf, doc_norms):
    """
    Conjunctive search ranked by TF-IDF
    :return: list of (pid, score) with the best document first. Ties are broken by pid so runs are reproducible
    """
    query_terms = build_terms(query)
    scores = score_tfidf(query_terms, conjunctive_match(query_terms, index), index, idf, doc_norms)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
