import math
import re
import zlib
from collections import Counter, defaultdict

import nltk
import numpy as np
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
    return 1 + math.log2(tf)


def create_tfidf_index(docs, field="all_terms"):
    """
    Precompute everything needed to score a query with TF-IDF and cosine similarity
    :return: (inverted index, idf of each term, length of each document vector)
    """
    index = create_inverted_index(docs, field)
    num_docs = len(docs)
    idf = {term: math.log2(num_docs / len(postings)) for term, postings in index.items()}

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


# ---------------------------------------------------------------- Part 3

def rank(scores):
    """Best document first. Ties are broken by pid so runs are reproducible"""
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def create_bm25_index(docs, field="all_terms"):
    """
    Precompute what BM25 needs: the inverted index, the BM25 idf, the length of each document and the average length
    """
    index = create_inverted_index(docs, field)
    num_docs = len(docs)
    bm25_idf = {term: math.log(1 + (num_docs - len(postings) + 0.5) / (len(postings) + 0.5))
                for term, postings in index.items()}
    doc_lengths = {doc["pid"]: len(doc[field]) for doc in docs}
    avg_length = sum(doc_lengths.values()) / num_docs
    return index, bm25_idf, doc_lengths, avg_length


def score_bm25(query_terms, pids, index, bm25_idf, doc_lengths, avg_length, k1=1.2, b=0.75):
    """BM25 score of each document in pids. Each query term is counted once"""
    scores = {}
    for pid in pids:
        length_norm = k1 * (1 - b + b * doc_lengths[pid] / avg_length)
        scores[pid] = sum(bm25_idf[t] * index[t][pid] * (k1 + 1) / (index[t][pid] + length_norm)
                          for t in set(query_terms) if pid in index.get(t, {}))
    return scores


def score_custom(query_terms, bm25_scores, doc_store, mean_rating, weights=(0.5, 0.3, 0.15, 0.05), stock_penalty=0.5):
    """
    Our score: text relevance (BM25), query terms found in the title, rating and discount.
    Out-of-stock products keep only stock_penalty of their score
    :param bm25_scores: {pid: BM25 score} of the documents that passed the AND filter
    :param mean_rating: rating used when a product has none, so that unrated products are neither rewarded nor punished
    """
    w_text, w_title, w_rating, w_discount = weights
    max_bm25 = max(bm25_scores.values(), default=0) or 1
    unique_terms = set(query_terms)
    scores = {}
    for pid, bm25 in bm25_scores.items():
        doc = doc_store[pid]
        title_match = len(unique_terms & set(doc["title_terms"])) / len(unique_terms)
        rating = doc["average_rating"] if doc["average_rating"] is not None else mean_rating
        discount = doc["discount"] or 0
        score = (w_text * bm25 / max_bm25 + w_title * title_match
                 + w_rating * (rating - 1) / 4 + w_discount * discount / 100)
        scores[pid] = score * stock_penalty if doc["out_of_stock"] else score
    return scores


def _stable_hash(text):
    """Python's hash() changes between runs; word2vec uses this one so that its vectors are reproducible"""
    return zlib.crc32(text.encode("utf-8"))


def train_word2vec(docs, field="all_terms", vector_size=100, seed=0):
    """Skip-gram word2vec trained on the processed terms of the collection"""
    from gensim.models import Word2Vec   # imported here: Part 2 does not need gensim
    return Word2Vec(sentences=[doc[field] for doc in docs], vector_size=vector_size, window=5, min_count=2,
                    sg=1, epochs=5, workers=1, seed=seed, hashfxn=_stable_hash)


def text_vector(terms, model):
    """Average of the word vectors of the terms; terms outside the vocabulary are skipped"""
    vectors = [model.wv[t] for t in terms if t in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)


def score_word2vec(query_terms, pids, doc_vectors, model):
    """Cosine similarity between the averaged word2vec vector of the query and of each document"""
    query_vector = text_vector(query_terms, model)
    query_norm = np.linalg.norm(query_vector)
    scores = {}
    for pid in pids:
        norm = query_norm * np.linalg.norm(doc_vectors[pid])
        scores[pid] = float(query_vector @ doc_vectors[pid] / norm) if norm else 0.0
    return scores
