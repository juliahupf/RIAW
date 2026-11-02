import numpy as np


def rank_labels(y_true, y_score):
    """Relevance labels sorted by predicted score, best first. The sort is stable, so ties keep their input order"""
    order = np.argsort(-np.asarray(y_score, dtype=float), kind="stable")
    return np.take(np.asarray(y_true), order)


def precision_at_k(y_true, y_score, k=10):
    ranked = rank_labels(y_true, y_score)[:k]
    return np.sum(ranked) / k


def recall_at_k(y_true, y_score, k=10, total_relevant=None):
    """total_relevant counts the relevant documents the system did not retrieve, which y_true cannot contain"""
    ranked = rank_labels(y_true, y_score)[:k]
    total = np.sum(y_true) if total_relevant is None else total_relevant
    return np.sum(ranked) / total if total else 0.0


def f1_at_k(y_true, y_score, k=10, total_relevant=None):
    precision = precision_at_k(y_true, y_score, k)
    recall = recall_at_k(y_true, y_score, k, total_relevant)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def avg_precision_at_k(y_true, y_score, k=10):
    """Practice Session 2 definition: normalised by the relevant documents found within the top k"""
    ranked = rank_labels(y_true, y_score)[:k]
    if np.sum(ranked) == 0:
        return 0.0
    precisions = np.cumsum(ranked) / (np.arange(len(ranked)) + 1)
    return np.sum(precisions * ranked) / np.sum(ranked)


def map_at_k(rankings, k=10):
    """rankings: list with one (y_true, y_score) pair per query"""
    return np.mean([avg_precision_at_k(y_true, y_score, k) for y_true, y_score in rankings])


def rr_at_k(y_true, y_score, k=10):
    ranked = rank_labels(y_true, y_score)[:k]
    if np.sum(ranked) == 0:
        return 0.0
    return 1 / (np.argmax(ranked) + 1)


def mrr_at_k(rankings, k=10):
    return np.mean([rr_at_k(y_true, y_score, k) for y_true, y_score in rankings])


def dcg_at_k(y_true, y_score, k=10):
    ranked = rank_labels(y_true, y_score)[:k]
    gain = 2 ** ranked - 1
    discounts = np.log2(np.arange(len(ranked)) + 2)
    return np.sum(gain / discounts)


def ndcg_at_k(y_true, y_score, k=10, ideal_labels=None):
    """ideal_labels are all the judged labels of the query, in case some of them were not retrieved"""
    ideal = y_true if ideal_labels is None else ideal_labels
    dcg_max = dcg_at_k(ideal, ideal, k)
    return dcg_at_k(y_true, y_score, k) / dcg_max if dcg_max else 0.0
