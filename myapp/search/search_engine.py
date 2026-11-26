"""
Search engine used by the web application.

The index is built once, when the application starts. A query is ranked with the score of Part 3 (BM25 plus
title match, rating, discount and stock), which was the best method in our evaluation.
"""
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

from myapp.search.algorithms import conjunctive_match, create_bm25_index, rank, score_bm25, score_custom
from myapp.search.objects import Document
from myapp.search.preprocessing import build_terms, details_text

# "under 1000", "below rs 500", "less than ₹800"... Users write budgets as words, but no product contains them,
# so with AND semantics they would empty the results. We turn them into a price filter instead
PRICE_LIMIT = re.compile(r"\b(?:under|below|less than|cheaper than|up ?to|max(?:imum)?)\s*(?:rs\.?|inr|₹)?\s*(\d[\d,]*)",
                         re.IGNORECASE)

SORT_OPTIONS = ("relevance", "price_asc", "price_desc", "rating", "discount")


@dataclass
class SearchFilters:
    """Options of the results page. max_price may also come from the query text"""
    sort: str = "relevance"
    max_price: Optional[float] = None
    min_rating: Optional[float] = None
    in_stock_only: bool = False


@dataclass
class SearchHit:
    document: Document
    score: float
    rank: int


@dataclass
class SearchResponse:
    query: str
    terms: List[str]
    hits: List[SearchHit]
    total: int
    page: int
    page_size: int
    relaxed: bool = False
    price_from_query: Optional[float] = None
    elapsed_ms: float = 0.0
    filters: SearchFilters = field(default_factory=SearchFilters)

    @property
    def pages(self):
        return max(1, -(-self.total // self.page_size))


class SearchEngine:
    """Class that implements the search engine logic"""

    def __init__(self, corpus: Dict[str, Document], cache_size: int = 256):
        self.corpus = corpus
        self.docs = {pid: self._process(doc) for pid, doc in corpus.items()}
        self.index, self.idf, self.doc_lengths, self.avg_length = create_bm25_index(list(self.docs.values()))
        self.mean_rating = float(np.mean([doc.average_rating for doc in corpus.values() if doc.average_rating is not None]))
        self._ranking = lru_cache(maxsize=cache_size)(self._rank)

    @staticmethod
    def _process(doc: Document) -> dict:
        """Fields used by the ranking functions, in the format of the processed documents of Part 1"""
        title_terms = build_terms(doc.title)
        all_terms = (title_terms + build_terms(doc.description) + build_terms(doc.brand)
                     + build_terms(f"{doc.category or ''} {doc.sub_category or ''}")
                     + build_terms(details_text(doc.product_details)))
        return {"pid": doc.pid, "title_terms": title_terms, "all_terms": all_terms,
                "average_rating": doc.average_rating, "discount": doc.discount, "out_of_stock": doc.out_of_stock}

    @staticmethod
    def parse_query(query: str) -> Tuple[List[str], Optional[float]]:
        """Query terms, and the price limit written in the query if there is one"""
        match = PRICE_LIMIT.search(query)
        price = float(match.group(1).replace(",", "")) if match else None
        text = PRICE_LIMIT.sub(" ", query) if match else query
        return build_terms(text), price

    def _partial_match(self, terms: Tuple[str, ...]) -> set:
        """Documents with all the query terms but one, used when no document has all of them"""
        unique = set(terms)
        needed = max(1, len(unique) - 1)
        counts = Counter(pid for t in unique if t in self.index for pid in self.index[t])
        return {pid for pid, count in counts.items() if count >= needed}

    def _rank(self, terms: Tuple[str, ...]) -> Tuple[Tuple[Tuple[str, float], ...], bool]:
        """Full ranking of a query. Cached, so that changing page, filter or order does not rank it again"""
        pids = conjunctive_match(list(terms), self.index)
        relaxed = False
        if not pids and len(set(terms)) > 1:
            pids, relaxed = self._partial_match(terms), True
        bm25 = score_bm25(terms, pids, self.index, self.idf, self.doc_lengths, self.avg_length)
        return tuple(rank(score_custom(terms, bm25, self.docs, self.mean_rating))), relaxed

    @staticmethod
    def _keep(doc: Document, filters: SearchFilters, max_price: Optional[float]) -> bool:
        if filters.in_stock_only and doc.out_of_stock:
            return False
        if max_price is not None and (doc.selling_price is None or doc.selling_price > max_price):
            return False
        if filters.min_rating is not None and (doc.average_rating or 0) < filters.min_rating:
            return False
        return True

    @staticmethod
    def _sort(hits: List[Tuple[Document, float]], sort: str) -> List[Tuple[Document, float]]:
        """Relevance keeps the ranking. The other orders keep it as the tie-breaker (sorted is stable)"""
        missing_last = float("inf")
        keys = {
            "price_asc": lambda h: h[0].selling_price if h[0].selling_price is not None else missing_last,
            "price_desc": lambda h: -(h[0].selling_price or 0),
            "rating": lambda h: -(h[0].average_rating or 0),
            "discount": lambda h: -(h[0].discount or 0),
        }
        return sorted(hits, key=keys[sort]) if sort in keys else hits

    def search(self, query: str, page: int = 1, page_size: int = 20,
               filters: Optional[SearchFilters] = None) -> SearchResponse:
        """
        Search the corpus.
        :param query: text written by the user
        :param page: page of results to return, starting at 1
        :param page_size: results per page
        :param filters: order and filters of the results page
        """
        start = time.perf_counter()
        filters = filters or SearchFilters()
        terms, price_from_query = self.parse_query(query)
        price_limits = [p for p in (filters.max_price, price_from_query) if p is not None]
        max_price = min(price_limits) if price_limits else None

        ranking, relaxed = self._ranking(tuple(terms)) if terms else ((), False)
        hits = [(self.corpus[pid], score) for pid, score in ranking]
        hits = [(doc, score) for doc, score in hits if self._keep(doc, filters, max_price)]
        hits = self._sort(hits, filters.sort)

        page = max(1, page)
        first = (page - 1) * page_size
        page_hits = [SearchHit(doc, score, first + i + 1) for i, (doc, score) in enumerate(hits[first:first + page_size])]
        return SearchResponse(query=query, terms=terms, hits=page_hits, total=len(hits), page=page, page_size=page_size,
                              relaxed=relaxed, price_from_query=price_from_query,
                              elapsed_ms=(time.perf_counter() - start) * 1000, filters=filters)
