"""
In-memory web analytics, organised as a star schema.

Fact tables:   RequestFact (every HTTP request), QueryFact (every search), ClickFact (every click on a result)
Dimensions:    SessionDim (physical session, with the visitor context), VisitorDim (browser cookie)
               Documents are the corpus itself, joined by pid. Time is stored as datetimes in the facts.

Data lives in memory, as the statement allows. export() returns every table as plain dictionaries, which can be
saved as JSON to keep or inspect the data.
"""
import threading
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from statistics import mean, median
from typing import Callable, Dict, List, Optional

import httpagentparser

SESSION_TIMEOUT = timedelta(minutes=30)   # usual definition of a physical session (sit-down)
MISSION_GAP = timedelta(minutes=10)       # a query after a long pause starts a new goal even if it shares words


@dataclass
class VisitorDim:
    visitor_id: str
    first_seen: datetime
    sessions: int = 0


@dataclass
class SessionDim:
    session_id: str
    visitor_id: str
    started_at: datetime
    last_seen: datetime
    ip: str
    browser: str
    browser_version: str
    os: str
    device: str
    user_agent: str
    requests: int = 0
    queries: int = 0
    clicks: int = 0
    missions: int = 0


@dataclass
class RequestFact:
    request_id: int
    session_id: str
    timestamp: datetime
    method: str
    path: str
    endpoint: str
    status: int
    latency_ms: float
    referrer: Optional[str]


@dataclass
class QueryFact:
    query_id: int
    session_id: str
    mission_id: str
    timestamp: datetime
    text: str
    terms: List[str]
    n_terms: int
    order_in_session: int
    n_results: int
    relaxed: bool
    elapsed_ms: float
    price_limit: Optional[float] = None
    filters: Dict[str, object] = field(default_factory=dict)


@dataclass
class ClickFact:
    click_id: int
    session_id: str
    query_id: Optional[int]
    pid: str
    rank: Optional[int]
    timestamp: datetime
    kind: str = "details"            # details: our document page | external: link to the original shop
    dwell_seconds: Optional[float] = None


def parse_user_agent(user_agent: str) -> dict:
    detected = httpagentparser.detect(user_agent or "")
    browser = detected.get("browser", {}) or {}
    platform = detected.get("os", {}) or detected.get("platform", {}) or {}
    os_name = platform.get("name") or "Unknown"
    if "Android" in (user_agent or ""):
        os_name = "Android"          # httpagentparser reports Android phones as "Linux"
    mobile = any(word in (user_agent or "") for word in ("Mobile", "Android", "iPhone"))
    tablet = "iPad" in (user_agent or "") or ("Android" in (user_agent or "") and "Mobile" not in (user_agent or ""))
    device = "Tablet" if tablet else "Mobile" if mobile else "Desktop"
    if detected.get("bot"):
        device = "Bot"
    readable = {"ChromiumEdge": "Edge", "Microsoft Edge": "Edge", "Macintosh": "macOS"}
    browser_name = browser.get("name") or "Unknown"
    return {"browser": readable.get(browser_name, browser_name), "browser_version": browser.get("version") or "",
            "os": readable.get(os_name, os_name), "device": device}


class AnalyticsData:
    """Collects the usage of the website. Thread safe, since Flask may serve requests in parallel"""

    def __init__(self, clock: Callable[[], datetime] = datetime.now):
        self.clock = clock
        self._lock = threading.Lock()
        self.visitors: Dict[str, VisitorDim] = {}
        self.sessions: Dict[str, SessionDim] = {}
        self.requests: List[RequestFact] = []
        self.queries: Dict[int, QueryFact] = {}
        self.clicks: List[ClickFact] = []
        self._open_click: Dict[str, ClickFact] = {}     # last click of each session waiting for its dwell time

    # ------------------------------------------------------------------ collection

    def session_for(self, visitor_id: Optional[str], session_id: Optional[str], ip: str, user_agent: str):
        """
        Return (visitor_id, session_id) for a request, creating them when needed.
        A session ends after SESSION_TIMEOUT without requests: the next request starts a new one.
        """
        now = self.clock()
        with self._lock:
            if not visitor_id or visitor_id not in self.visitors:
                visitor_id = visitor_id or uuid.uuid4().hex
                self.visitors[visitor_id] = VisitorDim(visitor_id, now)
            session = self.sessions.get(session_id) if session_id else None
            if session is None or session.visitor_id != visitor_id or now - session.last_seen > SESSION_TIMEOUT:
                context = parse_user_agent(user_agent)
                session = SessionDim(uuid.uuid4().hex, visitor_id, now, now, ip, user_agent=user_agent or "", **context)
                self.sessions[session.session_id] = session
                self.visitors[visitor_id].sessions += 1
            session.last_seen = now
            return visitor_id, session.session_id

    def log_request(self, session_id: str, method: str, path: str, endpoint: str, status: int, latency_ms: float,
                    referrer: Optional[str]):
        now = self.clock()
        with self._lock:
            self.requests.append(RequestFact(len(self.requests) + 1, session_id, now, method, path, endpoint or "",
                                             status, round(latency_ms, 2), referrer))
            if session_id in self.sessions:
                self.sessions[session_id].requests += 1

    def close_dwell(self, session_id: str):
        """
        Called when the user comes back to the results page: the dwell time of the last clicked document is the time
        between the click and this moment
        """
        now = self.clock()
        with self._lock:
            click = self._open_click.pop(session_id, None)
            if click is not None:
                click.dwell_seconds = round((now - click.timestamp).total_seconds(), 1)

    def log_query(self, session_id: str, text: str, terms: List[str], n_results: int, relaxed: bool,
                  elapsed_ms: float, price_limit: Optional[float] = None, filters: Optional[dict] = None) -> int:
        now = self.clock()
        with self._lock:
            session = self.sessions[session_id]
            previous = [q for q in self.queries.values() if q.session_id == session_id]
            last = previous[-1] if previous else None
            # A new mission starts when the query shares no term with the previous one, or after a long pause
            same_goal = last is not None and set(terms) & set(last.terms) and now - last.timestamp <= MISSION_GAP
            if same_goal:
                mission_id = last.mission_id
            else:
                session.missions += 1
                mission_id = f"{session_id[:8]}-{session.missions}"
            query_id = len(self.queries) + 1
            self.queries[query_id] = QueryFact(query_id, session_id, mission_id, now, text, list(terms), len(terms),
                                               len(previous) + 1, n_results, relaxed, round(elapsed_ms, 2),
                                               price_limit, dict(filters or {}))
            session.queries += 1
            return query_id

    def log_click(self, session_id: str, pid: str, query_id: Optional[int], rank: Optional[int],
                  kind: str = "details"):
        now = self.clock()
        with self._lock:
            if query_id is not None and (query_id not in self.queries or self.queries[query_id].session_id != session_id):
                query_id = None      # ignore query ids that do not belong to this session (old or forged links)
            click = ClickFact(len(self.clicks) + 1, session_id, query_id, pid, rank, now, kind)
            self.clicks.append(click)
            if kind == "details":
                self._open_click[session_id] = click
            if session_id in self.sessions:
                self.sessions[session_id].clicks += 1

    def query(self, query_id: Optional[int]) -> Optional[QueryFact]:
        return self.queries.get(query_id) if query_id is not None else None

    # ------------------------------------------------------------------ reports

    def summary(self, corpus: Optional[dict] = None, top: int = 10) -> dict:
        """Key indicators and the data of every chart of the dashboard"""
        with self._lock:
            sessions = list(self.sessions.values())
            queries = list(self.queries.values())
            clicks = list(self.clicks)
            requests = list(self.requests)

        doc_clicks = [c for c in clicks if c.kind == "details"]
        clicked_queries = {c.query_id for c in clicks if c.query_id is not None}
        first_click_rank = {}
        for c in sorted(doc_clicks, key=lambda c: c.timestamp):
            if c.query_id is not None and c.rank and c.query_id not in first_click_rank:
                first_click_rank[c.query_id] = c.rank
        dwell = [c.dwell_seconds for c in doc_clicks if c.dwell_seconds is not None]
        durations = [(s.last_seen - s.started_at).total_seconds() / 60 for s in sessions]
        missions = Counter(q.mission_id for q in queries)
        page_requests = [r for r in requests if r.endpoint != "static"]

        def ratio(a, b):
            return round(a / b, 3) if b else 0.0

        kpis = {
            "visitors": len(self.visitors),
            "sessions": len(sessions),
            "requests": len(page_requests),
            "queries": len(queries),
            "unique_queries": len({q.text.lower().strip() for q in queries}),
            "clicks": len(doc_clicks),
            "external_clicks": sum(c.kind == "external" for c in clicks),
            "ctr": ratio(len(clicked_queries), len(queries)),
            "zero_result_rate": ratio(sum(q.n_results == 0 for q in queries), len(queries)),
            "relaxed_rate": ratio(sum(q.relaxed and q.n_results > 0 for q in queries), len(queries)),
            "mrr_first_click": round(mean(1 / r for r in first_click_rank.values()), 3) if first_click_rank else 0.0,
            "avg_terms": round(mean(q.n_terms for q in queries), 2) if queries else 0.0,
            "avg_queries_per_session": ratio(len(queries), len(sessions)),
            "avg_missions_per_session": ratio(len(missions), len(sessions)),
            "median_dwell_s": round(median(dwell), 1) if dwell else 0.0,
            "avg_session_min": round(mean(durations), 1) if durations else 0.0,
            "avg_search_ms": round(mean(q.elapsed_ms for q in queries), 2) if queries else 0.0,
            "avg_latency_ms": round(mean(r.latency_ms for r in page_requests), 2) if page_requests else 0.0,
        }

        terms = Counter(t for q in queries for t in q.terms)
        top_queries = Counter(q.text.lower().strip() for q in queries).most_common(top)
        zero_queries = Counter(q.text.lower().strip() for q in queries if q.n_results == 0).most_common(top)
        doc_counter = Counter(c.pid for c in doc_clicks)
        top_docs = [{"pid": pid, "clicks": n,
                     "title": " ".join(corpus[pid].title.split()) if corpus and pid in corpus else pid}
                    for pid, n in doc_counter.most_common(top)]
        rank_counts = Counter(min(c.rank, 21) for c in doc_clicks if c.rank)
        dwell_bins = [("< 10 s", 0, 10), ("10-30 s", 10, 30), ("30-60 s", 30, 60), ("1-2 min", 60, 120),
                      ("2-5 min", 120, 300), ("> 5 min", 300, float("inf"))]
        queries_by_hour = Counter(q.timestamp.hour for q in queries)
        queries_by_day = Counter(q.timestamp.strftime("%Y-%m-%d") for q in queries)
        status = Counter(r.status for r in page_requests)
        endpoint = Counter(r.endpoint for r in page_requests)
        length_counts = Counter(min(q.n_terms, 6) for q in queries)
        sorts = Counter(q.filters.get("sort", "relevance") for q in queries)

        return {
            "kpis": kpis,
            "queries_by_hour": [queries_by_hour.get(h, 0) for h in range(24)],
            "queries_by_day": sorted(queries_by_day.items()),
            "top_queries": top_queries,
            "zero_queries": zero_queries,
            "top_terms": terms.most_common(15),
            "query_length": [(("6+" if n == 6 else str(n)), length_counts.get(n, 0)) for n in range(1, 7)],
            "clicks_by_rank": [(("21+" if r == 21 else str(r)), rank_counts.get(r, 0)) for r in range(1, 22)],
            "dwell_bins": [(label, sum(lo <= d < hi for d in dwell)) for label, lo, hi in dwell_bins],
            "top_docs": top_docs,
            "browsers": Counter(s.browser for s in sessions).most_common(),
            "os": Counter(s.os for s in sessions).most_common(),
            "devices": Counter(s.device for s in sessions).most_common(),
            "status": sorted(status.items()),
            "endpoints": endpoint.most_common(),
            "sorts": sorts.most_common(),
            "queries_per_mission": sorted(Counter(missions.values()).items()),
            "sessions_table": sorted(
                ({"session": s.session_id[:8], "start": s.started_at.strftime("%d/%m %H:%M"), "browser": s.browser,
                  "os": s.os, "device": s.device, "queries": s.queries, "missions": s.missions, "clicks": s.clicks,
                  "minutes": round((s.last_seen - s.started_at).total_seconds() / 60, 1)} for s in sessions),
                key=lambda row: row["start"], reverse=True)[:15],
        }

    def export(self) -> dict:
        """All tables as JSON-ready dictionaries"""
        def rows(items):
            return [{k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in asdict(i).items()} for i in items]
        with self._lock:
            return {"visitors": rows(self.visitors.values()), "sessions": rows(self.sessions.values()),
                    "requests": rows(self.requests), "queries": rows(self.queries.values()), "clicks": rows(self.clicks)}
