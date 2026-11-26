import os
import re
import secrets
import time
from urllib.parse import urlencode

from dotenv import load_dotenv
from flask import Flask, abort, g, jsonify, redirect, render_template, request, session, url_for
from markupsafe import Markup, escape

from myapp.analytics.analytics_data import AnalyticsData
from myapp.generation.rag import ImprovedRAG
from myapp.search.load_corpus import load_corpus
from myapp.search.search_engine import SORT_OPTIONS, SearchEngine, SearchFilters

load_dotenv()  # take environment variables from .env

ROOT = os.path.dirname(os.path.realpath(__file__))
UNTRACKED_ENDPOINTS = {"static", None}


def _float_arg(name):
    try:
        value = request.args.get(name, type=float)
        return value if value is not None and value >= 0 else None
    except ValueError:
        return None


def read_filters() -> SearchFilters:
    sort = request.args.get("sort", "relevance")
    return SearchFilters(sort=sort if sort in SORT_OPTIONS else "relevance",
                         max_price=_float_arg("max_price"),
                         min_rating=_float_arg("min_rating"),
                         in_stock_only=request.args.get("in_stock") == "1")


def render_answer(text: str, query_id) -> Markup:
    """RAG answers use **bold** and one line per item. Escape them, then add bold, line breaks and product links"""
    html = str(escape(text))
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"\[([A-Z0-9]{16})\]",
                  lambda m: f'[<a href="{url_for("doc_details", pid=m.group(1), qid=query_id)}">{m.group(1)}</a>]', html)
    return Markup(html.replace("\n", "<br>"))


def create_app(data_file=None, analytics=None, rag=None, engine=None):
    """
    Build the application.
    :param data_file: path of the corpus (default: DATA_FILE_PATH in .env, or the dataset in data/)
    :param analytics: AnalyticsData object (tests and simulations pass one with their own clock)
    """
    app = Flask(__name__)
    # random 'secret_key' is used for persisting data in secure cookie
    app.secret_key = os.getenv("SECRET_KEY") or secrets.token_hex(16)
    app.config["SESSION_COOKIE_NAME"] = os.getenv("SESSION_COOKIE_NAME", "irwa_session")

    if engine is None:
        path = data_file or os.path.join(ROOT, os.getenv("DATA_FILE_PATH", "data/fashion_products_dataset.json"))
        corpus = load_corpus(path)
        engine = SearchEngine(corpus)
    corpus = engine.corpus
    analytics = analytics or AnalyticsData()
    rag = rag or ImprovedRAG()
    rag_cache = {}
    app.config.update(ENGINE=engine, ANALYTICS=analytics)

    # ------------------------------------------------------------------ tracking of every request

    @app.before_request
    def start_tracking():
        g.start = time.perf_counter()
        if request.endpoint in UNTRACKED_ENDPOINTS:
            return
        visitor_id, session_id = analytics.session_for(session.get("visitor_id"), session.get("session_id"),
                                                       request.headers.get("X-Forwarded-For", request.remote_addr or ""),
                                                       request.headers.get("User-Agent", ""))
        session["visitor_id"], session["session_id"] = visitor_id, session_id
        session.permanent = True
        g.session_id = session_id

    @app.after_request
    def finish_tracking(response):
        if request.endpoint not in UNTRACKED_ENDPOINTS and "session_id" in g:
            analytics.log_request(g.session_id, request.method, request.full_path.rstrip("?"), request.endpoint,
                                  response.status_code, (time.perf_counter() - g.start) * 1000, request.referrer)
        return response

    # ------------------------------------------------------------------ pages

    @app.route("/")
    def index():
        return render_template("index.html", page_title="Welcome", sort_options=SORT_OPTIONS)

    @app.route("/search", methods=["GET", "POST"])
    def search():
        if request.method == "POST":   # old form of the template: same search with GET
            return redirect(url_for("search", q=request.form.get("search-query", "")))
        query = request.args.get("q", "").strip()
        if not query:
            return redirect(url_for("index"))
        filters = read_filters()
        page = request.args.get("page", 1, type=int) or 1
        response = engine.search(query, page=page, filters=filters)

        # Coming back to the results page closes the dwell time of the last clicked document
        analytics.close_dwell(g.session_id)
        known = analytics.query(request.args.get("qid", type=int))
        if known is None or known.session_id != g.session_id or known.text != query:
            # New search: log it once and redirect to a URL with its id, so that going back to this page,
            # changing page or reloading does not count as a new query
            query_id = analytics.log_query(g.session_id, query, response.terms, response.total, response.relaxed,
                                           response.elapsed_ms, response.price_from_query,
                                           {"sort": filters.sort, "max_price": filters.max_price,
                                            "min_rating": filters.min_rating, "in_stock": filters.in_stock_only})
            args = request.args.to_dict()
            args["qid"] = query_id
            return redirect(url_for("search") + "?" + urlencode(args))

        page_args = {k: v for k, v in request.args.items() if k != "page"}
        html = render_template("results.html", page_title="Results", response=response, qid=known.query_id,
                               filters=filters, sort_options=SORT_OPTIONS, page_args=page_args)
        result = app.make_response(html)
        result.headers["Cache-Control"] = "no-store"   # the back button reloads the page, so the return is logged
        return result

    @app.route("/rag")
    def rag_summary():
        """Called by the results page after it loads, so the user does not wait for the LLM to see the results"""
        query = analytics.query(request.args.get("qid", type=int))
        if query is None or query.session_id != g.session_id:
            abort(404)
        if query.query_id not in rag_cache:
            filters = SearchFilters(**{"sort": query.filters.get("sort", "relevance"),
                                       "max_price": query.filters.get("max_price"),
                                       "min_rating": query.filters.get("min_rating"),
                                       "in_stock_only": bool(query.filters.get("in_stock"))})
            response = engine.search(query.text, page_size=20, filters=filters)
            max_price = min([p for p in (filters.max_price, response.price_from_query) if p is not None], default=None)
            rag_cache[query.query_id] = rag.generate_response(query.text, response.hits, query_terms=response.terms,
                                                              max_price=max_price, relaxed=response.relaxed)
        answer = rag_cache[query.query_id]
        return jsonify(status=answer.status, html=str(render_answer(answer.text, query.query_id)))

    @app.route("/doc_details")
    def doc_details():
        pid = request.args.get("pid", "")
        if pid not in corpus:
            abort(404)
        query_id, rank = request.args.get("qid", type=int), request.args.get("rank", type=int)
        analytics.log_click(g.session_id, pid, query_id, rank, kind="details")
        query = analytics.query(query_id)
        back = url_for("search", q=query.text, qid=query.query_id) if query else None
        return render_template("doc_details.html", page_title="Details", doc=corpus[pid], back_url=back,
                               qid=query_id, rank=rank)

    @app.route("/out")
    def outbound():
        """Link to the product in the original shop, through our server so that the click is recorded"""
        pid = request.args.get("pid", "")
        if pid not in corpus or not corpus[pid].url:
            abort(404)
        analytics.log_click(g.session_id, pid, request.args.get("qid", type=int), request.args.get("rank", type=int),
                            kind="external")
        return redirect(corpus[pid].url)

    @app.route("/dashboard")
    def dashboard():
        return render_template("dashboard.html", page_title="Dashboard", data=analytics.summary(corpus))

    @app.route("/analytics/export")
    def analytics_export():
        return jsonify(analytics.export())

    @app.errorhandler(404)
    def not_found(_):
        return render_template("not_found.html", page_title="Not found"), 404

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(port=8088, host="0.0.0.0", threaded=False, debug=os.getenv("DEBUG") == "True")
