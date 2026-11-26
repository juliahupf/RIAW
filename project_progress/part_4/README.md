# Part 4: RAG, User Interface and Web Analytics

The web application is [`web_app.py`](../../web_app.py), in the repository root.
This folder contains the notebook used for the report
([`part4.ipynb`](./part4.ipynb)) and the traffic simulator
([`simulate_traffic.py`](./simulate_traffic.py)).

## 1. Prepare the environment

From the repository root, with the `irwa_env` environment (any Python 3.10+):

```bash
python -m pip install flask groq python-dotenv httpagentparser pydantic nltk numpy pandas
```

Create a file named `.env` in the repository root:

```
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.1-8b-instant
SECRET_KEY=any_random_text
DATA_FILE_PATH=data/fashion_products_dataset.json
```

Only `GROQ_API_KEY` is needed for the AI summary. Without it the web works, and
the summary box says that the RAG is not available. Never commit `.env`
(it is in `.gitignore`).

## 2. Run the web application

```bash
python web_app.py
```

Open http://127.0.0.1:8088. The index is built when the app starts
(a few seconds).

| Page | URL |
|---|---|
| Search | `/` |
| Results (AI summary, filters, pages) | `/search?q=...` |
| Product details | `/doc_details?pid=...` |
| Analytics dashboard | `/dashboard` |
| All analytics data as JSON | `/analytics/export` |

## 3. See the dashboard with data

The analytics are kept in memory, so they start empty every time the app
starts. To fill them with one week of simulated traffic (40 visitors) and open
the app with that data:

```bash
python project_progress/part_4/simulate_traffic.py --serve
```

Without `--serve` it only prints the key indicators.

## 4. Code

| File | Content |
|---|---|
| `myapp/search/preprocessing.py` | Text processing of Part 1 with the fixes of Parts 2 and 3 |
| `myapp/search/search_engine.py` | `SearchEngine.search(query, page, page_size, filters)`: AND search, ranking with our score of Part 3, price in the query, relaxed search, filters, cache |
| `myapp/generation/rag.py` | `BaselineRAG` (given pipeline) and `ImprovedRAG` |
| `myapp/analytics/analytics_data.py` | Star schema in memory, sessions, missions, dwell time and dashboard data |
| `web_app.py` | Flask routes and request tracking |
| `templates/`, `static/` | Pages and style |

## 5. Reproduce the results of the report

Open `part4.ipynb` with the `irwa_env` kernel and use **Run All**. The RAG
answers need `GROQ_API_KEY`; the rest runs without it. The traffic simulation
uses a fixed seed, so it always gives the same numbers.
