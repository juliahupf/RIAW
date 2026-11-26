"""
Synthetic traffic for the analytics dashboard.

Real users are not available for the project, so this script imitates them: visitors with different browsers and
devices, sessions spread over one week, missions in which the user refines the query, clicks that depend on the rank
of the result (position bias) and dwell times. Every action goes through the Flask application, so the data is
collected by the same code as real traffic. A simulated clock makes the timestamps realistic.

    python project_progress/part_4/simulate_traffic.py            # simulate and print the key indicators
    python project_progress/part_4/simulate_traffic.py --serve    # simulate, then open the app with that data
"""
import argparse
import os
import random
import re
import sys
from datetime import datetime, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from myapp.analytics.analytics_data import AnalyticsData  # noqa: E402
from web_app import create_app  # noqa: E402

USER_AGENTS = [   # (user agent, share of visitors)
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36", 0.32),
    ("Mozilla/5.0 (Linux; Android 13; SM-S911B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36", 0.24),
    ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1", 0.16),
    ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15", 0.12),
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0", 0.08),
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0", 0.05),
    ("Mozilla/5.0 (iPad; CPU OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1", 0.03),
]

# Missions: a first query and possible refinements of the same goal
MISSIONS = [
    ["kurta", "women cotton kurta", "women cotton kurta under 800"],
    ["running shoes", "men running shoes", "men running shoes under 1500"],
    ["tshirt", "polo neck t-shirt men", "polo neck tshirt men white"],
    ["jacket", "winter hooded jacket men", "hooded jacket men under 1500"],
    ["sneakers", "white sneakers women"],
    ["jeans", "men slim jeans blue", "men slim fit jeans black"],
    ["sweatshirt women", "women full sleeve sweatshirt cotton"],
    ["track pants men", "men track pants black"],
    ["formal shirt men", "men formal shirt white cotton"],
    ["women leggings", "women leggings black cotton"],
    ["socks men", "men ankle socks pack"],
    ["cap", "men cotton cap black"],
    ["iphone charger"],
    ["smart watch", "watch men"],
    ["saree silk red"],
]

# When people search: weights for each hour of the day (more traffic in the evening)
HOUR_WEIGHTS = [1, 1, 0, 0, 0, 0, 1, 2, 3, 4, 5, 5, 6, 5, 4, 4, 5, 6, 8, 10, 11, 10, 7, 3]


class SimClock:
    """Clock that the simulation moves forward; AnalyticsData reads it instead of the real time"""

    def __init__(self, start):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


def results_of(page_html):
    """(link to the document page, rank) of every result in a results page"""
    links = re.findall(r'href="(/doc_details\?pid=[A-Z0-9]+&amp;qid=\d+&amp;rank=(\d+))"', page_html)
    return [(link.replace("&amp;", "&"), int(rank)) for link, rank in links]


def simulate(app, clock, visitors=40, days=7, seed=7):
    rng = random.Random(seed)
    start = clock.now
    agents, weights = zip(*USER_AGENTS)
    for _ in range(visitors):
        client = app.test_client()                     # one cookie jar = one visitor
        headers = {"User-Agent": rng.choices(agents, weights)[0]}
        # Sit-downs of this visitor, in chronological order and on different days
        sit_downs = sorted(start + timedelta(days=day, hours=rng.choices(range(24), HOUR_WEIGHTS)[0],
                                             minutes=rng.randrange(60))
                           for day in rng.sample(range(days), rng.choices([1, 2, 3], [0.55, 0.3, 0.15])[0]))
        for sit_down in sit_downs:
            clock.now = sit_down
            client.get("/", headers=headers)
            clock.advance(rng.uniform(3, 15))
            for mission in rng.sample(MISSIONS, rng.choices([1, 2, 3], [0.5, 0.35, 0.15])[0]):
                steps = mission[:rng.randint(1, len(mission))]
                for query in steps:
                    response = client.get("/search", query_string={"q": query}, headers=headers)
                    location = response.headers.get("Location")
                    page = client.get(location, headers=headers).get_data(as_text=True) if location else ""
                    clock.advance(rng.uniform(4, 20))
                    # The user scans the results from the top: the chance to click falls with the rank
                    for link, rank in results_of(page)[:20]:
                        if rng.random() < 0.55 / rank ** 0.9:
                            client.get(link, headers=headers)
                            dwell = rng.lognormvariate(3.4, 0.9)         # median of about 30 s
                            clock.advance(dwell)
                            if rng.random() < 0.15:                      # sometimes goes to the shop
                                pid = re.search(r"pid=([A-Z0-9]+)", link).group(1)
                                client.get("/out", query_string={"pid": pid, "rank": rank}, headers=headers)
                            client.get(location, headers=headers)        # back to the results
                            clock.advance(rng.uniform(2, 8))
                    clock.advance(rng.uniform(5, 30))
            clock.advance(rng.uniform(60, 600))
        client.get("/dashboard", headers=headers) if rng.random() < 0.1 else None
    return app.config["ANALYTICS"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--visitors", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--serve", action="store_true", help="start the web app with the simulated data")
    args = parser.parse_args()

    clock = SimClock(datetime(2026, 11, 16, 0, 0))
    analytics = AnalyticsData(clock=clock)
    app = create_app(analytics=analytics)
    simulate(app, clock, visitors=args.visitors, seed=args.seed)
    for key, value in analytics.summary()["kpis"].items():
        print(f"{key:>26}: {value}")
    if args.serve:
        analytics.clock = datetime.now     # from now on, real time for real requests
        app.run(port=8088, host="127.0.0.1", threaded=False)


if __name__ == "__main__":
    main()
