import random
from typing import Optional
import requests

from src.retrievers.constants import USER_AGENTS


class WikipediaRetriever:
    def __init__(self):
        self.base_url = "https://en.wikipedia.org/w/api.php"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": random.choice(USER_AGENTS),
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

    def search_wikipedia(self, title: str, artist: str) -> Optional[str]:
        """Search Wikipedia for a painting URL using title and artist heuristics."""

        search_terms = [f"{title} {artist}", f"{title} painting", title]

        for search_term in search_terms:
            try:
                params = {
                    "action": "opensearch",
                    "search": search_term,
                    "limit": 1,
                    "namespace": 0,
                    "format": "json",
                }

                response = self.session.get(self.wiki_api, params=params, timeout=5)
                data = response.json()

                if len(data) > 3 and data[3]:
                    return data[3][0]

            except Exception:
                continue

        return None
