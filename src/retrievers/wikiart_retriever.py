import logging
import random
import requests
import urllib

from typing import Dict

from src.retrievers.wikipedia_retriever import WikipediaRetriever
from src.retrievers.constants import USER_AGENTS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WikiArtRetriever:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": random.choice(USER_AGENTS),
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        self.wikipedia_retriever = WikipediaRetriever()

    def get_paintings_page(self, paginationToken: str) -> Dict:
        """Fetch a page of most-viewed paintings from the WikiArt API."""

        url = "https://www.wikiart.org/en/api/2/MostViewedPaintings"

        if paginationToken:
            decoded_token = urllib.parse.unquote(paginationToken)
            params = {"paginationToken": decoded_token}
        else:
            params = {}

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.warning("WikiArt request timed out.")
        except requests.exceptions.RequestException as e:
            logger.warning(f"WikiArt request failed: {e.__class__.__name__}")

        return {"data": [], "hasMore": False}

    def parse_and_enrich_painting(self, painting: Dict) -> Dict:
        """Normalize and enrich a WikiArt painting record."""

        painting.pop("artistUrl")
        painting.pop("artistId")
        painting.pop("width")
        painting.pop("height")
        painting["wikiart_id"] = painting.pop("id")
        painting["artist"] = painting.pop("artistName")
        painting["image_url"] = painting.pop("image")

        title = painting.get("title")
        artist = painting.get("artist")

        if not title or not artist:
            painting["url"] = None
            return painting

        wiki_url = self.wikipedia_retriever.search_wikipedia(title, artist)
        painting["url"] = wiki_url

        return painting

    def get_painting_details(self, painting_id: str) -> Dict:
        """Retrieve museum and description details for a painting."""

        url = f"https://www.wikiart.org/en/api/2/Painting/{painting_id}"
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            response_dict = response.json()
            return {
                "museum": response_dict.get("galleries"),
                "description": response_dict.get("description"),
            }
        except requests.exceptions.Timeout:
            logger.warning("WikiArt painting details request timed out.")
        except requests.exceptions.RequestException as e:
            logger.warning(f"WikiArt painting details request failed: {e.__class__.__name__}")

        return {}


if __name__ == "__main__":
    retriever = WikiArtRetriever()
    results = retriever.get_paintings_page(
        "%2fb%2bpV6JP%2f8p2XUjPW1r9A0o6GHn97FhFB6Yd%2bcTxyUowTEr%2bntRX2GWAu77cr07Zu%2fvVsHUVdeK7Wl5B4kcGLEvjcsBNS7KdH499VH8bQpU%3d"  # noqa
    )  # noqa
    paintings = results["data"]
    for painting in paintings[0:1]:
        enriched = retriever.parse_and_enrich_painting(painting)

    print()
