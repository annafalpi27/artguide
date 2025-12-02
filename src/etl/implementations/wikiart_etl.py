import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Dict, Generator, List

from dotenv import load_dotenv

from src.etl.base_paintings_etl import BasePaintingsETL
from src.retrievers.wikiart_retriever import WikiArtRetriever
from src.services.qdrant_db import QdrantDB

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WikiArtETL(BasePaintingsETL):
    def __init__(self, batch_size=600, workers=5):
        super().__init__(batch_size, workers)

        self.retriever = WikiArtRetriever()
        self.db = QdrantDB()

    def extract(self) -> Generator[List[Dict], None, None]:
        """Yield single batch"""
        logger.info("Starting extraction...")
        batch = []
        pagination_token = ""

        while True:
            result = self.retriever.get_paintings_page(pagination_token)
            paintings = result.get("data", [])

            if not paintings:
                logger.info("No paintings found")
                break

            batch.extend(paintings)

            while len(batch) >= self.batch_size:
                yield_batch = batch[: self.batch_size]
                batch = batch[self.batch_size :]  # noqa
                logger.info(f"Extracted batch of {len(yield_batch)} paintings")
                yield yield_batch

            if not result.get("hasMore", False):
                break

            pagination_token = result.get("paginationToken", "")
            time.sleep(0.5)

        if batch:
            logger.info(f"Extracted final batch of {len(batch)} paintings")
            yield batch

    def transform(self, batch: List[Dict]) -> List[Dict]:
        """Transform a single batch using internal threads"""
        logger.info(f"Transforming batch of {len(batch)} paintings...")
        enriched = []

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_painting = {
                executor.submit(self.retriever.parse_and_enrich_painting, painting): painting
                for painting in batch
            }

            for future in as_completed(future_to_painting):
                try:
                    enriched_painting = future.result()
                    enriched.append(enriched_painting)
                except Exception as e:
                    logger.error(f"Error enriching painting: {e}")
                    painting = future_to_painting[future]
                    enriched.append(painting)

        logger.info(f"Batch transformed: {len(enriched)} paintings")
        return enriched

    def load(self, batch: List[Dict]) -> None:
        """Load a single batch"""
        logger.info(f"Loading batch of {len(batch)} paintings to DB...")
        self.db.ingest_paintings(batch)


if __name__ == "__main__":
    load_dotenv()
    etl = WikiArtETL(batch_size=600, workers=10)
    etl.run()
