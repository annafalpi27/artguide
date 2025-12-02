import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Generator, List

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
        yield from self.db.scroll(batch_size=self.batch_size, limit=1)

    def transform(self, batch: List[Dict]) -> List[Dict]:
        """Transform a single batch using internal threads"""
        logger.info(f"Transforming batch of {len(batch)} paintings...")
        enriched = []

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            # Map future to original painting dict
            future_to_painting = {
                executor.submit(
                    self.retriever.get_painting_details,
                    painting["payload"]["wikiart_id"]
                ): painting
                for painting in batch
                if painting["payload"].get("wikiart_id")  # Only if ID exists
            }

            for future in as_completed(future_to_painting):
                painting = future_to_painting[future]
                try:
                    enriched_data = future.result()
                    # Update the payload with enriched data
                    painting["payload"].update(enriched_data)
                    enriched.append(painting)
                except Exception as e:
                    wikiart_id = painting["payload"].get("wikiart_id")
                    logger.error(f"Error enriching painting {wikiart_id}: {e}")
                    # Append original painting on error
                    enriched.append(painting)

        return enriched

    def load(self, batch: List[Dict]) -> None:
        """Load a single batch"""
        logger.info(f"Loading batch of {len(batch)} paintings to DB...")
        self.db.ingest_paintings(batch)
        pass


if __name__ == "__main__":
    etl = WikiArtETL(batch_size=100, workers=4)
    etl.run()
