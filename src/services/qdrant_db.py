import base64
import hashlib
import io
import logging
import os
import uuid
from typing import Dict, Generator, List
from PIL import Image

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

from config import qdrant_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QdrantDB:
    TOP_K = 5

    def __init__(self):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_HOST"), api_key=os.getenv("QDRANT_TOKEN") or None
        )
        self.collection_name = qdrant_config["collection"]
        try:
            self.client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' found")
        except Exception:
            raise

        self.model = SentenceTransformer(
            "clip-ViT-B-32", token=os.getenv("HF_TOKEN"), model_kwargs={"use_fast": False}
        )

    def to_uuid(self, text: str) -> str:
        """Return a deterministic UUID generated from the given text."""
        return str(uuid.UUID(hashlib.md5(text.encode("utf-8")).hexdigest()))

    def ingest_paintings(self, paintings: List[Dict]):
        """Encode and store painting embeddings in Qdrant."""
        logger.info("Generating paintings embeddings...")
        tokens = [f"{painting['title']} by {painting['artist']}" for painting in paintings]
        embeddings = self.model.encode(tokens, show_progress_bar=True)

        points = []
        for painting, embedding in zip(paintings, embeddings):
            points.append(
                PointStruct(
                    id=self.to_uuid(painting.get("id")),
                    vector=embedding.tolist(),
                    payload=painting,
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=points)

        logger.info(f"Qdrant updated with {len(paintings)} embeddings")

    def search(self, image_input: str) -> List[Dict]:
        """Return the top matching paintings for a given image."""
        logger.info("Searching for a painting...")

        try:
            if len(image_input) > 100:  # pure base64
                image_bytes = base64.b64decode(image_input)
                img = Image.open(io.BytesIO(image_bytes))
            else:
                img = Image.open(image_input)

            img_embedding = self.model.encode(img)

            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=img_embedding.tolist(),
                query_filter={},
                limit=self.TOP_K,
            )

            matches = []
            for hit in results:
                matches.append(
                    {
                        "title": hit.payload["title"],
                        "artist": hit.payload["artist"],
                        "image_url": hit.payload["image_url"],
                        "url": hit.payload["url"],
                        "score": hit.score,
                    }
                )

            return matches
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise
        finally:
            if 'img' in locals():
                img.close()

    def scroll(self, batch_size: int = 100, limit: int = None) -> Generator[List[Dict], None, None]:
        """Generator that retrieves all documents from QdrantDB in batches."""
        offset = None
        batches = 0
        while True and (limit is None or batches < limit):
            response = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points, offset = response[0], response[1]
            batches += 1
            if not points:
                break

            yield [dict(id=p.id, payload=p.payload) for p in points]
            if offset is None:
                break


if __name__ == "__main__":
    load_dotenv()

    image_path = "/home/ubuntu/artguide/img/matrimoni_arnolfini.jpg"
    with open(image_path, 'rb') as image_file:
        image_bytes = image_file.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    db = QdrantDB()
    results = db.search(image_base64)
    print(results[0])
