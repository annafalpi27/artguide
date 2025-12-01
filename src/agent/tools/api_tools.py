import os
import logging
from typing import Dict, List

import numpy as np
from requests import post


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


class APITools:

    def __init__(self, base_url: str):
        self.base_url = base_url

    def _get_headers(self) -> Dict:
        return {
            "Authorization": f"Bearer {os.getenv('API_TOKEN')}",
            "Content-Type": "application/json",
        }

    def search_painting(self, image_path: str) -> List[Dict]:
        """Searches for a painting in the Qdrant DB."""

        params = {"image_data": image_path}
        response = post(
            url=f"{self.base_url}/search",
            headers=self._get_headers(),
            json=params,
        )
        return response.json()

    def synthesize_speech(self, text: str, speaker: str, language: str) -> Dict:
        """Perform voice synthesis"""

        logger.info(f"Generating audio for text length: {len(text)}")

        params = {"text": text, "speaker": speaker, "language": language}

        response = post(
            url=f"{self.base_url}/synthesize",
            headers=self._get_headers(),
            json=params,
        )
        results = response.json()

        return {"samples": np.array(results["samples"], dtype=np.float32), "sr": results["sr"]}
