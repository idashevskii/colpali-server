import logging
from typing import Any, List
import httpx
import base64
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TimecodedText:
    text: str
    start: float
    end: float


BatchEmbeddings = List[List[List[float]]]


class ColpaliClient:
    def __init__(self, base_url: str) -> None:
        self.http = httpx.Client(base_url=base_url, timeout=3600)

    def process_queries(self, queries: List[str]):
        return self.__request(
            "/process-queries",
            {
                "queries": queries,
            },
        )['embedding_batches']

    def process_images(self, images: List[bytes]):
        return self.__request(
            "/process-images",
            {
                "images": [base64.b64encode(img).decode("ascii") for img in images],
            },
        )['embedding_batches']

    def score(
        self, heystack_batch: BatchEmbeddings, needle_batch: BatchEmbeddings
    ) -> float:
        return self.__request(
            "/score",
            {
                "heystack_batch": heystack_batch,
                "needle_batch": needle_batch,
            },
        )['score']

    def __request(self, endpoint: str, payload: Any) -> Any:
        for n in range(3):
            try:
                res = self.http.post(endpoint, json=payload)
                res.raise_for_status()
                return res.json()
            except httpx.HTTPError:
                logger.exception(f"Attempt {n=} failed")
        raise RuntimeError("Request to ColPali failed in all attempts")
