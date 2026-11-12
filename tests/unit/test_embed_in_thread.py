"""The embedding concurrency limit: requests queue instead of oversubscribing the CPU."""

import threading
import time
from collections.abc import Sequence

import anyio

from paper_similarity.embeddings.base import embed_in_thread


class SlowEmbedder:
    """Records how many ``embed`` calls overlap."""

    model_name = "test/slow"
    dimension = 1
    max_tokens = 512

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = 0
        self.max_running = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        with self._lock:
            self._running += 1
            self.max_running = max(self.max_running, self._running)
        time.sleep(0.05)
        with self._lock:
            self._running -= 1
        return [[1.0] for _ in texts]

    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        return []


async def run_four(embedder: SlowEmbedder, limiter: anyio.CapacityLimiter | None) -> None:
    async with anyio.create_task_group() as group:
        for _ in range(4):
            group.start_soon(embed_in_thread, embedder, ["text"], "submission_chunk", limiter)


async def test_limiter_bounds_concurrent_embeddings() -> None:
    embedder = SlowEmbedder()
    await run_four(embedder, anyio.CapacityLimiter(1))
    assert embedder.max_running == 1


async def test_without_a_limiter_embeddings_overlap() -> None:
    embedder = SlowEmbedder()
    await run_four(embedder, None)
    assert embedder.max_running > 1
