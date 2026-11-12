"""The embedding abstraction used by the rest of the application.

Services depend on this protocol, not on fastembed, so the model can be swapped (e.g. for
SPECTER2 or a hosted API) by adding one class, and tests can use a fast deterministic fake.
"""

import time
from collections.abc import Sequence
from typing import Literal, Protocol

import anyio
import anyio.to_thread

from paper_similarity.core import metrics


class Embedder(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def max_tokens(self) -> int:
        """Longest input (in tokens, special tokens included) the model reads without truncating."""
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one L2-normalised vector per text, in the same order. CPU-bound."""
        ...

    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        """Character span of every content token (no special tokens), without truncation."""
        ...


async def embed_in_thread(
    embedder: Embedder,
    texts: Sequence[str],
    purpose: Literal["paper", "submission_chunk"],
    limiter: anyio.CapacityLimiter | None = None,
) -> list[list[float]]:
    """Run the CPU-bound embedding in a worker thread so the event loop keeps serving requests.

    ONNX Runtime releases the GIL during inference, so this gives real parallelism.

    ``limiter`` caps how many embeddings run at once; further callers wait. Without it the
    only bound is anyio's default pool of 40 threads, and one inference already uses every
    core, so a burst of long submissions would only make all of them slower.
    """
    if not texts:
        return []
    batch = list(texts)

    def timed_embed() -> list[list[float]]:
        # Timed inside the thread: the metric is inference time, not time spent queueing.
        started = time.perf_counter()
        vectors = embedder.embed(batch)
        metrics.EMBEDDING_SECONDS.observe(time.perf_counter() - started)
        return vectors

    vectors = await anyio.to_thread.run_sync(timed_embed, limiter=limiter)
    metrics.EMBEDDINGS_COMPUTED.labels(purpose=purpose).inc(len(batch))
    return vectors
