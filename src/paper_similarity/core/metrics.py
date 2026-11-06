"""Prometheus metrics for the application's own operations.

HTTP request metrics (count, latency per route) are added by the instrumentator in the app
factory; these counters cover work that happens outside a request or inside a service.
"""

from prometheus_client import Counter, Histogram

ARXIV_FAILURES = Counter(
    "pulse_arxiv_request_failures_total",
    "arXiv API requests that failed after all retries.",
)
PAPERS_INGESTED = Counter(
    "pulse_papers_ingested_total",
    "arXiv papers written to the database, by outcome.",
    ["outcome"],  # inserted | updated | unchanged | skipped
)
EMBEDDINGS_COMPUTED = Counter(
    "pulse_embeddings_computed_total",
    "Texts embedded, by purpose.",
    ["purpose"],  # paper | submission_chunk
)
EMBEDDING_SECONDS = Histogram(
    "pulse_embedding_seconds",
    "Time spent embedding one batch of texts.",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
SIMILARITY_RESULTS = Histogram(
    "pulse_similarity_results",
    "Number of matches returned by a similarity query.",
    buckets=(0, 1, 5, 10, 20, 50),
)
