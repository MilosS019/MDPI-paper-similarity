"""Prometheus metrics for the application's own operations.

HTTP request metrics (count, latency per route) are added by the instrumentator in the app
factory. The counters below are recorded by whichever process does the work:

* In the API process (exported at /metrics): submission embeddings, embedding latency and
  similarity result sizes.
* In the ``pulse`` CLI (ingestion, indexing, arXiv failures): the process is short-lived and
  has no /metrics endpoint, so these values are not scraped. Its results are reported in its
  log lines and summary output instead. Exporting them would need a Prometheus Pushgateway
  (or running ingestion as a long-lived worker); see the README's known limitations.
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
