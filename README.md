# Paper Similarity Service

A Python backend that ingests scientific papers from arXiv, stores vector embeddings of them,
and returns the indexed papers most similar to a submitted manuscript.

- **Ingestion:** `pulse ingest` pulls papers from the arXiv API. The parsed records are landed in Parquet, the
  papers go to Postgres, and their embeddings go to Qdrant. It is safe to re-run.
- **Submit:** `POST /api/v1/papers` validates, chunks, embeds and stores a manuscript.
- **Similar:** `GET /api/v1/posts/{id}/similar` returns arXiv papers ranked by cosine similarity.

Stack: Python 3.12 · FastAPI · Pydantic v2 · SQLAlchemy 2 (async) + Alembic · PostgreSQL 16 ·
Qdrant · fastembed (`BAAI/bge-small-en-v1.5`, ONNX, runs locally) · httpx + tenacity ·
pyarrow / DuckDB · structlog · Prometheus · uv · ruff · mypy · pytest + testcontainers.

No paid service or API key is needed anywhere: the embedding model runs locally and arXiv is
free.

More detail on the design is in [docs/architecture.md](docs/architecture.md).

---

## Contents

1. [Quick start (Docker)](#1-quick-start-docker)
2. [Local development](#2-local-development)
3. [Configuration](#3-configuration)
4. [Ingesting arXiv papers](#4-ingesting-arxiv-papers)
5. [API usage](#5-api-usage)
6. [Running the tests](#6-running-the-tests)
7. [Architecture overview](#7-architecture-overview)
8. [Technical decisions and trade-offs](#8-technical-decisions-and-trade-offs)
9. [Known limitations](#9-known-limitations)
10. [Use of AI tools](#10-use-of-ai-tools)

---

## 1. Quick start (Docker)

**Prerequisites:** Docker with Compose v2 and about 2 GB of free disk space. Nothing else.

```bash
# Build the image (the embedding model is downloaded at build time) and start
# Postgres, Qdrant, the migration job and the API.
docker compose up -d --build

# Wait until the API is ready.
curl http://localhost:8000/ready

# Ingest papers from arXiv (about 1 minute for 90 papers on a laptop CPU).
docker compose exec api pulse ingest --query "cat:cs.IR OR cat:q-bio.BM OR cat:astro-ph.CO" --max 90

# Submit a manuscript...
curl -X POST http://localhost:8000/api/v1/papers \
  -H "Content-Type: application/json" \
  -d '{"title": "Learning to rank for product search",
       "text": "We propose a neural ranking model for e-commerce search that combines dense retrieval with user click feedback to improve recommendation relevance."}'

# ...and fetch the most similar indexed papers (use the "id" from the previous response).
curl "http://localhost:8000/api/v1/posts/<id>/similar?limit=5"
```

Interactive API documentation (OpenAPI / Swagger UI) is at **http://localhost:8000/docs**.

If ports 5432, 6333 or 8000 are taken on your machine, override them:
`POSTGRES_PORT=55432 QDRANT_PORT=56333 API_PORT=8080 docker compose up -d`.

Stop with `docker compose down`. Data is kept in named volumes; `docker compose down -v`
deletes it.

## 2. Local development

**Prerequisites:** [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 automatically if
needed) and Docker, for Postgres and Qdrant.

```bash
uv sync                                   # create .venv from the lockfile (incl. dev tools)
cp .env.example .env                      # local settings; defaults match docker compose
docker compose up -d postgres qdrant      # only the backing services

uv run alembic upgrade head               # create the schema
uv run pulse ingest --query "cat:cs.CL AND abs:retrieval" --max 50
uv run uvicorn --factory paper_similarity.api.app:create_app --reload --port 8000
```

If you changed `POSTGRES_PORT`, set the same port in `PULSE_DATABASE_URL` in `.env`.

Quality checks (also available as pre-commit hooks: `uv run pre-commit install`):

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy src tests
```

## 3. Configuration

All settings are environment variables with the `PULSE_` prefix. A local `.env` file is also
read. Every setting has a working default for the compose setup; see
[`.env.example`](.env.example) for the full documented list. No secrets are needed, and
`.env` is git-ignored.

| Variable | Default | Purpose |
|---|---|---|
| `PULSE_DATABASE_URL` | `postgresql+asyncpg://pulse:pulse@localhost:5432/pulse` | Postgres connection |
| `PULSE_QDRANT_URL` | `http://localhost:6333` | Qdrant HTTP endpoint |
| `PULSE_QDRANT_COLLECTION` | `arxiv_papers` | Collection name |
| `PULSE_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Any fastembed model; changing it requires `pulse reindex` |
| `PULSE_EMBEDDING_CONCURRENCY` | `2` | Requests allowed to embed at once; others queue |
| `PULSE_ARXIV_PAGE_SIZE` | `50` | Results per arXiv API request |
| `PULSE_ARXIV_REQUEST_DELAY_SECONDS` | `3.0` | Pause between arXiv requests, as arXiv asks |
| `PULSE_ARXIV_MAX_ATTEMPTS` | `4` | Total attempts per request (first try + retries) before giving up |
| `PULSE_RAW_DATA_DIR` | `data/raw/arxiv` | Parquet landing zone |
| `PULSE_MAX_SUBMISSION_CHARS` | `100000` | Longest accepted submission |
| `PULSE_CHUNK_MAX_TOKENS` / `PULSE_CHUNK_OVERLAP_TOKENS` | `512` / `64` | Chunk window and overlap |
| `PULSE_LOG_LEVEL` / `PULSE_LOG_FORMAT` | `INFO` / `json` | `console` for human-readable logs |

Invalid settings stop the process at startup with a clear message (e.g. an overlap larger than
the window).

## 4. Ingesting arXiv papers

```bash
pulse ingest --query "<arXiv query>" [--max 50] [--sort-by submittedDate|lastUpdatedDate|relevance]
```

**Selecting papers:** `--query` takes the
[arXiv API query syntax](https://info.arxiv.org/help/api/user-manual.html#query_details). The
default order is newest submissions first. Examples:

| Query | Selects |
|---|---|
| `cat:cs.CL` | Computation and Language |
| `cat:cs.CL AND abs:retrieval` | NLP papers mentioning "retrieval" in the abstract |
| `cat:cs.IR OR cat:q-bio.BM OR cat:astro-ph.CO` | A mixed corpus (good for demos) |
| `ti:"graph neural network"` | Title phrase |
| `au:hinton` | Author |

**What a run does:**
1. Fetches the results page by page.
2. Writes each page to `data/raw/arxiv/run_ts=<UTC time>/part-NNNNN-<run id>.parquet`.
3. Upserts the papers into Postgres.
4. Embeds new or changed papers.
5. Upserts their vectors into Qdrant.

It prints a summary such as
`fetched=90 inserted=90 updated=0 unchanged=0 skipped=0 indexed=90`.

**Re-running is safe:**
- Papers are keyed by their arXiv id without the version suffix (`2401.12345`), so a run
  never creates duplicates.
- A new version of a paper updates the existing record.
- A paper is re-embedded only if its title or abstract changed.

The same run again prints `inserted=0 unchanged=90 indexed=0`.

**When arXiv fails:**
- Timeouts, connection errors, HTTP 429/5xx, arXiv's in-band error responses and its
  occasional empty pages are retried with exponential backoff.
- An invalid query (HTTP 400) fails immediately with arXiv's message.
- If arXiv stays down partway through, every page fetched so far is still stored and indexed.
  The command then exits with code 1 and can simply be run again.
- A malformed entry is logged and skipped; it does not fail the page.

**Other commands:**

| Command | Purpose |
|---|---|
| `pulse index` | Index papers stored in Postgres but missing from Qdrant (e.g. after an interruption). |
| `pulse reindex` | Rebuild Postgres rows and the whole vector index from the raw Parquet files with DuckDB, without calling arXiv. Use it after changing the embedding model. `--from-db` re-embeds what is in Postgres instead. |
| `pulse stats` | Summarise the raw landing zone with DuckDB (runs, records, papers, top categories). |

With Docker, prefix commands with `docker compose exec api`.

## 5. API usage

Full schema at `/docs` (Swagger UI) and `/openapi.json`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/papers` | Submit a paper or manuscript |
| `GET` | `/api/v1/papers/{id}` | Get a submitted paper's metadata |
| `GET` | `/api/v1/posts/{id}/similar?limit=10` | Similar indexed papers (required endpoint) |
| `GET` | `/api/v1/papers/{id}/similar?limit=10` | Alias of the above (see [Assumptions](#assumptions)) |
| `GET` | `/health` | Liveness: the process is up |
| `GET` | `/ready` | Readiness: Postgres and Qdrant reachable |
| `GET` | `/stats` | Papers stored, papers indexed, vectors in Qdrant |
| `GET` | `/metrics` | Prometheus metrics |

### Submit a paper

```http
POST /api/v1/papers
Content-Type: application/json

{
  "title": "Learning to rank for product search",
  "text": "We propose a neural ranking model for e-commerce search that combines dense retrieval with user click feedback to improve recommendation relevance."
}
```

```http
HTTP/1.1 201 Created
Location: /api/v1/papers/798a8ca1-0e91-4959-8bb4-bffb103d2ce6

{
  "id": "798a8ca1-0e91-4959-8bb4-bffb103d2ce6",
  "status": "created",
  "chunks": 1,
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "created_at": "2026-11-06T11:37:18.191259Z"
}
```

Request rules:
- `text` is required, must not be blank, and may be at most 100,000 characters.
- `title` is optional.
- Unknown fields are rejected.
- `chunks` reports how many ~512-token windows the text was split into.

### Get similar papers

```http
GET /api/v1/posts/798a8ca1-0e91-4959-8bb4-bffb103d2ce6/similar?limit=2
```

```json
{
  "paper_id": "798a8ca1-0e91-4959-8bb4-bffb103d2ce6",
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "limit": 2,
  "matches": [
    {
      "arxiv_id": "2611.05908",
      "title": "Optimal Re-Ranking Depth",
      "authors": ["Siqing Huo", "Andrew Parry", "Debasis Ganguly", "Charles L. A. Clarke"],
      "url": "https://arxiv.org/abs/2611.05908",
      "published": "2026-11-04T04:00:27Z",
      "primary_category": "cs.IR",
      "score": 0.792359,
      "matched_chunk": 0
    },
    {
      "arxiv_id": "2611.07591",
      "title": "Beyond Independent Predictions: Learning the Browsing Journey for Sequence-Level Re-ranking",
      "authors": ["Jiayi Zhang", "Jingxin Hai", "Zhenjia Ma", "Hongbin Pei", "Dagui Chen", "Han Zhu", "Jian Xu"],
      "url": "https://arxiv.org/abs/2611.07591",
      "published": "2026-11-05T07:41:22Z",
      "primary_category": "cs.IR",
      "score": 0.789156,
      "matched_chunk": 0
    }
  ]
}
```

- **Ordering:** `score` descending, ties broken by `arxiv_id`, so results are deterministic.
- **`limit`:** 1–50, default 10.
- **`matched_chunk`:** which chunk of the submission produced the match. It is always 0 for
  short texts.

### Errors

Every error is an [RFC 9457](https://www.rfc-editor.org/rfc/rfc9457) `application/problem+json`
document. Its `request_id` matches the `X-Request-ID` response header and the server logs.

| Status | When |
|---|---|
| `422` | Invalid body (missing or blank `text`, wrong types, unknown fields, malformed JSON, text too long), malformed id (not a UUID), `limit` out of range |
| `404` | Well-formed id that does not exist |
| `413` | Request body larger than the submission limit allows; rejected before the JSON is parsed |
| `503` (+ `Retry-After`) | Postgres or Qdrant unreachable |
| `500` | Unexpected error; no internals are exposed, details go to the logs |

```json
{
  "type": "https://paper-similarity.example/problems/validation-error",
  "title": "Invalid request",
  "status": 422,
  "detail": "The request did not pass validation; see 'errors'.",
  "instance": "/api/v1/papers",
  "request_id": "10fe40c91fc44926896b53845bca58b3",
  "errors": [{"loc": ["body", "text"], "msg": "Value error, text must not be blank", "type": "value_error"}]
}
```

### Logs

Logs are one JSON object per line (`PULSE_LOG_FORMAT=console` for development). Each request
gets an access line with method, path, status and duration. Every line written while handling
a request carries its `request_id`. A caller-supplied `X-Request-ID` is propagated.

## 6. Running the tests

```bash
uv run pytest                     # everything: unit + integration (needs Docker running)
uv run pytest tests/unit          # unit tests only, no Docker, about 3 s
uv run pytest -m model            # opt-in smoke test with the real embedding model (downloads ~70 MB)
uv run pytest --cov=paper_similarity --cov-report=term-missing
```

- **No network:**
  - arXiv responses are recorded fixtures served through respx (real responses captured
    from the API, plus a hand-written malformed feed).
  - Embeddings come from a deterministic hashing embedder. Texts that share words get higher
    cosine scores, so ranking assertions are meaningful.
- **Real databases:** integration tests start throwaway Postgres and Qdrant containers
  (testcontainers). The schema is created by the Alembic migrations, so the migrations are
  tested too.

What is covered, chosen by risk:

| Area | Tests |
|---|---|
| arXiv parsing | id/version normalisation (new and old style), whitespace, malformed entries skipped, in-band errors, empty results |
| arXiv client | retry on 503/timeouts/in-band errors, giving up after N attempts, no retry on 400, pagination, polite delay |
| Ingestion (real DB) | no duplicates on re-ingest, re-embedding only on text change, partial arXiv failure keeps earlier pages, healing an interrupted run, reindex from Parquet without arXiv |
| Chunking | single chunk when the text fits, overlap, spans map back to the source, no window exceeds the model limit |
| Ranking | best-chunk aggregation, deterministic ties, a randomised check that per-chunk top-k equals brute-force top-k |
| API (real DB) | status codes and problem bodies for every error, ranking order, limit, alias path, multi-topic matching, persistence across an app restart, 503 when Postgres or Qdrant is down, health/ready/stats/metrics |

Test limitations:
- Ranking quality with the real model is not asserted beyond a smoke test (related > unrelated).
  A relevance evaluation set would be the next step.
- The CLI is tested for output, exit codes and argument validation; a full CLI ingestion
  against live arXiv is verified manually (the pipeline it runs has integration tests).

## 7. Architecture overview

```mermaid
flowchart LR
    subgraph Ingestion ["pulse ingest (CLI / CronJob)"]
        A[arXiv API] -->|Atom XML| P[parser]
        P --> R[(Parquet<br/>raw landing)]
        P --> U[upsert]
    end
    U --> PG[(PostgreSQL<br/>source of truth)]
    PG -->|pending rows| E[embedder<br/>bge-small ONNX]
    E --> Q[(Qdrant<br/>vector index)]
    R -.->|pulse reindex via DuckDB| U

    subgraph API ["FastAPI"]
        S[POST /papers] -->|chunk + embed| PG
        G[GET /posts/id/similar] -->|load chunk vectors| PG
        G -->|batch search| Q
        G -->|metadata| PG
    end
```

Code layout (`src/paper_similarity/`):

| Package | Responsibility |
|---|---|
| `api/` | App factory and lifespan, routes, request/response schemas, problem+json errors, request-id middleware |
| `services/` | Use cases: submission and similarity, indexing (the only dual write), reindex, ranking |
| `ingestion/` | arXiv HTTP client, Atom parser, Parquet writer, DuckDB reader, ingestion pipeline |
| `embeddings/` | `Embedder` protocol, fastembed implementation, token-aware chunking |
| `vectorstore/` | Qdrant wrapper (collection, upsert, batch search, error translation) |
| `db/` | SQLAlchemy models, repositories (all SQL lives here), engine/session |
| `domain/` | Framework-free dataclasses, the embedding-text rule, domain errors |
| `core/` | Settings, logging, metrics |
| `cli.py`, `bootstrap.py` | `pulse` commands; wiring shared by the CLI and the API |

The dependency direction is api/cli → services → (db, vectorstore, embeddings, ingestion) →
domain. Routes contain no business logic, and services know nothing about HTTP.

## 8. Technical decisions and trade-offs

Short versions here; the reasoning and rejected alternatives are in
[docs/architecture.md](docs/architecture.md).

**Embedding model: `BAAI/bge-small-en-v1.5` via fastembed (ONNX).**
- It is a strong general-purpose English model for its size (384 dimensions, 33M parameters)
  and runs on a CPU: about 40 ms per abstract submission end to end.
- No API key and no cost, so reviewers can run it as-is.
- ONNX Runtime instead of PyTorch keeps the image several GB smaller.
- *Trade-off:* it is not trained specifically on science. SPECTER2 (trained on citation links)
  would likely rank scientific relatedness better but needs PyTorch and adapters.
- The `Embedder` protocol makes swapping models a one-class change, plus `pulse reindex`.

**Text used for embedding:** `"<title>. <abstract>"`, with whitespace and arXiv's hard line
breaks normalised. The same rule (`domain/text.py`) builds the text for submissions
(`"<title>. <text>"`), so both sides are built the same way. No query prefix is used: this is
document-to-document similarity, not a short query against passages.

**Storage: Postgres (source of truth) + Qdrant (derived index).**
- Postgres holds what must not be lost and needs constraints: paper metadata with a unique
  arXiv id, submissions, migrations.
- Qdrant does what it is built for: fast vector search (HNSW), payload filters, and scaling
  independently of the relational database.
- The index can always be rebuilt from Postgres or from the Parquet files.
- *Rejected:* pgvector, which would be simpler (one store, transactional), but MDPI's stack
  names Qdrant and search would compete with OLTP load on one database. A Qdrant-only design
  would give up relational integrity and migrations.

**Similarity metric: cosine** on L2-normalised vectors. That is the same ranking as dot
product and Euclidean distance, and it is the metric bge was trained for.

**Long manuscripts: chunk, search with every chunk, keep each paper's best score ("MaxP").**
- The model reads 512 tokens and silently ignores the rest. Text is split into overlapping
  token windows using the model's own tokenizer.
- Every chunk queries Qdrant, all in one batch request. A paper's score is its best chunk
  score.
- Averaging chunk vectors was rejected: it blurs a multi-topic manuscript into a vector that
  matches nothing well.
- Short texts are one chunk, which is identical to plain search.
- Fetching the top `limit` per chunk loses nothing compared with ranking every paper against
  every chunk (proved in `services/ranking.py` and checked by a randomised test). That is a
  statement about the aggregation: each per-chunk search is as exact as Qdrant makes it,
  which means exact for a small collection and approximate (HNSW) for a large one.

**Submission vectors live in Postgres, not Qdrant.**
- Submissions are only ever read back by id, never searched. So `POST /papers` is a single
  database transaction, with no dual write and nothing to reconcile on the request path.
- The only dual write is in batch indexing, where the order "row pending → Qdrant upsert →
  mark indexed" makes every failure safe to retry.
- The production-grade alternative is the transactional outbox pattern with a worker. It is
  noted, but not worth the machinery here.

**Ingestion as a CLI command**, not a startup task or API endpoint:
- it is explicit and safe to re-run;
- it doesn't block or slow down API startup;
- it doesn't expose a costly, abusable endpoint;
- it maps directly to a Kubernetes CronJob.

The arXiv client is written on httpx rather than the `arxiv` package, to control timeouts,
retries and arXiv's quirks explicitly.

**Raw landing zone in Parquet, read with DuckDB.**
- Every run keeps the records it parsed from arXiv: an audit trail and the ability to replay.
  `pulse reindex` rebuilds everything after a model change without calling arXiv again.
- The files hold *parsed* records, not the original Atom XML, and entries the parser skipped
  are not in them. Replaying after a parser fix would need the XML responses stored as well.
- It is the usual "bronze layer" of a data pipeline, kept deliberately small.
- dbt was considered and left out: there is a single transformation (latest version per
  paper). dbt would fit once there are several SQL models on top of the raw data.

**Score interpretation:**
- `score` is cosine similarity in [-1, 1]; higher means more similar. It is meaningful for
  ranking within one model, not as a probability, and not comparable across models.
- With bge, scores are compressed into the upper range (the embedding space is anisotropic):
  in the demo corpus, clearly related papers scored about 0.75–0.85 and same-field-but-different
  papers about 0.70.
- Any "is this relevant?" threshold should be calibrated on labelled examples.

**Operations:**
- Structured JSON logs with request ids, and RFC 9457 errors.
- `/health` (liveness, no dependencies) vs `/ready` (dependencies), matching Kubernetes
  probe semantics. Index counts are at `/stats`, not in the probe: exact counts scan the
  table and the collection, which is too much work to repeat every few seconds.
- Prometheus metrics per route template, plus ingestion and embedding counters.
- Migrations run as a separate one-off job before the API starts.
- One uvicorn worker per container, because the model lives in process memory; scale with
  replicas.
- CPU-bound tokenising and embedding run in a worker thread so the event loop keeps serving
  other requests (ONNX Runtime releases the GIL).
- At most `PULSE_EMBEDDING_CONCURRENCY` requests embed at once and the rest queue: one
  inference already uses every core, so running more in parallel only slows all of them.

### Assumptions

- **Path naming:** the assignment uses `POST /api/v1/papers` but `GET
  /api/v1/posts/:id/similar`. The required path is implemented exactly as written, and
  `/api/v1/papers/{id}/similar` is provided as an equivalent alias for consistency.
- **Status codes:** a malformed id returns 422 (validation failure), a well-formed unknown id
  returns 404.
- **Versions:** a paper's versions are one record (the latest version wins); old versions are
  not kept separately.
- **What is compared:** submissions are compared only with indexed arXiv papers, not with other
  submissions.

## 9. Known limitations

- **Dominant topic crowds out the rest.** Best-chunk ranking is global: in a two-topic
  manuscript, the topic whose chunk scores higher fills the top results. In a test with a
  cosmology + protein manuscript, the first protein match was ranked 21st. A diversity step
  (interleaving per-chunk results, or MMR) would fix this.
- **Asymmetry:** arXiv papers are represented by title + abstract only, while a submission may
  be a full manuscript. Method and related-work sections can pull in papers that are cited
  rather than truly similar.
- **General-purpose model:** bge-small is not domain-tuned. Exact term matches (gene names,
  acronyms) are not boosted; hybrid search (BM25 + dense, which Qdrant supports) would help.
  There is no reranking stage (a cross-encoder over the top candidates).
- **Reindex is in place:** during `pulse reindex` the collection is rebuilt and results are
  incomplete. In production, build a new collection and switch with a Qdrant alias.
- **Submissions** are stored indefinitely (no retention or deletion endpoint), and there is no
  authentication or rate limiting (out of scope per the assignment).
- **Synchronous embedding:** `POST /papers` embeds within the request. Measured in Docker on a
  laptop CPU: about 40 ms for an abstract, about 6.5 s for a 100,000-character manuscript
  (33 chunks). For heavy traffic or long texts, return 202 and embed in a worker.
- **CLI metrics:** the ingestion CLI's metrics are not scraped because it is a short-lived
  process (a Pushgateway would fix that); its results are in its logs and summary line.
- **Throughput:** ingesting 90 papers takes about a minute in Docker on a laptop, mostly
  embedding on CPU and arXiv's requested request spacing.
- **Partial result count:** if Qdrant returns hits whose Postgres row is missing (only possible
  after an interrupted dual write), they are dropped. A small over-fetch usually still fills
  `limit`.

## 10. Use of AI tools

I used **Claude Code** (Anthropic) as a pair-programming assistant throughout:

- **Design discussion:** comparing stack options (pgvector vs Qdrant, chunk averaging vs
  best-chunk ranking, where to store submission vectors) before writing code. I made the final
  decisions.
- **Implementation:** drafting modules, tests and documentation, one commit-sized step at a
  time.
- **Investigation:** recording real arXiv responses (including its error formats) to base the
  parser and retry logic on observed behaviour.

How I validated the output:

- I read and reviewed every change before committing.
- Every step had to pass ruff, mypy `--strict` and the test suite, including integration tests
  against real Postgres and Qdrant.
- I added tests for the claims that matter, e.g. a randomised check of the top-k ranking
  argument, re-ingest idempotency, and dependency outages.
- I ran the full workflow manually with Docker against the live arXiv API: ingesting 90 papers,
  re-ingesting without duplicates, submitting, querying, restarting, and checking error
  responses.
- Several issues were caught this way, and the fixes are in the history: a test mock
  intercepting Qdrant traffic, DuckDB returning local-time timestamps, asyncpg connection
  errors not mapped to 503, and a port clash with a local Postgres.
