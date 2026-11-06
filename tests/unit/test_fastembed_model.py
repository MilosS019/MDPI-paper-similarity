"""Smoke test against the real embedding model. Opt-in: ``pytest -m model`` (downloads ~70 MB)."""

import math

import pytest

from paper_similarity.embeddings.chunking import chunk_text
from paper_similarity.embeddings.fastembed_embedder import FastEmbedEmbedder

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def embedder() -> FastEmbedEmbedder:
    return FastEmbedEmbedder("BAAI/bge-small-en-v1.5")


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_vectors_are_normalised_and_have_expected_dimension(embedder: FastEmbedEmbedder) -> None:
    [vector] = embedder.embed(["Dense passage retrieval for open-domain question answering."])
    assert len(vector) == embedder.dimension == 384
    assert math.isclose(math.sqrt(dot(vector, vector)), 1.0, rel_tol=1e-5)


def test_related_texts_score_higher_than_unrelated(embedder: FastEmbedEmbedder) -> None:
    query, related, unrelated = embedder.embed(
        [
            "Neural retrieval models for searching scientific literature.",
            "Dense vector search over research papers using transformer encoders.",
            "Protein folding dynamics in yeast cells under heat stress.",
        ]
    )
    assert dot(query, related) > dot(query, unrelated)


def test_long_text_is_chunked_with_the_real_tokenizer(embedder: FastEmbedEmbedder) -> None:
    text = "Transformers improve retrieval of scientific documents. " * 300
    offsets = embedder.token_offsets(text)
    assert len(offsets) > 512  # not truncated

    chunks = chunk_text(text, embedder.token_offsets, max_tokens=512, overlap_tokens=64)
    assert len(chunks) > 1
    for chunk in chunks:
        # +2 special tokens: every window fits the model exactly.
        assert len(embedder.token_offsets(chunk.text)) + 2 <= 512
