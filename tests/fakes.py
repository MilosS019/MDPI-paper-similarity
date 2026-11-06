"""Test doubles."""

import math
import re
import zlib
from collections.abc import Sequence

_WORD = re.compile(r"\w+")


class HashingEmbedder:
    """Deterministic stand-in for the real model: a hashed bag of words, L2-normalised.

    Texts sharing words get higher cosine similarity, so tests can assert on meaningful
    rankings without downloading or running a neural model. Tokens are whitespace-separated
    words, which keeps chunking tests easy to reason about.
    """

    model_name = "test/hashing-embedder"
    dimension = 256
    max_tokens = 512

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for word in _WORD.findall(text.lower()):
            digest = zlib.crc32(word.encode("utf-8"))
            sign = 1.0 if digest & 1 else -1.0
            vector[(digest >> 1) % self.dimension] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            vector[0] = 1.0
            return vector
        return [v / norm for v in vector]

    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
