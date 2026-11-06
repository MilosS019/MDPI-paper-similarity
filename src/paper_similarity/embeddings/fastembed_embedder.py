"""Local embedding model via fastembed (ONNX Runtime): no API keys, no GPU, small image."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import structlog
from fastembed import TextEmbedding
from tokenizers import Tokenizer

log = structlog.get_logger(__name__)


class FastEmbedEmbedder:
    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: Path | None = None,
        batch_size: int = 32,
        max_tokens: int = 512,
    ) -> None:
        log.info("embedding_model_loading", model=model_name)
        self._model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        self._model_name = model_name
        self._batch_size = batch_size
        self._max_tokens = max_tokens
        self._dimension = int(TextEmbedding.get_embedding_size(model_name))
        self._tokenizer = self._load_untruncated_tokenizer()
        log.info("embedding_model_loaded", model=model_name, dimension=self._dimension)

    def _load_untruncated_tokenizer(self) -> Tokenizer:
        # fastembed truncates at 512 tokens internally. For chunking we need offsets of the
        # *whole* text, so load the same tokenizer.json from the downloaded model files and
        # switch truncation and padding off. (``_model_dir`` is not public API; the fastembed
        # version is pinned in uv.lock and this is covered by the model smoke test.)
        model_dir = Path(self._model.model._model_dir)  # type: ignore[attr-defined]
        tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        return tokenizer

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = np.asarray(list(self._model.embed(list(texts), batch_size=self._batch_size)))
        # bge already returns unit vectors; normalising again is cheap and guarantees that
        # cosine similarity equals the dot product whatever model is configured.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalised = vectors / np.clip(norms, 1e-12, None)
        return [row.astype(float).tolist() for row in normalised]

    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        return [(int(start), int(end)) for start, end in encoding.offsets]
