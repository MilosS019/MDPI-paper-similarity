"""Split long texts into overlapping, token-aware windows that fit the embedding model.

The model reads at most ``max_tokens`` tokens (512 for bge-small, including the two special
tokens it adds) and silently ignores the rest. Chunking makes the whole manuscript count:
each window is embedded separately and the similarity search keeps, for every indexed paper,
the best-matching chunk (see ``services.similarity``).

Windows overlap so that a sentence cut at one boundary appears whole in the next window.
"""

from collections.abc import Callable

from paper_similarity.domain.models import TextChunk

# [CLS] and [SEP] are added by the model to every input.
SPECIAL_TOKENS = 2

TokenOffsets = Callable[[str], list[tuple[int, int]]]


def chunk_text(
    text: str,
    token_offsets: TokenOffsets,
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> list[TextChunk]:
    """Return chunks covering ``text``; a text that fits the model is a single chunk.

    ``char_start``/``char_end`` index into ``text`` so every chunk can be traced back to the
    exact passage of the manuscript it came from.
    """
    window = max_tokens - SPECIAL_TOKENS
    if window <= 0:
        raise ValueError("max_tokens is too small to hold any content")
    if not 0 <= overlap_tokens < window:
        raise ValueError("overlap_tokens must be >= 0 and smaller than the content window")

    offsets = token_offsets(text)
    if len(offsets) <= window:
        return [TextChunk(index=0, text=text, char_start=0, char_end=len(text))]

    step = window - overlap_tokens
    chunks: list[TextChunk] = []
    start_token = 0
    while True:
        end_token = min(start_token + window, len(offsets))
        char_start = offsets[start_token][0]
        char_end = offsets[end_token - 1][1]
        chunks.append(
            TextChunk(
                index=len(chunks),
                text=text[char_start:char_end],
                char_start=char_start,
                char_end=char_end,
            )
        )
        if end_token == len(offsets):
            return chunks
        start_token += step
