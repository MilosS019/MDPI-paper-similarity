import itertools

import pytest
from pydantic import ValidationError

from paper_similarity.core.config import Settings
from paper_similarity.embeddings.chunking import SPECIAL_TOKENS, chunk_text, effective_chunk_window
from tests.fakes import HashingEmbedder

offsets = HashingEmbedder().token_offsets


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_text_that_fits_is_a_single_chunk_covering_everything() -> None:
    text = words(510)  # exactly the content window for max_tokens=512

    chunks = chunk_text(text, offsets, max_tokens=512, overlap_tokens=64)

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert (chunks[0].char_start, chunks[0].char_end) == (0, len(text))


def test_long_text_is_split_into_overlapping_windows() -> None:
    text = words(30)
    max_tokens, overlap = 10 + SPECIAL_TOKENS, 3  # 10 content tokens per window, step 7

    chunks = chunk_text(text, offsets, max_tokens=max_tokens, overlap_tokens=overlap)

    token_lists = [c.text.split() for c in chunks]
    assert [len(t) for t in token_lists] == [10, 10, 10, 9]
    assert token_lists[0][0] == "w0"
    assert token_lists[-1][-1] == "w29"  # nothing is lost at the end
    for previous, current in itertools.pairwise(token_lists):
        assert previous[-overlap:] == current[:overlap]
    assert [c.index for c in chunks] == [0, 1, 2, 3]


def test_char_spans_point_back_into_the_source_text() -> None:
    text = "alpha beta  gamma\n delta epsilon zeta eta theta"

    chunks = chunk_text(text, offsets, max_tokens=4 + SPECIAL_TOKENS, overlap_tokens=1)

    for chunk in chunks:
        assert text[chunk.char_start : chunk.char_end] == chunk.text
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(text)


def test_no_window_exceeds_the_model_limit() -> None:
    chunks = chunk_text(words(5000), offsets, max_tokens=512, overlap_tokens=64)

    assert all(len(c.text.split()) <= 512 - SPECIAL_TOKENS for c in chunks)
    assert len(chunks) == 12  # ceil((5000 - 510) / 446) + 1


@pytest.mark.parametrize(("max_tokens", "overlap"), [(2, 0), (12, 10), (12, -1)])
def test_invalid_parameters_are_rejected(max_tokens: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("some text", offsets, max_tokens=max_tokens, overlap_tokens=overlap)


def test_effective_window_is_capped_at_the_model_limit() -> None:
    assert effective_chunk_window(2048, model_limit=512, overlap=64) == 512
    assert effective_chunk_window(256, model_limit=512, overlap=64) == 256


@pytest.mark.parametrize(("requested", "overlap"), [(128, 126), (128, 127), (2048, 510)])
def test_effective_window_rejects_overlap_that_fills_the_window(
    requested: int, overlap: int
) -> None:
    with pytest.raises(ValueError, match="overlap"):
        effective_chunk_window(requested, model_limit=512, overlap=overlap)


@pytest.mark.parametrize("overlap", [126, 127])
def test_settings_reject_overlap_that_leaves_no_new_tokens(overlap: int) -> None:
    with pytest.raises(ValidationError, match="chunk_overlap_tokens"):
        Settings(chunk_max_tokens=128, chunk_overlap_tokens=overlap, _env_file=None)
