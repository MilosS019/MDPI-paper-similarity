from pathlib import Path

import pytest

from paper_similarity.core.config import Settings


def test_empty_cache_dir_means_library_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PULSE_EMBEDDING_CACHE_DIR", "")
    assert Settings(_env_file=None).embedding_cache_dir is None


def test_cache_dir_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PULSE_EMBEDDING_CACHE_DIR", "/models")
    assert Settings(_env_file=None).embedding_cache_dir == Path("/models")
