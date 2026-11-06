"""Shared test fixtures available to unit and integration tests."""

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_bytes() -> "FixtureLoader":
    return FixtureLoader()


class FixtureLoader:
    def __call__(self, name: str) -> bytes:
        return (FIXTURES / name).read_bytes()
