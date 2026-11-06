import random

from paper_similarity.domain.models import VectorHit
from paper_similarity.services.ranking import aggregate_max


def hits(*pairs: tuple[str, float]) -> list[VectorHit]:
    return [VectorHit(arxiv_id=a, score=s) for a, s in pairs]


def test_keeps_the_best_score_per_paper_and_remembers_the_chunk() -> None:
    ranked = aggregate_max(
        [
            hits(("a", 0.70), ("b", 0.60)),
            hits(("b", 0.90), ("c", 0.50)),
        ]
    )

    assert [(h.arxiv_id, h.score, h.chunk_index) for h in ranked] == [
        ("b", 0.90, 1),
        ("a", 0.70, 0),
        ("c", 0.50, 1),
    ]


def test_ties_are_ordered_by_arxiv_id_and_keep_the_first_chunk() -> None:
    ranked = aggregate_max([hits(("z", 0.8), ("m", 0.8)), hits(("m", 0.8))])

    assert [h.arxiv_id for h in ranked] == ["m", "z"]
    assert ranked[0].chunk_index == 0


def test_single_chunk_is_plain_ranking() -> None:
    ranked = aggregate_max([hits(("a", 0.9), ("b", 0.8))])
    assert [h.arxiv_id for h in ranked] == ["a", "b"]


def test_empty_input() -> None:
    assert aggregate_max([]) == []
    assert aggregate_max([[], []]) == []


def test_top_k_per_chunk_gives_the_exact_global_top_k() -> None:
    """Property check of the exactness argument in the module docstring, on random data."""
    rng = random.Random(42)
    papers = [f"p{i}" for i in range(200)]
    for _ in range(50):
        n_chunks, k = rng.randint(1, 8), rng.randint(1, 20)
        scores = [{p: rng.uniform(-1, 1) for p in papers} for _ in range(n_chunks)]

        # Brute force: best score over all chunks for every paper.
        brute = sorted(
            ((max(s[p] for s in scores), p) for p in papers), key=lambda t: (-t[0], t[1])
        )[:k]

        # What the service does: top k per chunk from the vector store, then aggregate.
        per_chunk = [
            [VectorHit(p, sc) for p, sc in sorted(s.items(), key=lambda t: -t[1])[:k]]
            for s in scores
        ]
        aggregated = aggregate_max(per_chunk)[:k]

        assert [(h.score, h.arxiv_id) for h in aggregated] == brute
