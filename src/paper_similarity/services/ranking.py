"""Combine per-chunk search results into one ranking of papers ("MaxP" aggregation).

A long manuscript is searched chunk by chunk. A paper's score is the best score any chunk
achieved against it: the paper ranks high if *some* passage of the manuscript is strongly
related to it. Averaging chunk vectors instead would blur a multi-topic manuscript into a
vector that matches nothing well.

Exactness: if every chunk query returns its top ``k`` papers, the top ``k`` of the aggregated
ranking is exact. Suppose paper P belongs in the global top k with best chunk c, but P is
missing from c's top k. Then k papers beat P on chunk c, so each of them has a best score
above P's, and P is not in the global top k: a contradiction.
"""

from collections.abc import Sequence

from paper_similarity.domain.models import ChunkHit, VectorHit


def aggregate_max(hits_per_chunk: Sequence[Sequence[VectorHit]]) -> list[ChunkHit]:
    """Best score per paper across all chunks, sorted by score descending, then arxiv_id.

    The arxiv_id tie-break makes the order deterministic. On equal scores the lower chunk
    index is kept as the "matched chunk".
    """
    best: dict[str, ChunkHit] = {}
    for chunk_index, hits in enumerate(hits_per_chunk):
        for hit in hits:
            current = best.get(hit.arxiv_id)
            if current is None or hit.score > current.score:
                best[hit.arxiv_id] = ChunkHit(
                    arxiv_id=hit.arxiv_id, score=hit.score, chunk_index=chunk_index
                )
    return sorted(best.values(), key=lambda h: (-h.score, h.arxiv_id))
