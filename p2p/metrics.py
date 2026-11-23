"""Retrieval metrics: nDCG@k and Recall@k, written from the definitions.

`qrels` maps a query id to {page id: graded relevance}. `run` maps a query id
to a ranked list of page ids (best first).
"""

import numpy as np

Qrels = dict[str, dict[str, int]]
Run = dict[str, list[str]]


def dcg(gains: list[float]) -> float:
    """Discounted cumulative gain with the linear gain used by pytrec_eval / MTEB."""
    return float(sum(g / np.log2(rank + 2) for rank, g in enumerate(gains)))


def ndcg_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    """nDCG@k for one query: DCG of our top-k divided by DCG of the ideal top-k."""
    gains = [rels.get(pid, 0) for pid in ranked[:k]]
    ideal = sorted(rels.values(), reverse=True)[:k]
    best = dcg(ideal)
    return dcg(gains) / best if best > 0 else 0.0


def recall_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    """Fraction of the relevant pages that appear in the top-k."""
    relevant = {pid for pid, r in rels.items() if r > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def evaluate(run: Run, qrels: Qrels, ks: tuple[int, ...] = (1, 5, 10, 100)) -> dict:
    """Mean nDCG@k and Recall@k over all queries that have judgements."""
    out: dict[str, float] = {}
    qids = [q for q in qrels if q in run]
    for k in ks:
        out[f"ndcg@{k}"] = float(np.mean([ndcg_at_k(run[q], qrels[q], k) for q in qids]))
        out[f"recall@{k}"] = float(np.mean([recall_at_k(run[q], qrels[q], k) for q in qids]))
    return out


def per_query_ndcg(run: Run, qrels: Qrels, k: int) -> dict[str, float]:
    """nDCG@k for every query, used for the error analysis."""
    return {q: ndcg_at_k(run[q], qrels[q], k) for q in qrels if q in run}


def scores_to_run(scores: np.ndarray, qids: list[str], pids: list[str], top: int = 100) -> Run:
    """Turn a (n_queries, n_pages) score matrix into ranked page-id lists."""
    order = np.argsort(-scores, axis=1, kind="stable")[:, :top]
    return {q: [pids[j] for j in order[i]] for i, q in enumerate(qids)}
