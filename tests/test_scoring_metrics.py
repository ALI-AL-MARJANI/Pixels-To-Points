import numpy as np
import pytest

from p2p.metrics import evaluate, ndcg_at_k, recall_at_k, scores_to_run
from p2p.scoring import maxsim, maxsim_matrix


def test_maxsim_hand_computed():
    # Query tokens: e1, e2. Page vectors: e1, 0.5*e1+0.5*e2, e3.
    q = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    d = np.array([[1, 0, 0], [0.5, 0.5, 0], [0, 0, 1]], dtype=np.float32)
    # token 1 best match = 1.0 (first vector); token 2 best match = 0.5 (second vector)
    assert maxsim(q, d) == pytest.approx(1.5)


def test_maxsim_matrix_matches_loop():
    rng = np.random.default_rng(0)
    queries = [rng.normal(size=(n, 8)) for n in (3, 5)]
    docs = [rng.normal(size=(n, 8)) for n in (4, 1, 7)]
    m = maxsim_matrix(queries, docs)
    for i, q in enumerate(queries):
        for j, d in enumerate(docs):
            assert m[i, j] == pytest.approx(maxsim(q, d), rel=1e-5)


def test_ndcg_and_recall_toy_ranking():
    rels = {"a": 1, "c": 1}
    ranked = ["a", "b", "c"]
    # DCG = 1/log2(2) + 1/log2(4) = 1.5 ; ideal = 1 + 1/log2(3)
    assert ndcg_at_k(ranked, rels, 3) == pytest.approx(1.5 / (1 + 1 / np.log2(3)))
    assert recall_at_k(ranked, rels, 1) == 0.5
    assert recall_at_k(ranked, rels, 3) == 1.0
    assert ndcg_at_k(["a", "c", "b"], rels, 3) == pytest.approx(1.0)


def test_graded_relevance_and_evaluate():
    qrels = {"q1": {"x": 2, "y": 1}, "q2": {"z": 1}}
    scores = np.array([[0.1, 0.9, 0.5], [0.0, 0.0, 1.0]])  # pages x, y, z
    run = scores_to_run(scores, ["q1", "q2"], ["x", "y", "z"])
    assert run["q1"] == ["y", "z", "x"]
    res = evaluate(run, qrels, ks=(1,))
    # q1 top-1 = y (gain 1) vs ideal x (gain 2) -> 0.5 ; q2 perfect -> 1.0
    assert res["ndcg@1"] == pytest.approx(0.75)
