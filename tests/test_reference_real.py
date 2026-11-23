"""Check my implementations against colpali-engine on REAL embeddings.

Skipped when embeddings/ is missing (e.g. on CI). Run after encoding:
    pytest tests/test_reference_real.py -s
"""

from pathlib import Path

import numpy as np
import pytest

from p2p.metrics import evaluate, scores_to_run
from p2p.pooling import hierarchical_pool
from p2p.run_experiments import load_embeddings
from p2p.scoring import maxsim_matrix

ROOT = Path("embeddings/colsmol/economics_v2")
pytestmark = pytest.mark.skipif(not ROOT.exists(), reason="no saved embeddings")


@pytest.fixture(scope="module")
def data():
    return load_embeddings(ROOT)


def test_maxsim_matches_colpali_engine(data):
    torch = pytest.importorskip("torch")
    from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor as P
    pages, queries, _ = data
    pages, queries = pages[:60], queries[:10]
    ref = P.score_multi_vector([torch.from_numpy(q.astype(np.float32)) for q in queries],
                               [torch.from_numpy(p.astype(np.float32)) for p in pages]).numpy()
    np.testing.assert_allclose(maxsim_matrix(queries, pages), ref, rtol=1e-4, atol=1e-3)


def test_pooling_close_to_reference_pooler(data):
    """The reference clusters rows of the (1 - cos) matrix with Ward and keeps
    floor(n / f) clusters; mine runs Ward on the vectors and keeps ceil(n / f).
    Retrieval quality must be close."""
    torch = pytest.importorskip("torch")
    from colpali_engine.compression.token_pooling import HierarchicalTokenPooler
    pages, queries, meta = data
    pooler = HierarchicalTokenPooler()
    for f in (4, 16):
        ref = pooler.pool_embeddings([torch.from_numpy(p.astype(np.float32)) for p in pages],
                                     pool_factor=f)
        ref = [r.numpy() for r in ref]
        mine = [hierarchical_pool(p, f) for p in pages]
        score = lambda docs: evaluate(scores_to_run(maxsim_matrix(queries, docs), meta["query_ids"],
                                                    meta["page_ids"]), meta["qrels"])["ndcg@5"]
        s_ref, s_mine = score(ref), score(mine)
        print(f"pool x{f}: reference nDCG@5={s_ref:.3f}  mine={s_mine:.3f}")
        assert abs(s_ref - s_mine) < 0.05
