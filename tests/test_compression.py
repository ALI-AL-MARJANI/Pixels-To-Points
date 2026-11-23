import numpy as np
import pytest
from scipy.stats import spearmanr

from p2p import binarize, muvera, pooling
from p2p.scoring import hamming_maxsim_matrix, maxsim, maxsim_matrix


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def synthetic(seed: int = 0, n_docs: int = 60, dim: int = 32):
    """Each page = 6 private 'content' directions, each repeated 8x with noise
    (redundant patches, like a real page). Query i = 4 noisy copies of page i's
    directions, so page i is the right answer for query i."""
    rng = np.random.default_rng(seed)
    centers = [unit(rng.normal(size=(6, dim))) for _ in range(n_docs)]
    docs = [unit(c.repeat(8, 0) + 0.5 * rng.normal(size=(48, dim)) / np.sqrt(dim)) for c in centers]
    queries = [unit(c[rng.choice(6, 4, replace=False)] + 0.5 * rng.normal(size=(4, dim)) / np.sqrt(dim))
               for c in centers[:20]]
    return queries, docs


def test_pooling_factor_one_is_identity():
    doc = unit(np.random.default_rng(0).normal(size=(10, 8)))
    assert np.array_equal(pooling.hierarchical_pool(doc, 1), doc)


def test_pooling_sizes_and_duplicates():
    rng = np.random.default_rng(1)
    base = unit(rng.normal(size=(4, 8)))
    doc = np.repeat(base, 5, axis=0)                     # 4 distinct vectors, 5 copies each
    pooled = pooling.hierarchical_pool(doc, 5)
    assert len(pooled) == 4
    # exact duplicates must be merged together: pooled set == the 4 originals
    assert np.allclose(np.sort(pooled, axis=0), np.sort(base, axis=0), atol=1e-6)
    assert len(pooling.random_drop(doc, 4, rng)) == 5


def test_pooling_beats_random_drop():
    queries, docs = synthetic()
    full = maxsim_matrix(queries, docs)
    pooled = maxsim_matrix(queries, [pooling.hierarchical_pool(d, 8) for d in docs])
    rng = np.random.default_rng(0)
    dropped = maxsim_matrix(queries, [pooling.random_drop(d, 8, rng) for d in docs])
    corr = lambda m: spearmanr(full.ravel(), m.ravel()).statistic
    assert corr(pooled) > corr(dropped)


def test_binary_scores_correlate_with_full():
    queries, docs = synthetic()
    full = maxsim_matrix(queries, docs)
    center = binarize.fit_center(docs)
    qb = [binarize.pack_signs(q, center) for q in queries]
    db = [binarize.pack_signs(d, center) for d in docs]
    ham = hamming_maxsim_matrix(qb, db, dim=32)
    # 1 bit per dimension is coarse, so the global correlation is only moderate...
    assert spearmanr(full.ravel(), ham.ravel()).statistic > 0.4
    # ...but the right page still wins for (almost) every query.
    assert np.mean([np.argmax(ham[i]) == i for i in range(len(queries))]) >= 0.9
    # packing / unpacking round trip keeps the signs
    v = np.array([[0.5, -1.0, 2.0, -0.1, 0.0, 3.0, -2.0, 1.0]])
    assert np.array_equal(binarize.unpack_signs(binarize.pack_signs(v), 8)[0],
                          [1, -1, 1, -1, -1, 1, -1, 1])


def test_rescore_puts_exact_scores_on_top():
    queries, docs = synthetic()
    noisy = maxsim_matrix(queries, docs) + np.random.default_rng(0).normal(size=(20, 60))
    out = binarize.rescore(noisy, queries, docs, top=10)
    i = 0
    top = np.argsort(-out[i])[:10]
    assert set(top) == set(np.argsort(-noisy[i])[:10])
    assert out[i, top[0]] == pytest.approx(max(maxsim(queries[i], docs[j]) for j in top))


def test_fde_dot_product_approximates_maxsim_ranking():
    queries, docs = synthetic()
    p = muvera.make_params(dim=32, k_sim=4, d_proj=16, reps=20)
    fq = muvera.encode_many(queries, p, is_query=True)
    fd = muvera.encode_many(docs, p, is_query=False)
    assert fd.shape == (60, p.size)
    full = maxsim_matrix(queries, docs)
    approx = fq @ fd.T
    rhos = [spearmanr(full[i], approx[i]).statistic for i in range(len(queries))]
    assert np.mean(rhos) > 0.5
    # the true best page is usually in the FDE top-10
    hits = [np.argmax(full[i]) in np.argsort(-approx[i])[:10] for i in range(len(queries))]
    assert np.mean(hits) >= 0.8


def test_fde_matches_fastembed_reference_quality():
    """Same parameters as Qdrant FastEmbed's Muvera: ranking quality must be close.
    (Random draws differ, so we compare quality, not raw vectors.)"""
    fe = pytest.importorskip("fastembed.postprocess")
    queries, docs = synthetic(seed=3)
    full = maxsim_matrix(queries, docs)
    ref = fe.Muvera(dim=32, k_sim=4, dim_proj=16, r_reps=20, random_seed=0)
    ref_scores = np.stack([ref.process_query(q) for q in queries]) @ \
        np.stack([ref.process_document(d) for d in docs]).T
    p = muvera.make_params(dim=32, k_sim=4, d_proj=16, reps=20, seed=0)
    ours = muvera.encode_many(queries, p, True) @ muvera.encode_many(docs, p, False).T
    rho = lambda s: np.mean([spearmanr(full[i], s[i]).statistic for i in range(len(queries))])
    assert len(ref.process_document(docs[0])) == p.size
    assert abs(rho(ours) - rho(ref_scores)) < 0.1
