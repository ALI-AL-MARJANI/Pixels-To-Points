"""Late-interaction scoring: MaxSim on float vectors and on binary codes.

A page is a (n_doc, d) matrix, a query is a (n_q, d) matrix.
MaxSim(q, D) = sum over query tokens i of max over page vectors j of <q_i, d_j>.
"""

import numpy as np

# Number of set bits for every possible byte value, used for Hamming distance.
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def maxsim(query: np.ndarray, doc: np.ndarray) -> float:
    """MaxSim score of one query (n_q, d) against one page (n_doc, d)."""
    sims = query.astype(np.float32) @ doc.astype(np.float32).T  # (n_q, n_doc)
    return float(sims.max(axis=1).sum())


def maxsim_matrix(queries: list[np.ndarray], docs: list[np.ndarray]) -> np.ndarray:
    """Scores of every query against every page, shape (n_queries, n_docs).

    All page vectors are stacked once, so the whole corpus is one matmul per query.
    """
    flat = np.concatenate(docs).astype(np.float32)          # (total_vectors, d)
    offsets = np.cumsum([0] + [len(d) for d in docs])[:-1]  # first row of each page
    scores = np.empty((len(queries), len(docs)), dtype=np.float32)
    for i, q in enumerate(queries):
        sims = q.astype(np.float32) @ flat.T                # (n_q, total_vectors)
        per_page = np.maximum.reduceat(sims, offsets, axis=1)  # max inside each page
        scores[i] = per_page.sum(axis=0)
    return scores


def hamming_maxsim_matrix(
    queries: list[np.ndarray], docs: list[np.ndarray], dim: int
) -> np.ndarray:
    """MaxSim on packed sign bits (uint8 arrays from `binarize.pack_signs`).

    For ±1 vectors of length d: <a, b> = d - 2 * hamming(a, b),
    so maximising the dot product means minimising the Hamming distance.
    """
    flat = np.concatenate(docs)                              # (total_vectors, d/8)
    offsets = np.cumsum([0] + [len(d) for d in docs])[:-1]
    scores = np.empty((len(queries), len(docs)), dtype=np.float32)
    for i, q in enumerate(queries):
        xor = np.bitwise_xor(q[:, None, :], flat[None, :, :])   # (n_q, total, d/8)
        ham = POPCOUNT[xor].sum(axis=2, dtype=np.int32)         # (n_q, total)
        sims = (dim - 2 * ham).astype(np.float32)
        scores[i] = np.maximum.reduceat(sims, offsets, axis=1).sum(axis=0)
    return scores
