"""Binarization: keep only the sign of each dimension (1 bit instead of 16).

A 128-d fp16 vector takes 256 bytes; its sign code takes 16 bytes (16x smaller).
Scoring uses Hamming distance (see `scoring.hamming_maxsim_matrix`).
"""

import numpy as np


def fit_center(docs: list[np.ndarray]) -> np.ndarray:
    """Mean page vector of the corpus. If a dimension is almost always positive,
    its sign carries no information; subtracting the mean makes bits ~50/50."""
    return np.concatenate(docs).astype(np.float32).mean(axis=0)


def pack_signs(x: np.ndarray, center: np.ndarray | None = None) -> np.ndarray:
    """(n, d) floats -> (n, d/8) uint8: bit = 1 where the value is positive."""
    x = x.astype(np.float32)
    if center is not None:
        x = x - center
    return np.packbits(x > 0, axis=1)


def unpack_signs(bits: np.ndarray, dim: int) -> np.ndarray:
    """(n, d/8) uint8 -> (n, d) float32 vectors of +1 / -1."""
    return np.unpackbits(bits, axis=1)[:, :dim].astype(np.float32) * 2 - 1


def asymmetric_maxsim_matrix(queries: list[np.ndarray], docs_bits: list[np.ndarray],
                             dim: int) -> np.ndarray:
    """Float query x binary page: the query keeps full precision (it is small and
    never stored); only the stored pages are 1-bit. Usually more accurate than
    binarizing both sides."""
    from p2p.scoring import maxsim_matrix
    return maxsim_matrix(queries, [unpack_signs(b, dim) for b in docs_bits])


def rescore(first_stage: np.ndarray, queries: list[np.ndarray], docs: list[np.ndarray],
            top: int) -> np.ndarray:
    """Re-rank the `top` best pages of each query with exact fp MaxSim.

    Pages outside the shortlist keep a score below every rescored page,
    so the final order is: rescored shortlist first, then the rest."""
    from p2p.scoring import maxsim
    out = first_stage - first_stage.max() - 1e6  # push everything below the shortlist
    for i, q in enumerate(queries):
        for j in np.argsort(-first_stage[i])[:top]:
            out[i, j] = maxsim(q, docs[j])
    return out
