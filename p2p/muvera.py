"""MUVERA Fixed Dimensional Encodings (Dhulipala et al., 2024), minimal version.

Goal: turn a SET of vectors into ONE vector so that
    <FDE_query(Q), FDE_doc(D)>  ~  MaxSim(Q, D).

Recipe, repeated `reps` times with fresh randomness, then concatenated:
  1. Partition space with k_sim random hyperplanes (SimHash): a vector's cluster
     id is the k_sim-bit pattern "which side of each hyperplane am I on?".
     Close vectors usually land in the same cluster (2^k_sim clusters).
  2. Aggregate per cluster: SUM of query vectors, MEAN of page vectors.
     Inside one cluster, a query token mostly meets the page vectors closest to
     it, so <sum q, mean d> imitates "each query token picks its best match".
  3. Empty page clusters are filled with the page vector whose cluster id is
     closest in Hamming distance (so no query token scores 0 by bad luck).
  4. Randomly project each cluster block from d to d_proj dims (±1/sqrt(d_proj)).
Final size = reps * 2^k_sim * d_proj.
"""

from dataclasses import dataclass

import numpy as np

from p2p.scoring import POPCOUNT  # bit counts of 0..255 (k_sim <= 8)


@dataclass
class FDEParams:
    """Random hyperplanes and projections, shared by queries and pages."""
    planes: np.ndarray   # (reps, dim, k_sim)  Gaussian hyperplanes
    proj: np.ndarray     # (reps, dim, d_proj) ±1/sqrt(d_proj) projection

    @property
    def k_sim(self) -> int:
        return self.planes.shape[2]

    @property
    def size(self) -> int:
        reps, _, d_proj = self.proj.shape
        return reps * (2 ** self.k_sim) * d_proj


def make_params(dim: int, k_sim: int = 5, d_proj: int = 16, reps: int = 20,
                seed: int = 42) -> FDEParams:
    """Defaults match the Qdrant FastEmbed reference (k_sim=5, d_proj=16, r=20)."""
    rng = np.random.default_rng(seed)
    planes = rng.normal(size=(reps, dim, k_sim)).astype(np.float32)
    proj = rng.choice([-1.0, 1.0], size=(reps, dim, d_proj)).astype(np.float32)
    return FDEParams(planes, proj / np.sqrt(d_proj))


def cluster_ids(x: np.ndarray, planes: np.ndarray) -> np.ndarray:
    """(n, dim) -> (n,) integer cluster id from the sign pattern on k_sim planes."""
    bits = (x @ planes > 0).astype(np.int64)            # (n, k_sim)
    return bits @ (1 << np.arange(planes.shape[1]))      # binary -> int


def encode(x: np.ndarray, p: FDEParams, is_query: bool) -> np.ndarray:
    """FDE of one multi-vector set (n, dim). Query: sum per cluster, no filling.
    Page: mean per cluster + empty-cluster filling."""
    x = x.astype(np.float32)
    n_clusters = 2 ** p.k_sim
    blocks = []
    for r in range(p.planes.shape[0]):
        ids = cluster_ids(x, p.planes[r])
        agg = np.zeros((n_clusters, x.shape[1]), dtype=np.float32)
        np.add.at(agg, ids, x)                            # sum per cluster
        counts = np.bincount(ids, minlength=n_clusters)
        if not is_query:
            filled = counts > 0
            agg[filled] /= counts[filled, None]           # sum -> mean
            for c in np.flatnonzero(~filled):             # fill empty clusters
                agg[c] = x[np.argmin(POPCOUNT[ids ^ c])]
        blocks.append(agg @ p.proj[r])                    # (n_clusters, d_proj)
    return np.concatenate(blocks).ravel()


def encode_many(sets: list[np.ndarray], p: FDEParams, is_query: bool) -> np.ndarray:
    """Stack the FDEs of many sets into a (n_sets, size) matrix."""
    return np.stack([encode(s, p, is_query) for s in sets])
