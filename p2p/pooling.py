"""Token pooling (Clavié et al., 2024) and a random-dropping baseline.

Idea: many patch vectors of a page are near-duplicates (white margins, flat
backgrounds). Cluster them and keep one mean vector per cluster.
Pool factor f means: n vectors -> ceil(n / f) vectors.
"""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage


def n_clusters(n: int, factor: int) -> int:
    """Number of vectors kept for a page of n vectors at pool factor f."""
    return max(1, int(np.ceil(n / factor)))


def ward_tree(doc: np.ndarray) -> np.ndarray:
    """Agglomerative (Ward) clustering tree of the page vectors.

    Like the colpali-engine reference, each vector is described by its row of
    cosine distances to all other vectors of the page (its "similarity profile"),
    and Ward merges the two groups whose union adds the least variance.
    In my ablation this beat Ward on the raw vectors at high pool factors
    (results/ablation_pooling_colsmol_economics.json).
    The tree is built once and can be cut at any pool factor.
    """
    x = doc.astype(np.float64)
    return linkage(1.0 - x @ x.T, method="ward")


def pool_with_tree(doc: np.ndarray, tree: np.ndarray, factor: int) -> np.ndarray:
    """Cut the tree into ceil(n / f) clusters, average each cluster, and
    re-normalise the mean to unit length (a mean of different unit vectors is
    shorter than 1, which would unfairly lower its MaxSim scores)."""
    if factor <= 1 or len(doc) <= 1:
        return doc
    labels = fcluster(tree, t=n_clusters(len(doc), factor), criterion="maxclust")
    k = labels.max()
    sums = np.zeros((k, doc.shape[1]), dtype=np.float32)
    np.add.at(sums, labels - 1, doc.astype(np.float32))      # sum per cluster
    counts = np.bincount(labels - 1, minlength=k)[:, None]
    means = sums / counts                                     # mean per cluster
    return (means / np.linalg.norm(means, axis=1, keepdims=True)).astype(doc.dtype)


def hierarchical_pool(doc: np.ndarray, factor: int) -> np.ndarray:
    """Pool one page (n, d) at a given factor."""
    if factor <= 1 or len(doc) <= 1:
        return doc
    return pool_with_tree(doc, ward_tree(doc), factor)


def random_drop(doc: np.ndarray, factor: int, rng: np.random.Generator) -> np.ndarray:
    """Baseline: keep ceil(n / f) vectors chosen uniformly at random."""
    if factor <= 1:
        return doc
    keep = rng.choice(len(doc), size=n_clusters(len(doc), factor), replace=False)
    return doc[np.sort(keep)]


def cluster_labels(doc: np.ndarray, factor: int) -> np.ndarray:
    """Cluster id of every vector (used to draw which patches get merged)."""
    return fcluster(ward_tree(doc), t=n_clusters(len(doc), factor), criterion="maxclust")
