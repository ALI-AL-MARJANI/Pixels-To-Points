"""Figures drawn ON real pages (ColQwen2 only: its patches form a clean 2D grid).

- patch heatmaps: for a query, how strongly each patch matches (MaxSim view),
  before vs after pooling;
- which patches token pooling merges together;
- worst-case examples taken from results/per_query_colqwen2.json.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from p2p import pooling
from p2p.data import load
from p2p.plots import EMB, LABELS, RESULTS, save, set_style
from p2p.run_experiments import load_embeddings

MODEL = "colqwen2"


def patch_heat(query: np.ndarray, page: np.ndarray, imgmask: np.ndarray, grid: list[int],
               factor: int = 1) -> np.ndarray:
    """(h, w) map: for each image patch, its best similarity with any query token.
    With factor > 1 the page is pooled first and every patch shows the score of
    the cluster mean it was merged into (what the compressed index 'sees')."""
    vecs = page.astype(np.float32)
    if factor > 1:
        labels = pooling.cluster_labels(page, factor)
        means = np.stack([vecs[labels == c].mean(0) for c in range(1, labels.max() + 1)])
        vecs = means[labels - 1]
    sims = (query.astype(np.float32) @ vecs.T).max(axis=0)  # best query token per vector
    return sims[imgmask].reshape(grid)


def overlay(ax, image, heat: np.ndarray, title: str, vmin=None, vmax=None) -> None:
    """Draw the page, then the patch grid on top with a single-hue colour map."""
    ax.imshow(image)
    ax.imshow(heat, cmap="Oranges", alpha=0.6, vmin=vmin, vmax=vmax, interpolation="nearest",
              extent=(0, image.width, image.height, 0))
    ax.set_title(title, fontsize=12, loc="left")
    ax.axis("off")


def fig_pool_clusters(dataset: str = "economics_v2", page_idx: int = 3, factor: int = 8) -> None:
    """Where pooling merges: big clusters (flat background) light, unique content dark."""
    pages, _, meta = load_embeddings(EMB / MODEL / dataset)
    imgmask = np.load(EMB / MODEL / dataset / "pages_imgmask.npy")
    off = np.load(EMB / MODEL / dataset / "pages_offsets.npy")
    mask = imgmask[off[page_idx]:off[page_idx + 1]]
    labels = pooling.cluster_labels(pages[page_idx], factor)
    size = np.bincount(labels)[labels][mask].reshape(meta["grids"][page_idx])
    image = load(dataset)["image"](meta["page_ids"][page_idx])
    fig, axes = plt.subplots(1, 2, figsize=(11, 7))
    axes[0].imshow(image)
    axes[0].set_title("The page", loc="left")
    axes[0].axis("off")
    overlay(axes[1], image, -np.log(size), f"{len(mask.nonzero()[0])} patches -> "
            f"{labels.max()} clusters (dark = kept nearly alone)")
    fig.suptitle(f"Pooling {factor}x merges the empty background first", x=0.02, ha="left",
                 fontsize=16, fontweight="bold")
    save(fig, "pooling_clusters")


def worst_examples(n_per_type: int = 1, method: str = "pool_x16") -> list[tuple[str, str]]:
    """For each page type, the query whose nDCG@5 drops most from full to `method`."""
    per_q = json.loads((RESULTS / f"per_query_{MODEL}.json").read_text())
    picks = []
    for dataset, methods in per_q.items():
        drops = {q: methods["full"][q] - methods[method][q] for q in methods["full"]}
        picks += [(dataset, q) for q in sorted(drops, key=drops.get, reverse=True)[:n_per_type]]
    return picks


def fig_heatmaps(method: str = "pool_x16", factor: int = 16) -> None:
    """Rows = page types; columns: page, full heatmap, pooled heatmap."""
    per_q = json.loads((RESULTS / f"per_query_{MODEL}.json").read_text())
    picks = worst_examples(method=method)
    fig, axes = plt.subplots(len(picks), 3, figsize=(13, 5 * len(picks)))
    for row, (dataset, qid) in zip(np.atleast_2d(axes), picks):
        pages, queries, meta = load_embeddings(EMB / MODEL / dataset)
        imgmask = np.load(EMB / MODEL / dataset / "pages_imgmask.npy")
        off = np.load(EMB / MODEL / dataset / "pages_offsets.npy")
        qi = meta["query_ids"].index(qid)
        pid = max(meta["qrels"][qid], key=meta["qrels"][qid].get)  # most relevant page
        j = meta["page_ids"].index(pid)
        mask = imgmask[off[j]:off[j + 1]]
        full = patch_heat(queries[qi], pages[j], mask, meta["grids"][j])
        pooled = patch_heat(queries[qi], pages[j], mask, meta["grids"][j], factor)
        lo, hi = min(full.min(), pooled.min()), max(full.max(), pooled.max())
        image = load(dataset)["image"](pid)
        row[0].imshow(image)
        row[0].axis("off")
        text = load(dataset)["queries"][qid]
        row[0].set_title(f"{LABELS[meta['label']]}\n\"{text[:70]}{'...' if len(text) > 70 else ''}\"",
                         fontsize=11, loc="left")
        overlay(row[1], image, full, f"Full: nDCG@5 = {per_q[dataset]['full'][qid]:.2f}", lo, hi)
        overlay(row[2], image, pooled, f"Pooled {factor}x: nDCG@5 = {per_q[dataset][method][qid]:.2f}",
                lo, hi)
    fig.suptitle("Where pooling hurts: fine details get averaged away", x=0.02, ha="left",
                 fontsize=17, fontweight="bold")
    fig.tight_layout()
    save(fig, "heatmaps_worst_cases")


def main() -> None:
    set_style()
    fig_pool_clusters()
    fig_heatmaps()


if __name__ == "__main__":
    main()
