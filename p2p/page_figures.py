"""Figures drawn ON real pages (ColQwen2 only: its patches form a clean 2D grid).

- patch heatmaps: for a query, which patches win MaxSim, before vs after pooling;
- which patches token pooling merges together;
- worst-case examples taken from results/per_query_colqwen2.json.
"""

import json
import os
import textwrap
from pathlib import Path

# The page images were downloaded at encoding time: read them from the local
# cache instead of asking the Hugging Face Hub again (slow, can hang offline).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from p2p import pooling
from p2p.data import load
from p2p.plots import ASSETS, EMB, LABELS, RESULTS, set_style
from p2p.run_experiments import load_embeddings

MODEL = "colqwen2"


def patch_heat(query: np.ndarray, page: np.ndarray, imgmask: np.ndarray, grid: list[int],
               factor: int = 1) -> np.ndarray:
    """(h, w) MaxSim map: each query token picks its best vector on the page and
    adds its similarity to that vector's patch(es). Bright = patches that win.
    With factor > 1 the page is pooled first: a winning cluster lights up all the
    patches merged into it, which shows how pooling blurs the evidence."""
    vecs = page.astype(np.float32)
    labels = np.arange(len(vecs)) + 1
    if factor > 1:
        labels = pooling.cluster_labels(page, factor)
        vecs = np.stack([vecs[labels == c].mean(0) for c in range(1, labels.max() + 1)])
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    sims = query.astype(np.float32) @ vecs.T                 # (n_q, n_vectors)
    heat = np.zeros(len(vecs), dtype=np.float32)
    np.add.at(heat, sims.argmax(axis=1), sims.max(axis=1))   # credit to each winner
    return heat[labels - 1][imgmask].reshape(grid)


def overlay(ax, image, heat: np.ndarray, title: str, vmax=None) -> None:
    """Draw the page, then the winning patches on top (transparent where 0)."""
    ax.imshow(image)
    ax.imshow(np.ma.masked_equal(heat, 0), cmap="Oranges", alpha=0.75, vmin=0, vmax=vmax,
              interpolation="nearest", extent=(0, image.width, image.height, 0))
    ax.set_title(title, fontsize=12, loc="left")
    ax.axis("off")


def save_png(fig, name: str) -> None:
    """Page figures embed photos of pages: PNG only, moderate DPI, small files."""
    fig.savefig(ASSETS / f"{name}.png", bbox_inches="tight", dpi=110)
    plt.close(fig)


def blank_vs_content(image, size: np.ndarray) -> tuple[float, float]:
    """Mean cluster size of blank patches (almost white pixels) vs content patches."""
    h, w = size.shape
    gray = np.asarray(image.convert("L").resize((w, h), 2), dtype=float)
    blank = gray > 245
    mean = lambda sel: float(size[sel].mean()) if sel.any() else float("nan")
    return mean(blank), mean(~blank)


def fig_pool_clusters(dataset: str = "economics_v2", page_idx: int = 63, factor: int = 8) -> None:
    """Which patches pooling merges: light = big cluster, dark = kept nearly alone.
    Also measures blank vs content cluster sizes on every 15th page of the corpus."""
    pages, _, meta = load_embeddings(EMB / MODEL / dataset)
    imgmask = np.load(EMB / MODEL / dataset / "pages_imgmask.npy")
    off = np.load(EMB / MODEL / dataset / "pages_offsets.npy")
    data, stats = load(dataset), []
    for i in list(range(0, len(pages), 15)) + [page_idx]:
        labels = pooling.cluster_labels(pages[i], factor)
        size = np.bincount(labels)[labels][imgmask[off[i]:off[i + 1]]].reshape(meta["grids"][i])
        stats.append(blank_vs_content(data["image"](meta["page_ids"][i]), size))
    blank, content = np.nanmean(stats[:-1], axis=0)
    (RESULTS / "pooling_blank_vs_content.json").write_text(json.dumps(
        {"model": MODEL, "dataset": dataset, "factor": factor, "pages": len(stats) - 1,
         "mean_cluster_size_blank": blank, "mean_cluster_size_content": content}, indent=1))
    image = data["image"](meta["page_ids"][page_idx])
    fig, axes = plt.subplots(1, 2, figsize=(11, 7))
    axes[0].imshow(image)
    axes[0].set_title("The page", loc="left")
    axes[0].axis("off")
    axes[1].imshow(image)
    axes[1].imshow(-np.log(size), cmap="Oranges", alpha=0.6, interpolation="nearest",
                   extent=(0, image.width, image.height, 0))
    axes[1].set_title(f"{int(size.size)} patches -> {labels.max()} vectors (dark = kept nearly alone)",
                      fontsize=12, loc="left")
    axes[1].axis("off")
    fig.suptitle(f"Pooling {factor}x: blank patches are merged only a bit more than text "
                 f"(avg cluster {blank:.1f} vs {content:.1f} patches)", x=0.02, ha="left",
                 fontsize=15, fontweight="bold")
    save_png(fig, "pooling_clusters")


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
        hi = max(full.max(), pooled.max())
        image = load(dataset)["image"](pid)
        row[0].imshow(image)
        row[0].axis("off")
        text = "\n".join(textwrap.wrap(f'"{load(dataset)["queries"][qid]}"', 42)[:3])
        row[0].set_title(f"{LABELS[meta['label']]}\n{text}", fontsize=11, loc="left")
        overlay(row[1], image, full, f"Full: nDCG@5 = {per_q[dataset]['full'][qid]:.2f}", hi)
        overlay(row[2], image, pooled, f"Pooled {factor}x: nDCG@5 = {per_q[dataset][method][qid]:.2f}", hi)
    fig.suptitle("Where pooling hurts: the winning patches get merged with their neighbours",
                 x=0.02, ha="left", fontsize=17, fontweight="bold")
    fig.tight_layout()
    save_png(fig, "heatmaps_worst_cases")


def _box(ax, x, y, w, h, text, color, text_color="white", size=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                facecolor=color, edgecolor="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=text_color,
            fontsize=size, fontweight="bold", linespacing=1.4)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#52514e", lw=1.8, mutation_scale=16))


def _vectors(ax, x, y, n, cmap, seed, w=0.22, h=1.5):
    """Draw n vectors as thin columns of coloured cells (one column = one vector)."""
    rng = np.random.default_rng(seed)
    for i in range(n):
        for j, v in enumerate(rng.uniform(0.25, 0.95, 8)):
            ax.add_patch(Rectangle((x + i * (w + 0.06), y + j * h / 8), w, h / 8 * 0.9,
                                   facecolor=plt.get_cmap(cmap)(v), edgecolor="none"))


def fig_architecture(dataset: str = "economics_v2", page_idx: int = 63) -> None:
    """Paper-style diagram: page -> patches -> model -> one vector per patch -> MaxSim."""
    meta = json.loads((EMB / MODEL / dataset / "meta.json").read_text())
    image = load(dataset)["image"](meta["page_ids"][page_idx])
    gh, gw = meta["grids"][page_idx]
    fig, ax = plt.subplots(figsize=(16, 6.8))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.9)
    ax.set_xlim(0, 16.2)
    ax.set_ylim(-0.5, 6.4)
    ax.axis("off")
    h = 4.2
    w = h * image.width / image.height
    ax.imshow(image, extent=(0.1, 0.1 + w, 1.0, 1.0 + h))
    for k in range(0, gw + 1, 2):  # patch grid (every 2nd line, to stay readable)
        ax.plot([0.1 + k * w / gw] * 2, [1.0, 1.0 + h], color="#2a78d6", lw=0.5, alpha=0.7)
    for k in range(0, gh + 1, 2):
        ax.plot([0.1, 0.1 + w], [1.0 + k * h / gh] * 2, color="#2a78d6", lw=0.5, alpha=0.7)
    ax.text(0.1, 6.2, "1. A page is cut into\nsmall squares (patches)", fontsize=12, va="top")
    _arrow(ax, 0.2 + w, 3.1, 4.0, 3.3)
    _box(ax, 4.0, 2.5, 2.6, 1.6, "Vision-language\nmodel", "#2a78d6")
    ax.text(4.0, 6.2, "2. A model reads\nthe whole page", fontsize=12, va="top")
    _arrow(ax, 6.7, 3.3, 7.3, 3.3)
    _vectors(ax, 7.4, 2.55, 9, "Blues", 0)
    ax.text(7.4, 6.2, f"3. One vector per patch\n({gh * gw} vectors x 128 numbers)", fontsize=12, va="top")
    ax.add_patch(FancyBboxPatch((7.2, 2.25), 2.95, 2.1, boxstyle="round,pad=0.02", fill=False,
                                edgecolor="#eb6834", lw=2, ls="--"))
    ax.text(8.67, 4.55, "what this project compresses", ha="center", color="#eb6834",
            fontsize=11, fontweight="bold")
    _box(ax, 4.0, 0.0, 2.6, 1.1, '"stock buybacks\nin 2024?"', "#fcefe6", "#0b0b0b", 11)
    ax.text(4.0, -0.35, "the question", fontsize=11, color="#52514e")
    _arrow(ax, 6.7, 0.55, 7.3, 0.55)
    _vectors(ax, 7.4, 0.0, 4, "Oranges", 1, h=1.1)
    ax.text(7.4, -0.4, "one vector per word", fontsize=11, color="#52514e")
    _arrow(ax, 10.25, 3.3, 11.0, 2.6)
    _arrow(ax, 8.6, 0.55, 11.0, 1.6)
    _box(ax, 11.0, 1.2, 2.5, 1.8, "MaxSim\neach word finds its\nbest patch, then\nadd the scores",
         "#1baf7a", size=11)
    ax.text(11.0, 6.2, "4. Match words\nwith patches", fontsize=12, va="top")
    _arrow(ax, 13.6, 2.1, 14.2, 2.1)
    _box(ax, 14.2, 1.5, 1.9, 1.2, "page\nscore", "#0b0b0b")
    fig.suptitle("From pixels to points: how ColPali-style models search a page", x=0.01, ha="left",
                 fontsize=17, fontweight="bold")
    save_png(fig, "architecture")


def main() -> None:
    set_style()
    fig_architecture()
    fig_pool_clusters()
    fig_heatmaps()


if __name__ == "__main__":
    main()
