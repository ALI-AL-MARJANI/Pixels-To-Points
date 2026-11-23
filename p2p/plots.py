"""All blog figures, drawn ONLY from results/ and the saved embeddings.

Usage: python -m p2p.plots   (writes PNG + SVG into assets/)
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ASSETS, RESULTS, EMB = Path("assets"), Path("results"), Path("embeddings")
MODELS = {"colsmol": "ColSmol-256M", "colqwen2": "ColQwen2-2B"}
LABELS = {"table": "Tables (finance)", "slide": "Slides (biomed)",
          "report": "Reports (economics)", "chart": "Figures (arXiv)"}
# Fixed categorical order (colourblind-validated palette); the random baseline is neutral grey.
COLORS = {"pool": "#2a78d6", "binary": "#eb6834", "pool_binary": "#1baf7a",
          "fde": "#eda100", "random": "#8a8985", "full": "#0b0b0b"}
NAMES = {"pool": "Token pooling", "binary": "Binary (1 bit)", "pool_binary": "Pooling + binary",
         "fde": "MUVERA + re-rank", "random": "Random dropping", "full": "Full fp16"}


def set_style() -> None:
    """One shared look for every figure: large fonts, light grid, no top/right spines."""
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 220, "font.size": 13, "axes.titlesize": 15,
        "axes.titleweight": "bold", "axes.labelsize": 13, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
        "grid.linewidth": 0.8, "axes.edgecolor": "#8a8985", "legend.frameon": False,
        "lines.linewidth": 2, "lines.markersize": 7, "figure.facecolor": "white",
        "axes.facecolor": "white", "text.color": "#0b0b0b", "axes.labelcolor": "#52514e",
        "xtick.color": "#52514e", "ytick.color": "#52514e"})


def save(fig, name: str) -> None:
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / f"{name}.png", bbox_inches="tight")
    fig.savefig(ASSETS / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def load_summary() -> list[dict]:
    with open(RESULTS / "summary.csv") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in r:
            if k not in ("model", "dataset", "method", "setting"):
                r[k] = float(r[k])
    return rows


def macro(rows: list[dict], metric: str = "ndcg@5") -> dict:
    """Average over datasets (each dataset counts once), then over seeds.
    Returns {(model, method, setting): (mean, std over seeds, bytes per page)}."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[(r["model"], r["method"], r["setting"])][r["seed"]].append((r[metric], r["bytes_per_page"]))
    out = {}
    for key, seeds in by.items():
        per_seed = [np.mean([m for m, _ in v]) for v in seeds.values()]
        nbytes = np.mean([b for v in seeds.values() for _, b in v])
        out[key] = (float(np.mean(per_seed)), float(np.std(per_seed)), float(nbytes))
    return out


def family_curve(agg: dict, model: str, family: str) -> tuple[np.ndarray, np.ndarray]:
    """(bytes, quality) points of one method family, sorted by size."""
    if family == "fde":
        keys = [k for k in agg if k[0] == model and k[1].startswith("fde_") and k[1].endswith("rerank")]
    elif family == "binary":
        keys = [k for k in agg if k[0] == model and k[1] in ("binary", "binary_rescore")]
    else:
        keys = [k for k in agg if k[0] == model and k[1] == family]
    pts = sorted((agg[k][2], agg[k][0]) for k in keys)
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def fig_pareto(agg: dict) -> None:
    """Hero figure: quality kept vs bytes per page, one panel per model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    for ax, (model, title) in zip(axes, MODELS.items()):
        full_q, _, full_b = agg[(model, "full", "fp16")]
        for fam in ("pool", "random", "binary", "pool_binary", "fde"):
            x, y = family_curve(agg, model, fam)
            ls = "--" if fam == "random" else "-"
            ax.plot(x, 100 * y / full_q, ls, marker="o", color=COLORS[fam], label=NAMES[fam])
        ax.scatter([full_b], [100], s=140, marker="*", color=COLORS["full"], zorder=5, label=NAMES["full"])
        ax.set_xscale("log")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Bytes per page (log scale)")
        ax.axhline(95, color="#c3c2b7", lw=1, ls=":")
    axes[0].set_ylabel("Quality kept (% of full nDCG@5)")
    axes[1].legend(loc="lower right", fontsize=11)
    fig.suptitle(pareto_headline(agg), x=0.01, ha="left", fontsize=17, fontweight="bold")
    save(fig, "pareto")


def pareto_headline(agg: dict) -> str:
    """Takeaway title computed from the numbers: smallest pooled size keeping >= 95%."""
    full_q, _, full_b = agg[("colqwen2", "full", "fp16")]
    best = max((full_b / b, 100 * q / full_q, m) for (mo, m, s), (q, _, b) in agg.items()
               if mo == "colqwen2" and q >= 0.95 * full_q and m not in ("full", "random"))
    return f"ColQwen2 keeps {best[1]:.0f}% of its quality with {best[0]:.0f}x less memory ({best[2]})"


def fig_pool_vs_random(agg: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (model, title) in zip(axes, MODELS.items()):
        full_q = agg[(model, "full", "fp16")][0]
        f = np.array([2, 4, 8, 16, 32])
        pool = np.array([agg[(model, "pool", str(x))][0] for x in f])
        rnd = np.array([agg[(model, "random", str(x))][:2] for x in f])
        ax.plot(f, pool, marker="o", color=COLORS["pool"], label=NAMES["pool"])
        ax.plot(f, rnd[:, 0], "--", marker="o", color=COLORS["random"], label=f"{NAMES['random']} (5 seeds)")
        ax.fill_between(f, rnd[:, 0] - rnd[:, 1], rnd[:, 0] + rnd[:, 1], color=COLORS["random"], alpha=0.2)
        ax.axhline(full_q, color=COLORS["full"], lw=1, ls=":", label="Full")
        ax.set_xscale("log", base=2)
        ax.set_xticks(f, [f"{x}x" for x in f])
        ax.set_title(title, loc="left")
        ax.set_xlabel("Fewer vectors per page (pool factor)")
    axes[0].set_ylabel("nDCG@5 (mean of 4 datasets)")
    axes[0].legend(loc="lower left", fontsize=11)
    gap = agg[("colqwen2", "pool", "16")][0] - agg[("colqwen2", "random", "16")][0]
    word = "beats" if gap > 0 else "does not beat"
    fig.suptitle(f"Clustering {word} random dropping (ColQwen2, 16x: {100 * gap:+.1f} nDCG@5 points)",
                 x=0.01, ha="left", fontsize=16, fontweight="bold")
    save(fig, "pool_vs_random")


def fig_by_page_type(rows: list[dict], model: str = "colqwen2") -> None:
    """Quality kept per page type for the main methods."""
    labels = {json.loads((EMB / model / d / "meta.json").read_text())["label"]: d
              for d in {r["dataset"] for r in rows if r["model"] == model}}
    get = {(r["dataset"], r["method"], r["setting"]): r["ndcg@5"] for r in rows
           if r["model"] == model and r["seed"] == 0}
    methods = [("pool", "4", "Pooling 4x"), ("pool", "16", "Pooling 16x"),
               ("binary", "1bit", "Binary"), ("fde_5_16_20", "10240", "MUVERA (no re-rank)")]
    order = [t for t in LABELS if t in labels]
    fig, ax = plt.subplots(figsize=(12, 5))
    w = 0.8 / len(methods)
    for i, ((m, s, name), col) in enumerate(zip(methods, ["#2a78d6", "#1baf7a", "#eb6834", "#eda100"])):
        vals = [100 * get[(labels[t], m, s)] / get[(labels[t], "full", "fp16")] for t in order]
        ax.bar(np.arange(len(order)) + (i - 1.5) * w, vals, w * 0.92, color=col, label=name)
    ax.set_xticks(range(len(order)), [LABELS[t] for t in order])
    ax.axhline(100, color=COLORS["full"], lw=1, ls=":")
    ax.set_ylabel("Quality kept (% of full nDCG@5)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=11)
    loss = {t: get[(labels[t], "pool", "16")] / get[(labels[t], "full", "fp16")] for t in order}
    worst = min(loss, key=loss.get)
    ax.set_title(f"{LABELS[worst].split(' (')[0]} suffer most from aggressive pooling "
                 f"({100 * loss[worst]:.0f}% kept at 16x, {MODELS[model]})", loc="left")
    save(fig, "by_page_type")


def fig_cost(agg: dict, model: str = "colqwen2") -> None:
    """Back-of-envelope card: index size for one million pages."""
    items = [("Full fp16", ("full", "fp16")), ("Pooling 4x", ("pool", "4")),
             ("Pooling 16x", ("pool", "16")), ("Binary", ("binary", "1bit")),
             ("Pooling 4x + binary", ("pool_binary", "4")), ("MUVERA vector", ("fde_5_16_20", "10240"))]
    gb = [agg[(model, *k)][2] * 1e6 / 1e9 for _, k in items]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.barh([n for n, _ in items][::-1], gb[::-1], color="#2a78d6", height=0.6)
    for y, v in enumerate(gb[::-1]):
        ax.text(v, y, f"  {v:,.1f} GB", va="center", color="#0b0b0b", fontsize=12)
    ax.set_xlabel("Index size for 1 million pages (GB)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"1 million pages = {gb[0]:,.0f} GB with full {MODELS[model]}", loc="left")
    save(fig, "cost_1m_pages")


def results_table(agg: dict, rows: list[dict]) -> None:
    """Markdown table for the README/TECHNICAL (macro average over the 4 datasets)."""
    lat = defaultdict(list)
    for r in rows:
        lat[(r["model"], r["method"], r["setting"])].append(r["ms_per_query"])
    lines = ["| Model | Method | Setting | nDCG@5 | % of full | Bytes/page | GB per 1M pages | ms/query |",
             "|---|---|---|---|---|---|---|---|"]
    for model, title in MODELS.items():
        full = agg[(model, "full", "fp16")][0]
        for (m, meth, s), (q, sd, b) in sorted(agg.items(), key=lambda kv: -kv[1][2]):
            if m != model:
                continue
            q_txt = f"{q:.3f} ± {sd:.3f}" if sd > 0 else f"{q:.3f}"
            lines.append(f"| {title} | {meth} | {s} | {q_txt} | {100 * q / full:.1f}% | {b:,.0f} | "
                         f"{b * 1e6 / 1e9:,.1f} | {np.mean(lat[(m, meth, s)]):.1f} |")
    (RESULTS / "table.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    set_style()
    rows = load_summary()
    agg = macro(rows)
    results_table(agg, rows)
    fig_pareto(agg)
    fig_pool_vs_random(agg)
    fig_by_page_type(rows)
    fig_cost(agg)
    from p2p.page_figures import main as page_figures
    page_figures()


if __name__ == "__main__":
    main()
