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
MAIN = "colqwen2"  # model used for single-model figures and headlines
LABELS = {"table": "Finance 10-K", "slide": "Slides (biomed)",
          "report": "Reports (economics)", "chart": "Figures (arXiv)"}
# Fixed categorical order (colourblind-validated palette); the random baseline is neutral grey.
COLORS = {"pool": "#2a78d6", "pool_binary": "#eb6834", "fde": "#1baf7a",
          "random": "#8a8985", "full": "#0b0b0b"}
NAMES = {"pool": "Token pooling (fp16)", "pool_binary": "Pooling + binary (1 bit)",
         "fde": "MUVERA, one vector (no re-rank)", "random": "Random dropping", "full": "Full fp16"}
# Each stand-alone family as (method, setting) points; factor 1 = no pooling.
FAMILIES = {
    "pool": [("full", "fp16")] + [("pool", str(f)) for f in (2, 4, 8, 16, 32)],
    "random": [("full", "fp16")] + [("random", str(f)) for f in (2, 4, 8, 16, 32)],
    "pool_binary": [("binary", "1bit")] + [("pool_binary", str(f)) for f in (2, 4, 8)],
    "fde": [("fde_3_8_8", "512"), ("fde_4_8_16", "2048"), ("fde_5_16_20", "10240")],
}
TWO_STAGE = [("binary_rescore", "top100", "binary"), ("fde_3_8_8_rerank", "512", "fde_3_8_8"),
             ("fde_4_8_16_rerank", "2048", "fde_4_8_16"), ("fde_5_16_20_rerank", "10240", "fde_5_16_20")]


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
    rows = []
    for path in sorted(RESULTS.glob("summary_*.csv")):
        with open(path) as fh:
            rows += list(csv.DictReader(fh))
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
    """(bytes, quality) points of one stand-alone method family, sorted by size."""
    pts = sorted((agg[(model, m, st)][2], agg[(model, m, st)][0]) for m, st in FAMILIES[family])
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def fig_pareto(agg: dict) -> None:
    """Hero figure: quality kept vs bytes per page, stand-alone methods only
    (re-ranking methods also need the full vectors, see fig_two_step)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    for ax, (model, title) in zip(axes, MODELS.items()):
        full_q, _, full_b = agg[(model, "full", "fp16")]
        for fam in ("pool", "random", "pool_binary", "fde"):
            x, y = family_curve(agg, model, fam)
            ax.plot(x, 100 * y / full_q, "--" if fam == "random" else "-", marker="o",
                    color=COLORS[fam], label=NAMES[fam])
        ax.scatter([full_b], [100], s=160, marker="*", color=COLORS["full"], zorder=5, label=NAMES["full"])
        ax.set_xscale("log")
        ax.set_title(title, loc="left")
        ax.set_xlabel("Bytes per page (log scale)")
        ax.axhline(95, color="#c3c2b7", lw=1, ls=":")
    axes[0].set_ylabel("Quality kept (% of full nDCG@5)")
    axes[-1].legend(loc="lower right", fontsize=11)
    fig.suptitle(pareto_headline(agg), x=0.01, ha="left", fontsize=17, fontweight="bold")
    save(fig, "pareto")


def pareto_headline(agg: dict) -> str:
    """Takeaway title computed from the numbers: biggest compression keeping >= 95%."""
    full_q, _, full_b = agg[(MAIN, "full", "fp16")]
    pts = [(full_b / agg[(MAIN, m, st)][2], 100 * agg[(MAIN, m, st)][0] / full_q, f"{m} {st}")
           for fam in ("pool", "pool_binary") for m, st in FAMILIES[fam]]
    best = max(p for p in pts if p[1] >= 95)
    return f"{MODELS[MAIN]} keeps {best[1]:.0f}% of its quality with {best[0]:.0f}x less memory"


def fig_two_step(rows: list[dict]) -> None:
    """Two-stage search: a cheap first stage finds candidates, exact MaxSim re-ranks
    the top 100. Left: is the right page in the shortlist? Right: final quality."""
    agg_r, agg_n = macro(rows, "recall@100"), macro(rows, "ndcg@5")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    labels = ["1 bit\nper number", "MUVERA\n512-d", "MUVERA\n2048-d", "MUVERA\n10240-d"]
    w = 0.38
    for i, (model, title) in enumerate(MODELS.items()):
        full_n = agg_n[(model, "full", "fp16")][0]
        rec = [100 * agg_r[(model, first, st if first != "binary" else "1bit")][0]
               for _, st, first in TWO_STAGE]
        kept = [100 * agg_n[(model, m, st)][0] / full_n for m, st, _ in TWO_STAGE]
        col = ["#2a78d6", "#eb6834"][i]
        axes[0].bar(np.arange(4) + (i - 0.5) * w, rec, w * 0.92, color=col, label=title)
        axes[1].bar(np.arange(4) + (i - 0.5) * w, kept, w * 0.92, color=col, label=title)
    for ax, lab in zip(axes, ["Step 1: right page in the top 100 (%)", "Step 2: quality kept after exact check (%)"]):
        ax.set_xticks(range(4), labels)
        ax.set_ylabel(lab)
        ax.set_ylim(0, 105)
        ax.grid(axis="x", visible=False)
    axes[1].axhline(100, color=COLORS["full"], lw=1, ls=":")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=11)
    kept_b = 100 * agg_n[(MAIN, "binary_rescore", "top100")][0] / agg_n[(MAIN, "full", "fp16")][0]
    fig.suptitle(f"Cheap first step + exact check of the top 100: 1-bit codes recover {kept_b:.0f}% of the quality "
                 f"({MODELS[MAIN]})", x=0.01, y=1.04, ha="left", fontsize=16, fontweight="bold")
    save(fig, "two_step")


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
    gap = agg[(MAIN, "pool", "16")][0] - agg[(MAIN, "random", "16")][0]
    word = "beats" if gap > 0 else "does not beat"
    fig.suptitle(f"Clustering {word} random dropping ({MODELS[MAIN]}, 16x: {100 * gap:+.1f} nDCG@5 points)",
                 x=0.01, ha="left", fontsize=16, fontweight="bold")
    save(fig, "pool_vs_random")


def fig_by_page_type(rows: list[dict], model: str | None = None) -> None:
    """Quality kept per page type for the main methods."""
    model = model or MAIN
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
    ax.set_title(f"Aggressive pooling hurts {LABELS[worst].split(' (')[0]} pages most "
                 f"({100 * loss[worst]:.0f}% kept at 16x, {MODELS[model]})", loc="left")
    save(fig, "by_page_type")


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
    fig_two_step(rows)
    from p2p.page_figures import main as page_figures
    page_figures()


if __name__ == "__main__":
    main()
