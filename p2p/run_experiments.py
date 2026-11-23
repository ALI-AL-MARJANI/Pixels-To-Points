"""Run every compression method on the saved embeddings and save the numbers.

Same queries, same pages, same model for every method. Writes:
  results/summary.csv            one row per (model, dataset, method, setting, seed)
  results/per_query_<model>.json nDCG@5 per query for key methods (error analysis)

Usage: python -m p2p.run_experiments [--models colsmol colqwen2]
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from p2p import binarize, muvera, pooling
from p2p.metrics import evaluate, per_query_ndcg, scores_to_run
from p2p.scoring import hamming_maxsim_matrix, maxsim_matrix

FACTORS = [2, 4, 8, 16, 32]
SEEDS = [0, 1, 2, 3, 4]
FDE_CONFIGS = [(3, 8, 8), (4, 8, 16), (5, 16, 20)]  # (k_sim, d_proj, reps)
RERANK_TOP = 100
KEY_METHODS = {"full", "pool_x4", "pool_x16", "binary", "binary_rescore",
               "fde_5_16_20_rerank", "random_x16"}


def split(flat: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    return [flat[a:b] for a, b in zip(offsets[:-1], offsets[1:])]


def load_embeddings(root: Path) -> tuple[list, list, dict]:
    pages = split(np.load(root / "pages.npy"), np.load(root / "pages_offsets.npy"))
    queries = split(np.load(root / "queries.npy"), np.load(root / "queries_offsets.npy"))
    return pages, queries, json.loads((root / "meta.json").read_text())


def timed(fn, n_queries: int) -> tuple[np.ndarray, float]:
    """Run a scoring function, return scores and milliseconds per query."""
    t = time.perf_counter()
    scores = fn()
    return scores, 1000 * (time.perf_counter() - t) / n_queries


def run_dataset(pages: list, queries: list, meta: dict) -> tuple[list, dict]:
    qids, pids, qrels = meta["query_ids"], meta["page_ids"], meta["qrels"]
    dim, n_pages = pages[0].shape[1], len(pages)
    rows, per_query = [], {}

    def record(method: str, setting, scores, ms, bytes_total, n_vectors, seed=0, comp_s=0.0):
        run = scores_to_run(scores, qids, pids)
        res = evaluate(run, qrels)
        rows.append({"method": method, "setting": setting, "seed": seed, **res,
                     "bytes_per_page": bytes_total / n_pages, "vectors_per_page": n_vectors / n_pages,
                     "ms_per_query": ms, "compress_ms_per_page": 1000 * comp_s / n_pages})
        name = f"{method}_x{setting}" if method in ("pool", "random") else method
        if name in KEY_METHODS and seed == 0:
            per_query[name] = per_query_ndcg(run, qrels, 5)

    n_vec = sum(len(p) for p in pages)
    full, ms = timed(lambda: maxsim_matrix(queries, pages), len(qids))
    record("full", "fp16", full, ms, n_vec * dim * 2, n_vec)

    # Token pooling: build each page's Ward tree once, cut it at every factor.
    t = time.perf_counter()
    trees = [pooling.ward_tree(p) for p in pages]
    tree_s = time.perf_counter() - t
    pooled_by_f = {}
    for f in FACTORS:
        t = time.perf_counter()
        pooled = [pooling.pool_with_tree(p, tr, f) for p, tr in zip(pages, trees)]
        comp = tree_s + time.perf_counter() - t
        pooled_by_f[f] = pooled
        n = sum(len(p) for p in pooled)
        s, ms = timed(lambda: maxsim_matrix(queries, pooled), len(qids))
        record("pool", f, s, ms, n * dim * 2, n, comp_s=comp)
        for seed in SEEDS:  # random dropping at the same budget, 5 seeds
            rng = np.random.default_rng(seed)
            dropped = [pooling.random_drop(p, f, rng) for p in pages]
            s, ms = timed(lambda: maxsim_matrix(queries, dropped), len(qids))
            record("random", f, s, ms, n * dim * 2, n, seed=seed)

    # Binarization: 1 bit per dimension, with and without centering.
    for method, center in (("binary_nocenter", None), ("binary", binarize.fit_center(pages))):
        t = time.perf_counter()
        pb = [binarize.pack_signs(p, center) for p in pages]
        comp = time.perf_counter() - t
        qb = [binarize.pack_signs(q, center) for q in queries]
        s, ms = timed(lambda: hamming_maxsim_matrix(qb, pb, dim), len(qids))
        record(method, "1bit", s, ms, n_vec * dim / 8, n_vec, comp_s=comp)
    s_asym, ms = timed(lambda: binarize.asymmetric_maxsim_matrix(queries, pb, dim), len(qids))
    record("binary_asym", "1bit", s_asym, ms, n_vec * dim / 8, n_vec)
    bin_scores = s  # centered symmetric Hamming scores
    s, ms2 = timed(lambda: binarize.rescore(bin_scores, queries, pages, RERANK_TOP), len(qids))
    record("binary_rescore", f"top{RERANK_TOP}", s, ms + ms2, n_vec * dim / 8, n_vec)

    # Combination: pooling then binarization.
    for f in (2, 4, 8):
        center = binarize.fit_center(pooled_by_f[f])
        pb = [binarize.pack_signs(p, center) for p in pooled_by_f[f]]
        qb = [binarize.pack_signs(q, center) for q in queries]
        n = sum(len(p) for p in pb)
        s, ms = timed(lambda: hamming_maxsim_matrix(qb, pb, dim), len(qids))
        record("pool_binary", f, s, ms, n * dim / 8, n)

    # MUVERA: one fp16 vector per page, then exact MaxSim re-rank of the top 100.
    for k_sim, d_proj, reps in FDE_CONFIGS:
        p = muvera.make_params(dim, k_sim, d_proj, reps)
        t = time.perf_counter()
        fd = muvera.encode_many(pages, p, is_query=False).astype(np.float16)
        comp = time.perf_counter() - t
        tag = f"{k_sim}_{d_proj}_{reps}"
        fq = muvera.encode_many(queries, p, is_query=True)
        s, ms = timed(lambda: fq @ fd.astype(np.float32).T, len(qids))
        record(f"fde_{tag}", p.size, s, ms, p.size * 2 * n_pages, n_pages, comp_s=comp)
        r, ms2 = timed(lambda: binarize.rescore(s, queries, pages, RERANK_TOP), len(qids))
        record(f"fde_{tag}_rerank", p.size, r, ms + ms2, p.size * 2 * n_pages, n_pages)
    return rows, per_query


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["colsmol", "colqwen2"])
    ap.add_argument("--emb", default="embeddings")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    all_rows = []
    for model in args.models:
        per_query_all = {}
        for root in sorted((Path(args.emb) / model).iterdir()):
            pages, queries, meta = load_embeddings(root)
            print(f"{model}/{root.name}: {len(pages)} pages, {len(queries)} queries", flush=True)
            rows, per_query = run_dataset(pages, queries, meta)
            all_rows += [{"model": model, "dataset": root.name, **r} for r in rows]
            per_query_all[root.name] = per_query
        (out / f"per_query_{model}.json").write_text(json.dumps(per_query_all))
    with open(out / "summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {len(all_rows)} rows to {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
