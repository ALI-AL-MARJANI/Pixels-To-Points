"""Load the four ViDoRe datasets into one simple shape.

Every dataset becomes: pages {page_id: row index in the image table},
queries {query_id: text}, qrels {query_id: {page_id: relevance}},
qtypes {query_id: content label}, and `image(page_id)` to decode a page lazily
(decoding thousands of 1700x2200 PNGs up front is slow).
Only English queries are kept.
"""

import numpy as np
from datasets import load_dataset
from PIL import Image

DATASETS = {
    "finance_v3": ("vidore/vidore_v3_finance_en", "v3", "table"),
    "biomedical_v2": ("vidore/biomedical_lectures_v2", "v2", "slide"),
    "economics_v2": ("vidore/economics_reports_v2", "v2", "report"),
    "arxivqa_v1": ("vidore/arxivqa_test_subsampled", "v1", "chart"),
}

# Page budget per dataset (fixed-seed subset, see `subsample`). None = full corpus.
MAX_PAGES = {"finance_v3": 400, "biomedical_v2": 350, "economics_v2": None, "arxivqa_v1": 200}


def _beir(repo: str, sep: str):
    """V2 uses 'corpus-id' style keys, V3 uses 'corpus_id'; `sep` picks one."""
    cid, qid = f"corpus{sep}id", f"query{sep}id"
    corpus = load_dataset(repo, "corpus", split="test")
    queries = load_dataset(repo, "queries", split="test")
    qrels_ds = load_dataset(repo, "qrels", split="test")
    pages = {str(p): i for i, p in enumerate(corpus[cid])}
    keep = [r for r in queries if r.get("language", "english") == "english"]
    qtexts = {str(r[qid]): r["query"] for r in keep}
    qtypes = {str(r[qid]): r.get("content_type", "") for r in keep}
    qrels: dict[str, dict[str, int]] = {}
    for r in qrels_ds:
        q = str(r[qid])
        if q in qtexts and int(r["score"]) > 0:
            qrels.setdefault(q, {})[str(r[cid])] = int(r["score"])
    return corpus, pages, qtexts, qrels, qtypes


def _v1(repo: str):
    """V1 rows are (query, image) pairs: the page of row i answers query i."""
    ds = load_dataset(repo, split="test")
    pages, qtexts, qrels = {}, {}, {}
    for i, (pid, q) in enumerate(zip(ds["image_filename"], ds["query"])):
        pages.setdefault(pid, i)
        if q:
            qtexts[str(i)] = q
            qrels[str(i)] = {pid: 1}
    return ds, pages, qtexts, qrels, {q: "" for q in qtexts}


def subsample(pages: dict, queries: dict, qrels: dict, max_pages: int, seed: int = 0):
    """Shrink a corpus to `max_pages` pages, fixed seed.

    Queries are visited in random order and kept while their relevant pages fit
    in half the budget; the other half is filled with random distractor pages.
    Every kept query keeps ALL its relevant pages, so qrels stay complete."""
    rng = np.random.default_rng(seed)
    kept_q, needed = set(), set()
    for q in rng.permutation(list(queries)):
        if len(needed | set(qrels[q])) <= max_pages // 2:
            kept_q.add(q)
            needed |= set(qrels[q])
    others = [p for p in pages if p not in needed]
    fill = rng.choice(len(others), size=min(len(others), max_pages - len(needed)), replace=False)
    keep = needed | {others[i] for i in fill}
    return {p: i for p, i in pages.items() if p in keep}, {q: t for q, t in queries.items() if q in kept_q}


def load(name: str, max_queries: int | None = None, max_pages: int | None = -1) -> dict:
    """Load one dataset. `max_pages=-1` uses the default budget in MAX_PAGES.
    `max_queries` gives a tiny smoke-test sample: the first queries, their
    relevant pages, plus as many other pages as distractors."""
    repo, version, label = DATASETS[name]
    table, pages, queries, qrels, qtypes = _v1(repo) if version == "v1" else _beir(
        repo, "_" if version == "v3" else "-")
    queries = {q: t for q, t in queries.items() if q in qrels}
    max_pages = MAX_PAGES[name] if max_pages == -1 else max_pages
    if max_pages and max_pages < len(pages):
        pages, queries = subsample(pages, queries, qrels, max_pages)
    if max_queries:
        queries = dict(list(queries.items())[:max_queries])
        needed = {p for q in queries for p in qrels[q]}
        extra = [p for p in pages if p not in needed][: len(needed)]
        pages = {p: pages[p] for p in pages if p in needed or p in extra}
    return {"name": name, "label": label, "pages": pages, "queries": queries,
            "qrels": {q: {p: s for p, s in qrels[q].items() if p in pages} for q in queries},
            "qtypes": {q: qtypes.get(q, "") for q in queries},
            "image": lambda pid: to_rgb(table[pages[pid]]["image"])}


def to_rgb(img: Image.Image) -> Image.Image:
    """Models expect 3-channel images."""
    return img if img.mode == "RGB" else img.convert("RGB")
