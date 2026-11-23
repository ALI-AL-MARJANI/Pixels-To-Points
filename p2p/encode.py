"""Encode pages and queries ONCE with a ColPali-family model and save them in fp16.

Output layout, for one (model, dataset) pair, in `embeddings/<model>/<dataset>/`:
  pages.npy          all page vectors stacked, (total_vectors, dim) float16
  pages_offsets.npy  start row of each page (+ final end row), int64
  pages_imgmask.npy  True where the vector comes from an image patch (not a text token)
  queries.npy / queries_offsets.npy   same for queries
  meta.json          ids, grid sizes, qrels, query types, timings

Usage: python -m p2p.encode --model colqwen2 --dataset economics_v2 [--max-queries 5]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from p2p.data import DATASETS, load

MODELS = {
    "colsmol": ("vidore/colSmol-256M", "ColIdefics3", "ColIdefics3Processor"),
    "colqwen2": ("vidore/colqwen2-v1.0", "ColQwen2", "ColQwen2Processor"),
}


def pick_device() -> tuple[str, torch.dtype]:
    """T4 has no native bf16, so CUDA uses fp16. Apple GPUs (MPS) handle bf16."""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.float32


def load_model(key: str, device: str, dtype: torch.dtype):
    import colpali_engine.models as cm
    repo, model_cls, proc_cls = MODELS[key]
    # Load on CPU then move: `device_map="mps"` segfaults with torch 2.13 on macOS.
    model = getattr(cm, model_cls).from_pretrained(repo, torch_dtype=dtype).to(device).eval()
    return model, getattr(cm, proc_cls).from_pretrained(repo)


@torch.no_grad()
def embed(model, batch) -> list[np.ndarray]:
    """Run the model and drop padding rows (attention_mask == 0) of each item."""
    batch = batch.to(model.device)
    out = model(**batch)
    keep = batch["attention_mask"].bool()
    return [out[i][keep[i]].float().cpu().numpy().astype(np.float16) for i in range(len(out))]


def encode_pages(model, proc, image_fn, pids: list[str], bs: int) -> tuple[list, list, list]:
    """`image_fn(page_id)` decodes one page lazily (see data.load)."""
    vecs, masks, grids = [], [], []
    for s in range(0, len(pids), bs):
        batch = proc.process_images([image_fn(p) for p in pids[s:s + bs]])
        img_mask = proc.get_image_mask(batch)  # (B, seq) True on image-patch tokens
        keep = batch["attention_mask"].bool()
        vecs += embed(model, batch)
        masks += [img_mask[i][keep[i]].cpu().numpy() for i in range(len(keep))]
        if "image_grid_thw" in batch:  # Qwen2-VL: patch grid, then 2x2 spatial merge
            grids += [[int(t[1]) // 2, int(t[2]) // 2] for t in batch["image_grid_thw"]]
        print(f"  pages {min(s + bs, len(pids))}/{len(pids)}", flush=True)
    return vecs, masks, grids


def encode_queries(model, proc, texts: list[str], bs: int) -> list[np.ndarray]:
    vecs = []
    for s in range(0, len(texts), bs):
        vecs += embed(model, proc.process_queries(texts[s:s + bs]))
    return vecs


def save_stack(path: Path, name: str, arrays: list[np.ndarray]) -> None:
    """Ragged list -> one big array + offsets (no padding stored on disk)."""
    np.save(path / f"{name}.npy", np.concatenate(arrays))
    np.save(path / f"{name}_offsets.npy", np.cumsum([0] + [len(a) for a in arrays]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", default="embeddings")
    args = ap.parse_args()

    device, dtype = pick_device()
    model, proc = load_model(args.model, device, dtype)
    data = load(args.dataset, args.max_queries)
    pids, qids = list(data["pages"]), list(data["queries"])
    out = Path(args.out) / args.model / args.dataset
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    pvecs, pmasks, grids = encode_pages(model, proc, data["image"], pids, args.batch_size)
    t1 = time.time()
    qvecs = encode_queries(model, proc, [data["queries"][q] for q in qids], 16)
    t2 = time.time()

    save_stack(out, "pages", pvecs)
    np.save(out / "pages_imgmask.npy", np.concatenate(pmasks))
    save_stack(out, "queries", qvecs)
    meta = {"model": MODELS[args.model][0], "dataset": args.dataset, "label": data["label"],
            "device": device, "dtype": str(dtype), "page_ids": pids, "query_ids": qids,
            "grids": grids, "qrels": data["qrels"], "qtypes": data["qtypes"],
            "seconds_pages": t1 - t0, "seconds_queries": t2 - t1}
    (out / "meta.json").write_text(json.dumps(meta))
    print(f"done: {len(pids)} pages in {t1 - t0:.0f}s, {len(qids)} queries in {t2 - t1:.0f}s -> {out}")


if __name__ == "__main__":
    main()
