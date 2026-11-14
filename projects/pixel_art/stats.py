"""Detailed statistics for Angais/pixel-art-atlas-1.4m (16x16 RGBA sprites + captions).

Run from anywhere (pyarrow is not a zoo dependency, so pull it in for the run):
    uv run --with pyarrow projects/pixel_art/stats.py
    uv run --with pyarrow projects/pixel_art/stats.py --max-files 2   # quick pass on 2 train shards

Downloads the parquet shards (~435 MB) to the Hugging Face cache, reads them directly (no
`datasets`), and writes to <repo>/runs/pixel_art/stats/:
  stats.json   every number below
  *.png        plots (nothing is shown interactively)

Reported, for train and validation separately plus their overlap:
  rows        row counts, row_id uniqueness/range
  subjects    count, frequency distribution (min/median/max, entropy, gini, head/tail), val-only subjects
  captions    unique/duplicate counts, caption == subject rate, subject-contained rate, length in
              chars/words, vocabulary, top words/bigrams, "with" clause rate, modifier words before the subject
  images      size/mode, alpha histogram (binary or soft?), transparent fraction, empty images,
              bounding box size and centering, border-touching rate, unique colors per image,
              global palette size and top colors, RGB mean/std over opaque pixels, per-pixel
              mean alpha, exact duplicate images (and duplicates that carry different captions),
              train/val exact image leakage
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections import Counter
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image

REPO_ID = "Angais/pixel-art-atlas-1.4m"
RUNS_ROOT = Path(__file__).resolve().parents[2] / "runs" / "pixel_art" / "stats"
WORD = re.compile(r"[a-z0-9']+")
import tyro

# ---------------------------------------------------------------- per-shard image stats


def decode(b: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGBA"))


def shard_stats(path: str) -> dict:
    """Aggregate everything image-related for one parquet shard, plus its text columns."""
    table = pq.read_table(path, columns=["image", "caption", "subject", "row_id"])
    imgs = table.column("image").to_pylist()
    n = len(imgs)
    sizes = Counter()
    arr = np.zeros((n, 16, 16, 4), np.uint8)
    for i, d in enumerate(imgs):
        a = decode(d["bytes"])
        sizes[a.shape[:2]] += 1
        if a.shape[:2] == (16, 16):
            arr[i] = a
    rgb, alpha = arr[..., :3], arr[..., 3]
    opaque = alpha > 0

    # Hash after zeroing hidden RGB so invisible pixels can't make identical sprites differ.
    clean = arr.copy()
    clean[~opaque] = 0
    hashes = np.array(
        [int.from_bytes(hashlib.blake2b(c.tobytes(), digest_size=8).digest(), "little") for c in clean],
        dtype=np.uint64,
    )

    packed = (rgb[..., 0].astype(np.uint32) << 16) | (rgb[..., 1].astype(np.uint32) << 8) | rgb[..., 2]
    colors = Counter(dict(zip(*[x.tolist() for x in np.unique(packed[opaque], return_counts=True)])))
    # unique opaque colors per image: sort each image's pixels (transparent -> sentinel) and count changes
    flat = np.where(opaque, packed, np.uint32(1 << 24)).reshape(n, -1)
    flat.sort(axis=1)
    ncolors = (np.diff(flat, axis=1) != 0).sum(1) + 1 - (flat[:, -1] == (1 << 24))

    cov = opaque.mean((1, 2))
    rows_any, cols_any = opaque.any(2), opaque.any(1)  # (n,16) each
    has = cov > 0
    r0, r1 = rows_any.argmax(1), 15 - rows_any[:, ::-1].argmax(1)
    c0, c1 = cols_any.argmax(1), 15 - cols_any[:, ::-1].argmax(1)
    ys, xs = np.mgrid[:16, :16]
    w = opaque.sum((1, 2)).clip(1)
    com_y = (opaque * ys).sum((1, 2)) / w
    com_x = (opaque * xs).sum((1, 2)) / w
    border = opaque[:, 0].any(1) | opaque[:, -1].any(1) | opaque[:, :, 0].any(1) | opaque[:, :, -1].any(1)

    o = rgb[opaque].astype(np.float64)
    return {
        "sizes": {f"{h}x{w_}": c for (h, w_), c in sizes.items()},
        "n": n,
        "alpha_hist": np.bincount(alpha.ravel(), minlength=256),
        "alpha_sum_px": alpha.astype(np.float64).sum(0) / 255.0,  # (16,16) summed over images
        "coverage": cov.astype(np.float32),
        "ncolors": ncolors.astype(np.int16),
        "bbox_h": np.where(has, r1 - r0 + 1, 0).astype(np.int8),
        "bbox_w": np.where(has, c1 - c0 + 1, 0).astype(np.int8),
        "com_y": np.where(has, com_y, np.nan).astype(np.float32),
        "com_x": np.where(has, com_x, np.nan).astype(np.float32),
        "border": border,
        "colors": colors,
        "rgb_sum": o.sum(0),
        "rgb_sq": (o**2).sum(0),
        "rgb_n": len(o),
        "hashes": hashes,
        "caption": table.column("caption").to_pylist(),
        "subject": table.column("subject").to_pylist(),
        "row_id": np.asarray(table.column("row_id").to_numpy(), dtype=np.int64),
        "sample": arr[:: max(1, n // 40)][:40].copy(),
        "sample_caption": table.column("caption").to_pylist()[:: max(1, n // 40)][:40],
    }


def merge(shards: list[dict]) -> dict:
    out: dict = {"n": sum(s["n"] for s in shards)}
    for k in ("alpha_hist", "alpha_sum_px", "rgb_sum", "rgb_sq"):
        out[k] = sum(s[k] for s in shards)
    out["rgb_n"] = sum(s["rgb_n"] for s in shards)
    for k in ("coverage", "ncolors", "bbox_h", "bbox_w", "com_y", "com_x", "border", "hashes", "row_id"):
        out[k] = np.concatenate([s[k] for s in shards])
    for k in ("caption", "subject"):
        out[k] = [x for s in shards for x in s[k]]
    out["sizes"] = sum((Counter(s["sizes"]) for s in shards), Counter())
    out["colors"] = sum((s["colors"] for s in shards), Counter())
    out["sample"] = shards[0]["sample"]
    out["sample_caption"] = shards[0]["sample_caption"]
    return out


# ---------------------------------------------------------------- summaries


def dist(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    q = np.percentile(x, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    return {
        "mean": float(x.mean()), "std": float(x.std()),
        **{f"p{p}": float(v) for p, v in zip([0, 1, 5, 25, 50, 75, 95, 99, 100], q)},
    }


def gini(counts: np.ndarray) -> float:
    c = np.sort(counts.astype(np.float64))
    n = len(c)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(c) / (n * c.sum()))


def text_stats(d: dict) -> dict:
    caps, subs = d["caption"], d["subject"]
    n = len(caps)
    sub_counts = Counter(subs)
    cnt = np.array(sorted(sub_counts.values()))
    p = cnt / cnt.sum()
    cap_counts = Counter(caps)
    words = [WORD.findall(c.lower()) for c in caps]
    wc = Counter(w for ws in words for w in ws)
    bi = Counter(b for ws in words for b in zip(ws, ws[1:]))
    nw = np.array([len(ws) for ws in words])
    nc = np.array([len(c) for c in caps])
    same = sum(c == s for c, s in zip(caps, subs))
    contains = sum(s in c for c, s in zip(caps, subs))
    with_clause = sum(" with " in c for c in caps)
    # modifier words: whatever precedes the subject when the caption contains it
    mods = Counter()
    for c, s in zip(caps, subs):
        i = c.find(s)
        if i > 0:
            mods.update(c[:i].split())
    # per-subject number of distinct captions
    per_sub_caps: dict[str, set] = {}
    for c, s in zip(caps, subs):
        per_sub_caps.setdefault(s, set()).add(c)
    distinct = np.array([len(v) for v in per_sub_caps.values()])
    sub_words = np.array([len(s.split()) for s in sub_counts])
    return {
        "rows": n,
        "unique_captions": len(cap_counts),
        "duplicate_caption_rows": n - len(cap_counts),
        "most_common_captions": cap_counts.most_common(15),
        "unique_subjects": len(sub_counts),
        "subject_count_dist": dist(cnt),
        "subject_entropy_bits": float(-(p * np.log2(p)).sum()),
        "subject_max_entropy_bits": float(np.log2(len(cnt))),
        "subject_gini": gini(cnt),
        "top_subjects": sub_counts.most_common(25),
        "bottom_subjects": sub_counts.most_common()[-15:],
        "subjects_with_lt_50_rows": int((cnt < 50).sum()),
        "subjects_with_lt_100_rows": int((cnt < 100).sum()),
        "rows_in_top_10pct_subjects": float(cnt[-max(1, len(cnt) // 10):].sum() / cnt.sum()),
        "subject_words_dist": dist(sub_words),
        "distinct_captions_per_subject": dist(distinct),
        "caption_equals_subject": same / n,
        "caption_contains_subject": contains / n,
        "caption_has_with_clause": with_clause / n,
        "caption_chars": dist(nc),
        "caption_words": dist(nw),
        "vocab_size": len(wc),
        "words_seen_once": sum(1 for v in wc.values() if v == 1),
        "top_words": wc.most_common(40),
        "top_bigrams": [(" ".join(b), c) for b, c in bi.most_common(25)],
        "top_modifiers": mods.most_common(40),
        "caption_nonascii": sum(not c.isascii() for c in caps) / n,
        "caption_empty": sum(not c.strip() for c in caps) / n,
    }


def image_stats(d: dict) -> dict:
    n = d["n"]
    ah = d["alpha_hist"].astype(np.float64)
    total = ah.sum()
    semi = ah[1:255].sum() / total
    cov = d["coverage"]
    hashes = d["hashes"]
    uniq, counts = np.unique(hashes, return_counts=True)
    ntotal = d["rgb_n"]
    mean = d["rgb_sum"] / ntotal
    std = np.sqrt(d["rgb_sq"] / ntotal - mean**2)
    pal = d["colors"]
    top = pal.most_common(20)
    cum = np.cumsum(sorted(pal.values(), reverse=True)) / ntotal
    return {
        "image_sizes": dict(d["sizes"]),
        "alpha_fully_transparent_px": float(ah[0] / total),
        "alpha_fully_opaque_px": float(ah[255] / total),
        "alpha_semi_transparent_px": float(semi),
        "alpha_distinct_values": int((ah > 0).sum()),
        "coverage": dist(cov),
        "empty_images": int((cov == 0).sum()),
        "full_images(no transparency)": int((cov == 1).sum()),
        "bbox_w": dist(d["bbox_w"]),
        "bbox_h": dist(d["bbox_h"]),
        "bbox_full_16x16_rate": float(((d["bbox_w"] == 16) & (d["bbox_h"] == 16)).mean()),
        "center_of_mass_x": dist(d["com_x"]),
        "center_of_mass_y": dist(d["com_y"]),
        "touches_border_rate": float(d["border"].mean()),
        "unique_colors_per_image": dist(d["ncolors"]),
        "global_unique_colors": len(pal),
        "colors_covering_90pct_of_pixels": int(np.searchsorted(cum, 0.90) + 1),
        "colors_covering_99pct_of_pixels": int(np.searchsorted(cum, 0.99) + 1),
        "top_colors": [(f"#{c:06x}", k, k / ntotal) for c, k in top],
        "rgb_mean_opaque_0_255": mean.tolist(),
        "rgb_std_opaque_0_255": std.tolist(),
        "exact_unique_images": int(len(uniq)),
        "exact_duplicate_images": int(n - len(uniq)),
        "largest_duplicate_groups": sorted(counts.tolist(), reverse=True)[:10],
        "mean_alpha_per_pixel": (d["alpha_sum_px"] / n).tolist(),
    }


def duplicate_caption_conflicts(d: dict) -> dict:
    """Identical images that appear under different captions, and the reverse."""
    by_img: dict[int, set] = {}
    for h, c in zip(d["hashes"].tolist(), d["caption"]):
        by_img.setdefault(h, set()).add(c)
    img_multi_cap = sum(1 for v in by_img.values() if len(v) > 1)
    by_cap: dict[str, set] = {}
    for h, c in zip(d["hashes"].tolist(), d["caption"]):
        by_cap.setdefault(c, set()).add(h)
    cap_multi_img = sum(1 for v in by_cap.values() if len(v) > 1)
    return {
        "images_shared_by_multiple_captions": img_multi_cap,
        "captions_mapping_to_multiple_images": cap_multi_img,
        "captions_total": len(by_cap),
    }


# ---------------------------------------------------------------- plots


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(RUNS_ROOT / name, dpi=130)
    plt.close()


def plot_all(tr: dict, va: dict, tstat: dict) -> None:
    sub_counts = np.array(sorted(Counter(tr["subject"]).values(), reverse=True))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(sub_counts)
    ax[0].set(xlabel="subject rank", ylabel="rows", yscale="log", title="Subject frequency (rank plot)")
    ax[1].hist(sub_counts, bins=60)
    ax[1].set(xlabel="rows per subject", ylabel="subjects", title="Rows per subject")
    ax[2].plot(np.cumsum(sub_counts) / sub_counts.sum())
    ax[2].set(xlabel="subject rank", ylabel="cumulative row share", title="Cumulative coverage")
    savefig("subjects.png")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    nw = [len(WORD.findall(c.lower())) for c in tr["caption"]]
    ax[0].hist(nw, bins=np.arange(0, max(nw) + 2) - 0.5)
    ax[0].set(xlabel="words per caption", ylabel="rows", title="Caption length (words)")
    top = tstat["train"]["top_words"][:25][::-1]
    ax[1].barh([w for w, _ in top], [c for _, c in top])
    ax[1].set(title="Top words")
    top = tstat["train"]["top_modifiers"][:25][::-1]
    ax[2].barh([w for w, _ in top], [c for _, c in top])
    ax[2].set(title="Top modifiers before subject")
    savefig("captions.png")

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0, 0].hist(tr["coverage"], bins=64)
    ax[0, 0].set(xlabel="opaque fraction", ylabel="images", title="Sprite coverage")
    ax[0, 1].hist(tr["ncolors"], bins=np.arange(0, tr["ncolors"].max() + 2) - 0.5)
    ax[0, 1].set(xlabel="unique opaque colors", ylabel="images", title="Colors per image")
    ax[0, 2].bar(np.arange(256), tr["alpha_hist"])
    ax[0, 2].set(yscale="log", xlabel="alpha", ylabel="pixels", title="Alpha histogram")
    im = ax[1, 0].imshow(tr["alpha_sum_px"] / tr["n"], cmap="magma")
    ax[1, 0].set(title="Mean alpha per pixel")
    fig.colorbar(im, ax=ax[1, 0])
    ax[1, 1].hist2d(tr["bbox_w"], tr["bbox_h"], bins=np.arange(0.5, 17.5), cmap="viridis")
    ax[1, 1].set(xlabel="bbox width", ylabel="bbox height", title="Bounding box size")
    ax[1, 2].hist2d(tr["com_x"][~np.isnan(tr["com_x"])], tr["com_y"][~np.isnan(tr["com_y"])], bins=48, cmap="viridis")
    ax[1, 2].invert_yaxis()
    ax[1, 2].set(xlabel="center of mass x", ylabel="y", title="Sprite centering")
    savefig("images.png")

    pal = tr["colors"].most_common(64)
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.imshow(
        [[[(c >> 16) & 255, (c >> 8) & 255, c & 255] for c, _ in pal[r * 32:(r + 1) * 32]] for r in range(2)]
    )
    ax.axis("off")
    ax.set_title("Top 64 colors by pixel count")
    savefig("palette.png")

    fig, ax = plt.subplots(5, 8, figsize=(16, 10.5))
    for a, img, cap in zip(ax.ravel(), tr["sample"], tr["sample_caption"]):
        bg = np.full((16, 16, 3), 200, np.uint8)
        bg[::2, 1::2] = bg[1::2, ::2] = 160  # checkerboard to show transparency
        al = img[..., 3:4] / 255.0
        a.imshow((img[..., :3] * al + bg * (1 - al)).astype(np.uint8), interpolation="nearest")
        a.set_title(cap[:28], fontsize=7)
        a.axis("off")
    savefig("samples.png")


# ---------------------------------------------------------------- main


def load_split(files: list[str], workers: int) -> dict:
    with Pool(min(workers, len(files))) as pool:
        return merge(pool.map(shard_stats, files))


@dataclass
class Args:
    max_files: int | None = None  # only use this many train shards
    workers: int = 4


def main() -> None:
    args = tyro.cli(Args, description=__doc__)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    root = Path(snapshot_download(REPO_ID, repo_type="dataset", allow_patterns=["*.parquet"]))
    train_files = sorted(map(str, root.rglob("train-*.parquet")))[: args.max_files]
    val_files = sorted(map(str, root.rglob("validation-*.parquet")))
    print(f"{len(train_files)} train shards, {len(val_files)} validation shards")

    tr, va = load_split(train_files, args.workers), load_split(val_files, args.workers)

    t_sub, v_sub = set(tr["subject"]), set(va["subject"])
    t_cap, v_cap = set(tr["caption"]), set(va["caption"])
    leak = np.isin(va["hashes"], tr["hashes"])
    result = {
        "partial_train": args.max_files is not None,
        "text": {"train": text_stats(tr), "validation": text_stats(va)},
        "images": {"train": image_stats(tr), "validation": image_stats(va)},
        "caption_image_conflicts": {"train": duplicate_caption_conflicts(tr)},
        "row_id": {
            "train_unique": int(len(np.unique(tr["row_id"]))), "train_min": int(tr["row_id"].min()),
            "train_max": int(tr["row_id"].max()),
            "val_unique": int(len(np.unique(va["row_id"]))), "val_min": int(va["row_id"].min()),
            "val_max": int(va["row_id"].max()),
        },
        "overlap": {
            "val_subjects_not_in_train": len(v_sub - t_sub),
            "val_captions_in_train": len(v_cap & t_cap), "val_captions_total": len(v_cap),
            "val_images_exactly_in_train": int(leak.sum()), "val_images_total": int(va["n"]),
        },
    }
    (RUNS_ROOT / "stats.json").write_text(json.dumps(result, indent=2, default=lambda o: o.item()))
    plot_all(tr, va, result["text"])
    print(f"wrote {RUNS_ROOT}")


if __name__ == "__main__":
    main()
