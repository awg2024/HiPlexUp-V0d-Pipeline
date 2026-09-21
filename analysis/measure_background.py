"""
measure_background.py

Estimate per-channel background fluorescence INSIDE the spinal cord ROI, using
the original 16-bit pixels (no downsampling, no 8-bit stretching).

Inputs (matched by filename stem)
    <stem>.tif                              16-bit CYX stack written by read_czi.py
    tissue_masks/<stem>_tissue_mask.tif     ROI written by tissue_roi.py

Outputs;;;
    background_measurements/<stem>_background.csv    one row per channel
    background_measurements/<stem>_background.json   same data + parameters

Definition of background here is the calculation of a background median. median of the in-cord pixels that fall at or below the
Nth percentile of that channel's in-cord intensities (default N = 10). Noise is deliberately not removed, it will be cancelled out during normalisation.
cord_median (median of ALL in-cord pixels) is reported alongside for reference, however it is not background it includes every positive cell and every bright nucleus in the cord.
background_std / _min / _max describe only the lower tail (pixels <= the percentile cutoff), so they are truncated and understate true noise.

Examples on how to use the script 
    python HiPlexUp-V0d-Pipeline/analysis/measure_background.py raw_png
    python HiPlexUp-V0d-Pipeline/analysis/measure_background.py "raw_png/Slide_141/.../..._FIJI.tif"
    python HiPlexUp-V0d-Pipeline/analysis/measure_background.py raw_png --percentile 5 --overwrite 
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


import numpy as np
import tifffile

CHANNEL_NAMES = ["DAPI", "EVX1", "PAX2", "DBX1", "VGAT"]   # acquisition order, same as read_czi.py
NBINS = 65536                                              # every possible uint16 intensity
LEVELS = np.arange(NBINS, dtype=np.float64)
SENSITIVITY_PCTS = (5, 10, 25)                             # reported so you can see how stable the estimate is

# histogram statistics
def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Median of `values`, where each value occurs `weights` times."""
    order = np.argsort(values, kind="stable")
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, 0.5 * cum[-1])])


def hist_percentile(counts: np.ndarray, pct: float) -> int:
    """Nearest-rank percentile (an intensity value) from a histogram."""
    cdf = np.cumsum(counts)
    return int(np.searchsorted(cdf, pct / 100.0 * cdf[-1]))


def low_tail_stats(counts: np.ndarray, pct: float) -> dict:
    """Statistics of the pixels at or below the `pct` percentile."""
    cutoff = hist_percentile(counts, pct)
    sub = counts[: cutoff + 1].astype(np.float64)
    lv = LEVELS[: cutoff + 1]
    n = sub.sum()
    mean = float((lv * sub).sum() / n)
    nonzero = np.nonzero(sub)[0]
    return {
        "cutoff": cutoff,
        "n": int(n),
        "median": weighted_median(lv, sub),
        "mean": mean,
        "std": float(np.sqrt((((lv - mean) ** 2) * sub).sum() / n)),
        "min": float(lv[nonzero[0]]),
        "max": float(lv[nonzero[-1]]),
    }


def masked_histogram(plane: np.ndarray, mask: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """Intensity histogram of plane[mask], built in row chunks to limit memory."""
    counts = np.zeros(NBINS, dtype=np.int64)
    for y0 in range(0, plane.shape[0], chunk):
        vals = plane[y0:y0 + chunk][mask[y0:y0 + chunk]]
        counts += np.bincount(vals, minlength=NBINS)
    return counts

# file handling
def find_tiffs(input_path: Path):
    """Find TIFFs recursively, skipping preview images."""
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.rglob("*.tif") if "PREVIEW" not in p.name.upper())


def iter_channels(tf: tifffile.TiffFile):
    """Yield one 2D channel at a time (one page per channel in read_czi.py's TIFFs)."""
    series = tf.series[0]
    if len(tf.pages) == series.shape[0]:
        for i in range(series.shape[0]):
            yield tf.pages[i].asarray()
    else:                                    # fallback: whole stack in memory
        for plane in series.asarray():
            yield plane


def measure_one(tiff_path: Path, mask_path: Path, pct: float, keep_zeros: bool):
    mask = tifffile.imread(mask_path).astype(bool)
    if not mask.any():
        raise ValueError(f"tissue mask is empty: {mask_path}")

    rows = []
    with tifffile.TiffFile(tiff_path) as tf:
        shape = tf.series[0].shape
        if len(shape) != 3 or shape[0] != len(CHANNEL_NAMES):
            raise ValueError(f"expected ({len(CHANNEL_NAMES)}, Y, X), got {shape}")
        if tuple(shape[1:]) != mask.shape:
            raise ValueError(f"image YX {tuple(shape[1:])} != mask YX {mask.shape}")

        pcts = sorted(set(SENSITIVITY_PCTS) | {pct})

        for idx, plane in enumerate(iter_channels(tf)):
            if plane.dtype != np.uint16:
                raise ValueError(f"channel {idx + 1} is {plane.dtype}, expected uint16")

            counts = masked_histogram(plane, mask)
            n_zero = int(counts[0])
            if not keep_zeros:
                counts[0] = 0            # zero-valued pixels are fill/gaps, not tissue signal
            n_used = int(counts.sum())
            if n_used == 0:
                raise ValueError(f"channel {idx + 1}: no usable pixels inside the mask")

            main = low_tail_stats(counts, pct)
            row = {
                "channel_number": idx + 1,
                "channel_name": CHANNEL_NAMES[idx],
                "channel_index": idx,
                "mask_pixels": int(mask.sum()),
                "zero_pixels_in_mask": n_zero,
                "pixels_used": n_used,
                "cord_median": weighted_median(LEVELS, counts.astype(np.float64)),
                "cord_mean": float((LEVELS * counts).sum() / n_used),
                "background_percentile": pct,
                "background_cutoff": main["cutoff"],
                "background_median": main["median"],
                "background_mean": main["mean"],
                "background_std": main["std"],
                "background_min": main["min"],
                "background_max": main["max"],
            }
            for p in pcts:
                row[f"background_median_p{p:g}"] = low_tail_stats(counts, p)["median"]
            rows.append(row)

    return rows


def save_results(out_dir: Path, stem: str, rows: list, meta: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}_background.csv"
    json_path = out_dir / f"{stem}_background.json"

    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(json.dumps({**meta, "measurements": rows}, indent=2))
    return csv_path, json_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", nargs="?", type=Path, default=Path("raw_png"),
                        help="16-bit TIFF file, or a folder searched recursively")
    parser.add_argument("--mask-dir", type=Path, default=Path("tissue_masks"))
    parser.add_argument("--output-dir", type=Path, default=Path("background_measurements"))
    parser.add_argument("--percentile", type=float, default=10.0,
                        help="background = median of in-cord pixels at or below this percentile")
    parser.add_argument("--keep-zeros", action="store_true",
                        help="include exact-zero pixels inside the mask (excluded by default)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0 < args.percentile < 100:
        raise SystemExit("--percentile must be between 0 and 100")

    files = find_tiffs(args.target)
    if not files:
        raise SystemExit(f"No TIFF files found under {args.target}")
    print(f"Found {len(files)} TIFF(s)")

    failed = []
    for i, path in enumerate(files, start=1):
        print(f"\n[{i}/{len(files)}] {path.name}")
        stem = path.stem
        mask_path = args.mask_dir / f"{stem}_tissue_mask.tif"
        csv_path = args.output_dir / f"{stem}_background.csv"

        if csv_path.exists() and not args.overwrite:
            print(f"  [SKIP] {csv_path} already exists")
            continue
        if not mask_path.exists():
            print(f"  [SKIP] no tissue mask at {mask_path} (run tissue_roi.py first)")
            failed.append(path)
            continue

        try:
            rows = measure_one(path, mask_path, args.percentile, args.keep_zeros)
        except Exception as error:
            print(f"  [ERROR] {error}")
            failed.append(path)
            continue

        meta = {"source_tiff": str(path), "tissue_mask": str(mask_path),
                "percentile": args.percentile, "zeros_excluded": not args.keep_zeros}
        save_results(args.output_dir, stem, rows, meta)

        print(f"  mask pixels: {rows[0]['mask_pixels']:,}")
        print(f"  {'channel':<6} {'cord median':>12} {'background':>11} {'cutoff':>7} {'zeros':>10}")
        for r in rows:
            print(f"  {r['channel_name']:<6} {r['cord_median']:>12.0f} {r['background_median']:>11.0f} "
                  f"{r['background_cutoff']:>7d} {r['zero_pixels_in_mask']:>10,}")
        print(f"  [OK] {csv_path}")

    print(f"\nDone. {len(files) - len(failed)}/{len(files)} processed without problems.")
    if failed:
        print("Needs attention:")
        for p in failed:
            print(f"  - {p}")


if __name__ == "__main__":
    main()