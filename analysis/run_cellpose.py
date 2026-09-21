#!/usr/bin/env python3

"""
run_cellpose.py

Run Cellpose segmentation on converted 16-bit multichannel TIFF images.

Cellpose in this V0d histological workflow is run on VGAT (channel 5), identifying
cells across the samples before quantification/analysis of V0ds can take place.

Outputs;;
masks/<stem>_cellpose_mask.tif
masks/<stem>_cellpose_overlay.png
masks/<stem>_cellpose_run.json

Progress display
    - one numbered line per stage, with how long each stage took
    - a live bar during the Cellpose step (tiles processed, elapsed, estimated remaining)
    - after every slide: how long it took and an estimate for the rest of the batch
    The live bar is only drawn in an interactive terminal. In a batch job that writes to
    a log file, only the stage lines and per-slide timings are printed.

Examples on how to utilise this script

# Isolated image tiff, channel 5 (note: --tissue-mask-dir takes a FOLDER, not a file)
python run_cellpose.py \
    "raw_png/Slide_141/.../..._ALLCHANNELS_16bit_FIJI.tif" \
    --tissue-mask-dir tissue_masks \
    --channels 5

# Process a directory, channel 5
python run_cellpose.py \
    raw_png/Slide_141 \
    --tissue-mask-dir tissue_masks \
    --channels 5

# Utilise GPU (only if the job actually has one)
python run_cellpose.py \
    raw_png/Slide_141 \
    --tissue-mask-dir tissue_masks \
    --channels 5 \
    --use-gpu
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from tqdm import tqdm

from cellpose import models

# progress helpers
def fmt_time(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s" if hours else f"{minutes}m {secs:02d}s"


@contextmanager
def stage(step: int, total_steps: int, label: str):
    """Print a numbered stage line, then how long it took (only if it succeeded)."""
    print(f"  [{step}/{total_steps}] {label} ...", flush=True)
    t0 = time.perf_counter()
    yield
    print(f"      done in {fmt_time(time.perf_counter() - t0)}", flush=True)


@contextmanager
def live_bar(desc: str, total: int | None):
    """
    tqdm bar that is refreshed every second by a helper thread, so the elapsed time keeps
    ticking even while Cellpose is busy between tiles.
    disable=None -> the bar is switched off automatically when output is not a terminal.
    """
    bar = tqdm(total=total, desc=desc, unit="tile", leave=False, disable=None, dynamic_ncols=True)
    stop = threading.Event()

    def tick():
        while not stop.wait(1.0):
            bar.refresh()

    thread = threading.Thread(target=tick, daemon=True)
    thread.start()
    try:
        yield bar
    finally:
        stop.set()
        thread.join()
        bar.close()


def estimate_tiles(shape_yx, diameter, bsize=256, tile_overlap=0.1) -> int:
    """
    Approximate number of 256x256 tiles Cellpose will push through the network.
    This is an ESTIMATE (it mirrors Cellpose's tiling rule), so the percentage is approximate.
    """
    scale = 30.0 / diameter if diameter else 1.0
    ny = max(2, int(np.ceil((1.0 + 2 * tile_overlap) * shape_yx[0] * scale / bsize)))
    nx = max(2, int(np.ceil((1.0 + 2 * tile_overlap) * shape_yx[1] * scale / bsize)))
    return ny * nx


def run_cellpose_eval(model, image, channel_axis, diameter, **eval_kwargs):
    """
    Run model.eval() with a live progress bar.

    Cellpose has no per-tile progress for a single image (its `progress` argument only serves
    the GUI), so we count how many tiles pass through the network with a forward hook.
    The hook can never break the run: if anything about it fails, it is ignored and the
    bar just shows elapsed time.
    """
    total = estimate_tiles(image.shape[:2], diameter)
    with live_bar("Cellpose (tiles; % is approximate)", total) as bar:

        def hook(module, inputs, output):
            try:
                bar.update(int(inputs[0].shape[0]))
                if bar.n > bar.total:      # estimate was low: extend the bar instead of overflowing
                    bar.total = bar.n
            except Exception:
                pass

        handle = None
        try:
            handle = model.net.register_forward_hook(hook)
        except Exception:
            bar.set_description("Cellpose (no tile counter, elapsed only)")

        try:
            masks, _flows, _styles = model.eval(image, channel_axis=channel_axis,
                                                diameter=diameter, **eval_kwargs)
        finally:
            if handle is not None:
                handle.remove()

    return np.asarray(masks)

# image helpers
def load_cyx(path: Path) -> np.ndarray:
    """
    Load the .tiff and return it in expected format for processing (C,Y,X; channel, height, width)
    """

    arr = tifffile.imread(path)

    if arr.ndim == 2:
        return arr[np.newaxis, ...]

    if arr.ndim != 3:
        raise ValueError(f"{path}: expected 2D/3D TIFF got {arr.shape}")

    if arr.shape[0] <= 16:
        return arr

    if arr.shape[-1] <= 16:
        return np.moveaxis(arr, -1, 0)

    raise ValueError(f"{path}: cannot infer channel axis from shape {arr.shape}")


def stretch_to_8bit(plane, lo_pct=1, hi_pct=99):
    """
    Similar helper function we've used in the past; TO DO start importing this in instead of defining it!!
    """

    plane = plane.astype(np.float32)
    lo, hi = np.percentile(plane, (lo_pct, hi_pct))

    if hi <= lo:
        return np.zeros(plane.shape, dtype=np.uint8)

    result = np.clip((plane - lo) / (hi - lo) * 255, 0, 255)
    return result.astype(np.uint8)


def prepare_cellpose_input(stack, channels_1based):
    """
    Select the single channel we send to Cellpose (VGAT = channel 5) from the C,Y,X stack.
    """

    if len(channels_1based) != 1:      # was `not 1 != len(...)`, which raised for exactly one channel
        raise ValueError("Please select exactly one channel for Cellpose processing (e.g. --channels 5).")

    index = channels_1based[0] - 1

    if not 0 <= index < stack.shape[0]:
        raise ValueError(f"Requested channel {channels_1based[0]}, but TIFF contains {stack.shape[0]} channels.")

    return stack[index], None          # 2D image, no channel axis


def apply_tissue_mask(image, mask):
    """
    Zero everything outside the tissue ROI before segmentation.
    """

    if mask.shape != image.shape[:2]:
        raise ValueError(f"Tissue mask shape {mask.shape} does not match image XY {image.shape[:2]}")

    output = image.copy()
    output[~mask] = 0
    return output


def relabel_sequential(mask):
    """
    Ensure labels are sequential so we can build row-wise cell counts:
    0 = background, 1 = cell 1, 2 = cell 2, ...
    Uses a lookup table, so it is one pass over the image regardless of how many cells there are.
    """

    present = np.unique(mask)
    present = present[present != 0]                    # if there are no labels, the output is all zeros

    lut = np.zeros(int(mask.max()) + 1, dtype=np.uint32)
    lut[present] = np.arange(1, present.size + 1, dtype=np.uint32)

    return lut[mask]


def mask_boundaries(mask):
    """
    Boolean image that is True on the boundary pixels between labels.
    """

    boundaries = np.zeros(mask.shape, dtype=bool)

    boundaries[1:, :] |= (mask[1:, :] != mask[:-1, :])
    boundaries[:-1, :] |= (mask[:-1, :] != mask[1:, :])
    boundaries[:, 1:] |= (mask[:, 1:] != mask[:, :-1])
    boundaries[:, :-1] |= (mask[:, :-1] != mask[:, 1:])

    return boundaries & (mask > 0)


def save_overlay(display_plane, mask, output_path):
    """
    Save a QC image showing the Cellpose boundaries (yellow) over the fluorescence image.
    QC only: no measurements should be taken from this image.
    """

    gray = stretch_to_8bit(display_plane)
    rgb = np.repeat(gray[..., None], 3, axis=-1)

    edges = mask_boundaries(mask)
    rgb[edges] = np.array([255, 255, 0], dtype=np.uint8)   # yellow boundaries

    plt.imsave(output_path, rgb)


# --------------------------------------------------------------------------
# file handling
# --------------------------------------------------------------------------

def find_tiffs(input_path):
    """
    Find .tiff files that are not a preview or mask from other scripts.
    """

    if input_path.is_file():
        return [input_path]

    return sorted(path for path in input_path.rglob("*.tif")
                  if "PREVIEW" not in path.name.upper() and "MASK" not in path.name.upper())


def find_tissue_mask(tiff_path, tissue_mask_dir):
    """
    Locate <tissue_mask_dir>/<stem>_tissue_mask.tif written by tissue_roi.py.
    """

    if tissue_mask_dir is None:
        return None

    candidate = tissue_mask_dir / f"{tiff_path.stem}_tissue_mask.tif"
    return candidate if candidate.exists() else None


def process_one(tiff_path, model, output_dir, channels, tissue_mask_dir, diameter,
                flow_threshold, cellprob_threshold, min_size, overwrite):
    """
    Process one .tiff with the Cellpose pipeline. Returns True if the slide was processed,
    False if it was skipped (so the batch ETA only uses slides that actually ran).
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / f"{tiff_path.stem}_cellpose_mask.tif"
    overlay_path = output_dir / f"{tiff_path.stem}_cellpose_overlay.png"
    run_path = output_dir / f"{tiff_path.stem}_cellpose_run.json"

    if mask_path.exists() and not overwrite:
        print(f"[SKIP] {mask_path} since it already exists")
        return False

    # The tissue ROI is required in this workflow: skip instead of segmenting empty slide.
    tissue_mask_path = find_tissue_mask(tiff_path, tissue_mask_dir)
    if tissue_mask_path is None:
        print(f"[SKIP] no tissue mask for {tiff_path.name} in {tissue_mask_dir} (run tissue_roi.py first)")
        return False

    t_slide = time.perf_counter()
    n_steps = 6

    with stage(1, n_steps, "Loading TIFF and tissue mask"):
        stack = load_cyx(tiff_path)
        cp_image, channel_axis = prepare_cellpose_input(stack, channels)
        tissue_mask = tifffile.imread(tissue_mask_path).astype(bool)
        cp_image = apply_tissue_mask(cp_image, tissue_mask)
        print(f"      tissue mask: {tissue_mask_path.name}  image: {cp_image.shape[1]} x {cp_image.shape[0]} px")

    with stage(2, n_steps, "Cellpose segmentation (the slow step)"):
        # normalize=True rescales the image by its own intensity percentiles (see model.eval docs).
        masks = run_cellpose_eval(model, cp_image, channel_axis, diameter,
                                  flow_threshold=flow_threshold,
                                  cellprob_threshold=cellprob_threshold,
                                  min_size=min_size,
                                  normalize=True)

    with stage(3, n_steps, "Enforcing tissue ROI and relabelling"):
        masks = masks.copy()
        masks[~tissue_mask] = 0            # cells outside the ROI are removed after segmentation
        masks = relabel_sequential(masks)

    with stage(4, n_steps, "Writing mask TIFF"):
        tifffile.imwrite(mask_path, masks.astype(np.uint32), compression="zlib")

    with stage(5, n_steps, "Writing QC overlay"):
        display_plane = stack[channels[0] - 1]     # first selected channel is the QC background
        save_overlay(display_plane, masks, overlay_path)

    with stage(6, n_steps, "Writing run info"):
        run_info = {
            "source_tiff": str(tiff_path),
            "channels_1_based": channels,
            "n_cells": int(masks.max()),
            "diameter": diameter,
            "flow_threshold": flow_threshold,
            "cellprob_threshold": cellprob_threshold,
            "min_size_pixels": min_size,
            "tissue_mask": str(tissue_mask_path),
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
            "device": str(getattr(model, "device", "unknown")),
            "seconds": round(time.perf_counter() - t_slide, 1)}
        run_path.write_text(json.dumps(run_info, indent=2))

    print(f"[OK] {tiff_path.name}: {int(masks.max())} detected cell masks")
    print(f"     mask: {mask_path}")
    print(f"     overlay: {overlay_path}")
    return True


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)                                   # one analysis .tiff or a dir of .tiffs
    parser.add_argument("--output-dir", type=Path, default=Path("masks"))
    parser.add_argument("--tissue-mask-dir", type=Path, required=True)        # FOLDER produced by tissue_roi.py
    parser.add_argument("--channels", type=int, nargs="+", required=True)     # channel 5 == VGAT
    parser.add_argument("--model", default="cpsam_v2")                        # needs cellpose >= 4.2
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--diameter", type=float, default=None)

    # cellpose parameters
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--min-size", type=int, default=15)                   # min mask size in PIXELS
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    files = find_tiffs(args.input)

    if not files:
        raise SystemExit(f"No TIFF files found under {args.input}")

    print(f"Loading Cellpose model: {args.model}")
    model = models.CellposeModel(gpu=args.use_gpu, pretrained_model=args.model)
    print(f"Running on: {getattr(model, 'device', 'unknown')}")
    if args.use_gpu and not getattr(model, "gpu", False):
        print("[WARN] --use-gpu was requested but no GPU was found. Running on CPU will be very slow.")

    print(f"Found {len(files)} TIFF(s)")

    ran_times = []
    failed = []
    for i, path in enumerate(files, start=1):
        print(f"\nrun_cellpose.py progress: [{i}/{len(files)}] {path}")
        t0 = time.perf_counter()

        try:
            ran = process_one(
                path,
                model=model,
                output_dir=args.output_dir,
                channels=args.channels,
                tissue_mask_dir=args.tissue_mask_dir,
                diameter=args.diameter,
                flow_threshold=args.flow_threshold,
                cellprob_threshold=args.cellprob_threshold,
                min_size=args.min_size,
                overwrite=args.overwrite)
        except Exception as error:               # one bad slide should not end a long batch
            print(f"[ERROR] {path.name}: {error}")
            traceback.print_exc()
            failed.append(path)
            continue

        if ran:
            ran_times.append(time.perf_counter() - t0)
            remaining = len(files) - i
            print(f"  slide took {fmt_time(ran_times[-1])}; {remaining} file(s) left, "
                  f"estimated {fmt_time(np.mean(ran_times) * remaining)} remaining (upper bound: skips are quicker)")

    print(f"\nDone. Processed {len(ran_times)} slide(s) in {fmt_time(sum(ran_times))}.")
    if failed:
        print("Failed:")
        for path in failed:
            print(f"  - {path}")


if __name__ == "__main__":
    main()