"""
measure_background.py

Measure per-channel background fluorescence for every sample in a slide.

Input:
    Cellpose QC TIFFs containing:
        C1 DAPI
        C2 EVX1
        C3 PAX2
        C4 DBX1
        C5 VGAT
        C6 Cellpose boundary
        C7 Cellpose ID

Only channels 1-5 are analysed.

For each sample, background is estimated from the low-intensity tail of
pixels inside the spinal-cord tissue mask.

Output:
    One CSV per slide, with one row per sample/channel.

Example:
    python HiPlexUp-V0d-Pipeline/analysis/measure_background.py \
        cellpose_masks/Slide_139 \
        --mask-dir tissue_masks_139
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import tifffile


#  only collect actual staining not cellpose 
CHANNEL_NAMES = ["DAPI", "EVX1", "PAX2", "DBX1", "VGAT"]
N_CHANNELS = 5


# each cellpose file has the same suffix just varies by sample 
CELLPOSE_SUFFIX = "_cellpose_QC_16bit_FIJI"
PERCENTILE = 10.0


def find_cellpose_tiffs(folder):
    """
    find cell pose tiff files to analyse 
    """


    return sorted(p for p in folder.rglob("*.tif") if CELLPOSE_SUFFIX in p.stem)


def original_stem(cellpose_path):
    """
    fnid cellpose mask stem 
    """
    stem = cellpose_path.stem

    if stem.endswith(CELLPOSE_SUFFIX):
        stem = stem[:-len(CELLPOSE_SUFFIX)]

    return stem


def find_tissue_mask(mask_dir, stem):
    """
    find tissue mask per sample 
    """
    candidates = list(
        mask_dir.rglob(f"{stem}_tissue_mask.tif"))

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) == 0:
        return None
    
    raise RuntimeError(f"Multiple tissue masks found for {stem}")


def measure_channel(plane, tissue_mask, percentile):
    """
    for each channel calculate background fluorescence stats 
    """
    
    
    values = plane[tissue_mask]

    # Ignore zero-valued fill/background pixels.
    values = values[values > 0]

    if values.size == 0:
        raise ValueError("No usable pixels inside tissue mask")

    cutoff = np.percentile(values,percentile)  # find the 10th percentile of the mean (montana rosell et al., 2024)
    low_tail = values[values <= cutoff]

    return {
        "pixels_in_tissue": int(values.size),
        "background_pixels": int(low_tail.size),
        "cord_median": float(np.median(values)),
        "cord_mean": float(np.mean(values)),
        "background_percentile": percentile,
        "background_cutoff": float(cutoff),
        "background_median": float(np.median(low_tail)),
        "background_mean": float(np.mean(low_tail)),
        "background_std": float(np.std(low_tail)),
        "background_min": float(np.min(low_tail)),
        "background_max": float(np.max(low_tail))}

def measure_sample(tiff_path, mask_path, percentile):
    """
    main pipeline function running and calculating stats per sample 
    """
    
    
    stack = tifffile.imread(tiff_path)
    tissue_mask = tifffile.imread(mask_path).astype(bool)

    if stack.ndim != 3:
        raise ValueError(f"Expected CYX stack, got {stack.shape}")

    if stack.shape[0] < N_CHANNELS:
        raise ValueError(f"Expected at least 5 channels, got {stack.shape[0]}")

    if stack.shape[1:] != tissue_mask.shape:
        raise ValueError(f"Image shape {stack.shape[1:]} does not match mask shape {tissue_mask.shape}")

    rows = []
    for idx, channel_name in enumerate(CHANNEL_NAMES):

        plane = stack[idx]
    
        stats = measure_channel(plane, tissue_mask, percentile)

        rows.append({
            "Sample": original_stem(tiff_path),
            "Channel": channel_name,
            "Channel_number": idx + 1,
            **stats})
        
    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("cellpose_dir", type=Path)
    parser.add_argument( "--mask-dir", type=Path, required=True)
    parser.add_argument("--percentile",type=float, default=PERCENTILE)
    parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()
    tiffs = find_cellpose_tiffs(args.cellpose_dir)

    if not tiffs:
        raise SystemExit(
            f"No Cellpose QC TIFFs found under "
            f"{args.cellpose_dir}"
        )

    print(
        f"Found {len(tiffs)} Cellpose TIFFs"
    )

    all_rows = []
    failed = []

    for i, tiff_path in enumerate(tiffs, start=1):

        stem = original_stem(tiff_path)

        print(f"[{i}/{len(tiffs)}] {stem}")

        mask_path = find_tissue_mask(args.mask_dir, stem)

        if mask_path is None:
            print("[SKIP] Tissue mask not found")
            failed.append(stem)
            continue

        try:
            rows = measure_sample(
                tiff_path,
                mask_path,
                args.percentile)

        except Exception as error:
            print(f"  [ERROR] {error}")
            failed.append(stem)
            continue

        all_rows.extend(rows)
        for row in rows:
            print(
                f"  {row['Channel']:<5} "
                f"background={row['background_median']:.1f} "
                f"SD={row['background_std']:.1f} "
                f"cutoff={row['background_cutoff']:.1f}")

    if not all_rows:
        raise SystemExit("No samples were successfully processed.")

    df = pd.DataFrame(all_rows)

    if args.output is None:
        output_path = (
            args.cellpose_dir
            / f"{args.cellpose_dir.name}_background.csv")
    else:
        output_path = args.output

    df.to_csv(output_path,index=False)

    print(
f"Saved: {output_path}")

    print(
        f"Samples processed: "
        f"{df['Sample'].nunique()}")

    if failed:
        print(
            f"\nSamples needing attention: "
            f"{len(failed)}")

        for stem in failed:
            print(f"  - {stem}")


if __name__ == "__main__":
    main()