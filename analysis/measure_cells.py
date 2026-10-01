"""
measure_cells.py

Generate a row-wise cell dataset using:

1. Cellpose per-cell CSV
2. Cellpose 7-channel QC TIFF
3. Slide background CSV

For each segmented cell:
- preserve Cell_ID and Classification
- measure fluorescence inside the Cellpose boundary
- subtract the sample/channel background

Output:
    one combined CSV with one row per cell
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import tifffile

CHANNEL_NAMES = ["DAPI", "EVX1", "PAX2", "DBX1", "VGAT"]


def find_matching_file(folder, stem, suffix):
    matches = list(folder.rglob(f"{stem}{suffix}"))

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        return None

    raise RuntimeError(f"Multiple matches found for {stem}{suffix}")


def load_background(background_csv):
    """
    ensure background measurements are present and load csv 
    """
    
    bg = pd.read_csv(background_csv)
    required = {"Sample","Channel","background_median"}

    missing = required - set(bg.columns)

    if missing:
        raise ValueError(f"Background CSV missing columns: {sorted(missing)}")
    return bg


def background_value(bg_df, sample, channel):
    """
    pull out sample specific background value for an individual channel 
    """
    
    # search criteria 
    row = bg_df[(bg_df["Sample"] == sample) & (bg_df["Channel"] == channel)]

    if len(row) != 1:
        raise ValueError(f"Expected one background value for {sample} / {channel}, found {len(row)}")

    return float(
        row.iloc[0]["background_median"])


def measure_cell_channel(values, background):
    
    # extract measurements from the channel 
    values = values.astype(np.float64)
    mean_raw = float(np.mean(values))
    median_raw = float(np.median(values))
    std_raw = float(np.std(values))

    return {
        "mean_raw": mean_raw,
        "median_raw": median_raw,
        "std_raw": std_raw,
        "mean_corrected": mean_raw - background,
        "median_corrected": median_raw - background}


def measure_sample(cells_csv, qc_tiff, background_df):
    """
    Main loop for loading cell csv, tiff file and background 
    """

    cells = pd.read_csv(cells_csv)
    stack = tifffile.imread( qc_tiff)

    if stack.ndim != 3 or stack.shape[0] < 7:
        raise ValueError(
            f"Expected 7-channel CYX TIFF, got {stack.shape}"
        )

    fluorescence = stack[:5]
    label_mask = stack[6]

    sample = qc_tiff.stem.replace("_cellpose_QC_16bit_FIJI", "")
    output_rows = []

    # for each sample in cells (cells refer to the csv files here) 
    for _, cell in cells.iterrows():

        label_id = int(cell["Label_ID"])
        cell_mask = (label_mask == label_id)

        if not cell_mask.any():
            print(f"[WARN] {cell['Cell_ID']} not found in label mask")
            continue

        row = cell.to_dict()
        for channel_idx, channel_name in enumerate(CHANNEL_NAMES):

            # extract out values, background and stats 
            values = fluorescence[channel_idx][cell_mask]
            background = background_value(background_df, sample, channel_name)
            stats = measure_cell_channel(values, background)

            # specify channel for stats output 
            row[f"{channel_name}_background"] = background
            row[f"{channel_name}_mean_raw"] = stats["mean_raw"]
            row[f"{channel_name}_median_raw"] = stats["median_raw"]
            row[f"{channel_name}_std_raw"] = stats["std_raw"]
            row[f"{channel_name}_mean_corrected"] = stats["mean_corrected"]
            row[f"{channel_name}_median_corrected"] = stats["median_corrected"]

        output_rows.append(
            row
        )

    return output_rows


def main():
    parser = argparse.ArgumentParser()

    # parser arguments the user can pass in 
    parser.add_argument("--cells-dir", type=Path, required=True)
    parser.add_argument("--tiff-dir", type=Path, required=True)
    parser.add_argument("--background-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    background_df = load_background(args.background_csv)

    cell_csvs = sorted(args.cells_dir.glob("*_cellpose_cells.csv"))

    if not cell_csvs:
        raise SystemExit(f"No cell CSVs found in {args.cells_dir}")

    all_rows = []

    # enumerate all csv provided 
    for i, cells_csv in enumerate(cell_csvs, start=1):

        stem = cells_csv.stem.replace("_cellpose_cells","")
        qc_tiff = find_matching_file(args.tiff_dir, stem, "_cellpose_QC_16bit_FIJI.tif")

        if qc_tiff is None:
            print(f"[SKIP] No QC TIFF for {stem}")
            continue

        print(f"[{i}/{len(cell_csvs)}] {stem}")

        rows = measure_sample(cells_csv, qc_tiff, background_df) # call main function to process  
        all_rows.extend(rows)

        print(f"  measured {len(rows)} cells")

    if not all_rows:
        raise SystemExit("No cells were measured.")

    final_df = pd.DataFrame(all_rows) # write all rows to the final df 

    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output, index=False)

    print(f"Saved {len(final_df)} cells to:")
    print(args.output)

if __name__ == "__main__":
    main()