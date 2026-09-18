#!/usr/bin/env python3

"""
run_cellpose.py

Run Cellpose segmentation on converted
16-bit multichannel TIFF images.

Cellpose in this V0d hisotological workflow will be ran on the VGAT (channel 5) 

Outputs
-------
masks/<stem>_cellpose_mask.tif
masks/<stem>_cellpose_overlay.png
masks/<stem>_cellpose_run.json

Examples
--------

# Use channel 1
python run_cellpose.py \
    converted_tiff/image.tif \
    --channels 5

# Process a directory
python run_cellpose.py \
    converted_tiff \
    --channels 1

# Use two channels
python run_cellpose.py \
    converted_tiff \
    --channels 1 3

# Use tissue masks ROI!!
python run_cellpose.py \
    converted_tiff \
    --channels 1 \
    --tissue-mask-dir tissue_masks

# Try GPU / Apple MPS
python run_cellpose.py \
    converted_tiff \
    --channels 1 \
    --use-gpu
"""


# do not start feeding in your four channels into cell pose
# current cellpose-sam tkaes up the first three supplied channels and its documentation reccomends starting with the cytoplasmic/nuclear channels rather than simply giving its fluorescent stains 


# what channels is VGAT? hmm let's get that classifed. 
# python scripts/run_cellpose.py raw_png/...tif --channels vgat channel --use-gpu 


from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile

from cellpose import models


def load_cyx(
    path: Path,
) -> np.ndarray:
    """
    Return TIFF as C,Y,X.
    """

    arr = tifffile.imread(path)

    if arr.ndim == 2:
        return arr[np.newaxis, ...]

    if arr.ndim != 3:
        raise ValueError(
            f"{path}: expected 2D/3D TIFF, "
            f"got {arr.shape}"
        )

    if arr.shape[0] <= 16:
        return arr

    if arr.shape[-1] <= 16:
        return np.moveaxis(
            arr,
            -1,
            0,
        )

    raise ValueError(
        f"{path}: cannot infer channel "
        f"axis from shape {arr.shape}"
    )


def stretch_to_8bit(
    plane,
    lo_pct=1,
    hi_pct=99,
):

    plane = plane.astype(
        np.float32
    )

    lo, hi = np.percentile(
        plane,
        (lo_pct, hi_pct),
    )

    if hi <= lo:

        return np.zeros(
            plane.shape,
            dtype=np.uint8,
        )

    result = np.clip(
        (plane - lo)
        / (hi - lo)
        * 255,
        0,
        255,
    )

    return result.astype(
        np.uint8
    )


def prepare_cellpose_input(
    stack,
    channels_1based,
):
    """
    Select 1-3 channels from the C,Y,X stack.

    Cellpose 4 no longer requires the old
    channels=[cytoplasm, nucleus] argument.

    Instead, we construct the exact image channels
    we want to send to Cellpose.
    """

    if not 1 <= len(channels_1based) <= 3:

        raise ValueError(
            "Choose between 1 and 3 "
            "channels for Cellpose."
        )

    indices = [
        channel - 1
        for channel
        in channels_1based
    ]

    if (
        min(indices) < 0
        or max(indices) >= stack.shape[0]
    ):

        raise ValueError(
            f"Requested channels "
            f"{channels_1based}, "
            f"but TIFF contains "
            f"{stack.shape[0]} channels."
        )

    selected = stack[
        indices
    ]

    # One-channel image.
    if selected.shape[0] == 1:

        return (
            selected[0],
            None,
        )

    # Convert C,Y,X -> Y,X,C
    image = np.moveaxis(
        selected,
        0,
        -1,
    )

    return (
        image,
        -1,
    )


def apply_tissue_mask(
    image,
    mask,
):

    if mask.shape != image.shape[:2]:

        raise ValueError(
            f"Tissue mask shape {mask.shape} "
            f"does not match image XY "
            f"{image.shape[:2]}"
        )

    output = image.copy()

    if output.ndim == 2:

        output[~mask] = 0

    else:

        output[~mask, :] = 0

    return output


def relabel_sequential(
    mask,
):
    """
    Ensure labels are:

    0 = background
    1 = cell 1
    2 = cell 2
    ...
    """

    labels = np.unique(mask)

    labels = labels[
        labels != 0
    ]

    output = np.zeros(
        mask.shape,
        dtype=np.uint32,
    )

    for new_id, old_id in enumerate(
        labels,
        start=1,
    ):

        output[
            mask == old_id
        ] = new_id

    return output


def mask_boundaries(
    mask,
):

    boundaries = np.zeros(
        mask.shape,
        dtype=bool,
    )

    boundaries[1:, :] |= (
        mask[1:, :]
        != mask[:-1, :]
    )

    boundaries[:-1, :] |= (
        mask[:-1, :]
        != mask[1:, :]
    )

    boundaries[:, 1:] |= (
        mask[:, 1:]
        != mask[:, :-1]
    )

    boundaries[:, :-1] |= (
        mask[:, :-1]
        != mask[:, 1:]
    )

    return (
        boundaries
        & (mask > 0)
    )


def save_overlay(
    display_plane,
    mask,
    output_path,
):
    """
    Save a QC image showing the Cellpose
    boundaries over the fluorescence image.

    This image is for QC only.
    """

    gray = stretch_to_8bit(
        display_plane
    )

    rgb = np.repeat(
        gray[..., None],
        3,
        axis=-1,
    )

    edges = mask_boundaries(
        mask
    )

    rgb[edges] = np.array(
        [255, 0, 0],
        dtype=np.uint8,
    )

    plt.imsave(
        output_path,
        rgb,
    )


def find_tiffs(
    input_path,
):

    if input_path.is_file():
        return [input_path]

    return sorted(
        path
        for path
        in input_path.rglob("*.tif")
        if "PREVIEW"
        not in path.name.upper()
        and "MASK"
        not in path.name.upper()
    )


def find_tissue_mask(
    tiff_path,
    tissue_mask_dir,
):

    if tissue_mask_dir is None:
        return None

    candidate = (
        tissue_mask_dir
        / (
            f"{tiff_path.stem}"
            "_tissue_mask.tif"
        )
    )

    if candidate.exists():
        return candidate

    return None


def process_one(
    tiff_path,
    model,
    output_dir,
    channels,
    tissue_mask_dir,
    diameter,
    flow_threshold,
    cellprob_threshold,
    min_size,
    overwrite,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask_path = (
        output_dir
        / (
            f"{tiff_path.stem}"
            "_cellpose_mask.tif"
        )
    )

    overlay_path = (
        output_dir
        / (
            f"{tiff_path.stem}"
            "_cellpose_overlay.png"
        )
    )

    run_path = (
        output_dir
        / (
            f"{tiff_path.stem}"
            "_cellpose_run.json"
        )
    )

    if (
        mask_path.exists()
        and not overwrite
    ):

        print(
            f"[SKIP] {mask_path} "
            "already exists"
        )

        return

    stack = load_cyx(
        tiff_path
    )

    cp_image, channel_axis = (
        prepare_cellpose_input(
            stack,
            channels,
        )
    )

    tissue_mask_path = (
        find_tissue_mask(
            tiff_path,
            tissue_mask_dir,
        )
    )

    tissue_mask = None

    if tissue_mask_path is not None:

        tissue_mask = (
            tifffile.imread(
                tissue_mask_path
            ).astype(bool)
        )

        cp_image = apply_tissue_mask(
            cp_image,
            tissue_mask,
        )

        print(
            f"Using tissue mask: "
            f"{tissue_mask_path}"
        )

    elif tissue_mask_dir is not None:

        print(
            "[WARN] No tissue mask found "
            f"for {tiff_path.name}. "
            "Using full image."
        )

    masks, flows, styles = model.eval(

        cp_image,

        channel_axis=channel_axis,

        diameter=diameter,

        flow_threshold=flow_threshold,

        cellprob_threshold=cellprob_threshold,

        min_size=min_size,

        normalize=True,
    )

    masks = np.asarray(
        masks
    )

    # Enforce tissue region after segmentation.
    if tissue_mask is not None:

        masks = masks.copy()

        masks[
            ~tissue_mask
        ] = 0

    masks = relabel_sequential(
        masks
    )

    tifffile.imwrite(

        mask_path,

        masks.astype(
            np.uint32
        ),

        compression="zlib",
    )

    # First selected channel used only
    # as background for QC overlay.
    display_plane = stack[
        channels[0] - 1
    ]

    save_overlay(
        display_plane,
        masks,
        overlay_path,
    )

    run_info = {

        "source_tiff":
            str(tiff_path),

        "channels_1_based":
            channels,

        "n_cells":
            int(masks.max()),

        "diameter":
            diameter,

        "flow_threshold":
            flow_threshold,

        "cellprob_threshold":
            cellprob_threshold,

        "min_size_pixels":
            min_size,

        "tissue_mask":
            (
                str(tissue_mask_path)
                if tissue_mask_path
                else None
            ),

        "mask_path":
            str(mask_path),

        "overlay_path":
            str(overlay_path),
    }

    run_path.write_text(
        json.dumps(
            run_info,
            indent=2,
        )
    )

    print(
        f"[OK] {tiff_path.name}: "
        f"{int(masks.max())} "
        "detected cell masks"
    )

    print(
        f"     mask: {mask_path}"
    )

    print(
        f"     overlay: {overlay_path}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input",
        type=Path,
        help=(
            "One analysis TIFF or "
            "directory of TIFFs."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("masks"),
    )

    parser.add_argument(
        "--tissue-mask-dir",
        type=Path,
        default=None,
        help=(
            "Directory produced by "
            "tissue_roi.py. Optional."
        ),
    )

    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        required=True,
        help=(
            "1-based channel number(s). "
            "Example: --channels 1 "
            "or --channels 1 3"
        ),
    )

    parser.add_argument(
        "--model",
        default="cpsam_v2",
    )

    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help=(
            "Use available GPU / Apple "
            "MPS acceleration."
        ),
    )

    parser.add_argument(
        "--diameter",
        type=float,
        default=None,
        help=(
            "Approximate cell diameter "
            "in pixels. Initially leave "
            "unset."
        ),
    )

    parser.add_argument(
        "--flow-threshold",
        type=float,
        default=0.4,
    )

    parser.add_argument(
        "--cellprob-threshold",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--min-size",
        type=int,
        default=15,
        help=(
            "Minimum Cellpose mask size "
            "in PIXELS, not um^2."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    files = find_tiffs(
        args.input
    )

    if not files:

        raise SystemExit(
            f"No TIFF files found "
            f"under {args.input}"
        )

    print(
        f"Loading Cellpose model: "
        f"{args.model}"
    )

    model = models.CellposeModel(

        gpu=args.use_gpu,

        pretrained_model=args.model,
    )

    print(
        f"Found {len(files)} TIFF(s)"
    )

    for i, path in enumerate(
        files,
        start=1,
    ):

        print(
            f"\n[{i}/{len(files)}] "
            f"{path}"
        )

        process_one(

            path,

            model=model,

            output_dir=args.output_dir,

            channels=args.channels,

            tissue_mask_dir=
                args.tissue_mask_dir,

            diameter=args.diameter,

            flow_threshold=
                args.flow_threshold,

            cellprob_threshold=
                args.cellprob_threshold,

            min_size=args.min_size,

            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()

