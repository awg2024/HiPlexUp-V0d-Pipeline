#!/usr/bin/env python3

"""
run_cellpose.py

Run Cellpose segmentation on converted 16-bit multichannel TIFF images.

Cellpose in this V0d hisotological workflow will be ran on the VGAT (channel 5), identifying a number of cells across the samples 
before quantification/analysis of V0ds can take place. 

Outputs;;
masks/<stem>_cellpose_mask.tif
masks/<stem>_cellpose_overlay.png
masks/<stem>_cellpose_run.json

Examples on how to utilise this script

# Isolated image tiff channel 5 run
python run_cellpose.py \
    raw_png/141-.../141-....tif \
    --channels 5

# Process a directory channel 5 run 
python run_cellpose.py \
    raw_png/Slide_141 \
    --channels 5

# Utilising pre-defined tissue masks ROIs with specified .tiff 
python run_cellpose.py \
    raw_png/141-.../141-....tif \
    --channels 5 \
    --tissue-mask-dir tissue_masks/141-.../141-....tiff

# Utilise GPU  setting 
python run_cellpose.py \
    converted_tiff \
    --channels 1 \
    --use-gpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile

from cellpose import models


def load_cyx(path: Path) -> np.ndarray:
    """
    Load the .tiff and return it in expected format for processing (C,Y,X; channel, width, height)
    """

    arr = tifffile.imread(path)

    if arr.ndim == 2:
        return arr[np.newaxis, ...]

    if arr.ndim != 3:
        raise ValueError(f"{path}: expected 2D/3D TIFF got {arr.shape}")

    if arr.shape[0] <= 16:
        return arr

    if arr.shape[-1] <= 16:
        return np.moveaxis(arr,-1,0)

    raise ValueError(f"{path}: cannot infer channel axis from shape {arr.shape}")


def stretch_to_8bit(plane,lo_pct=1, hi_pct=99):
    """
    Similar helper function we've used in the past; TO DO start importing this in instead of defining it!!
    """

    plane = plane.astype(np.float32)

    lo, hi = np.percentile(plane,(lo_pct, hi_pct))

    if hi <= lo:

        return np.zeros(plane.shape,dtype=np.uint8)

    result = np.clip((plane - lo) / (hi - lo) * 255, 0, 255)

    return result.astype(np.uint8)


def prepare_cellpose_input(stack, channels_1based):
    """
    Select channels from the C,Y,X stack. Construct the exact image channels we want to send to Cellpose.
    """

    if not 1 != len(channels_1based):
        raise ValueError("Please select one channel for cellpose processing.")

    indices = [channel - 1 for channel in channels_1based] # extract specified channels 

    if (min(indices) < 0 or max(indices) >= stack.shape[0]): # if we are selecting channels outside the rnage 
        raise ValueError(f"Requested channels {channels_1based}, but TIFF contains {stack.shape[0]} channels.")
        
    selected = stack[indices]

    # One-channel image, if we have selected one channel (common VGAT workflow we expect.)
    if selected.shape[0] == 1: # One-channel image.

        return (selected[0], None)

    # Convert C,Y,X -> Y,X,C for cellpose? data expects this format? 
    image = np.moveaxis(selected, 0, -1)
    return (image,-1)


def apply_tissue_mask(image,mask):
    """
    Apply generated mask by cellpose the image, overlaying the original image 
    """


    if mask.shape != image.shape[:2]:

        raise ValueError(f"Tissue mask shape {mask.shape} does not match image XY {image.shape[:2]}")
        

    output = image.copy()
    
    if output.ndim == 2: # two dimension for the channel mask, if cell pose was ran over multiple channels ? 
        output[~mask] = 0

    else:
        output[~mask, :] = 0

    return output


def relabel_sequential(mask):
    """
    Ensure labels format are, essential so we can begin constructing our large collection of row-wise cell counts 
    0 = background
    1 = cell 1
    2 = cell 2
    etc... 
    """

    labels = np.unique(mask)
    labels = labels[labels != 0] # if there is no labels (e.g., cellpose didnt label any cells? )

    output = np.zeros(mask.shape, dtype=np.uint32)

    for new_id, old_id in enumerate(labels, start=1):

        output[mask == old_id] = new_id

    return output


def mask_boundaries(mask):
    """
    loop through each boundaries of the mask shape and then apply it to the mask
    """

    boundaries = np.zeros(mask.shape,dtype=bool)

    boundaries[1:, :] |= (mask[1:, :] != mask[:-1, :])

    boundaries[:-1, :] |= (mask[:-1, :] != mask[1:, :])

    boundaries[:, 1:] |= (mask[:, 1:] != mask[:, :-1])

    boundaries[:, :-1] |= (mask[:, :-1] != mask[:, 1:])

    return (boundaries & (mask > 0))


def save_overlay(display_plane,mask,output_path):
    """
    Save a QC image showing the Cellpose
    boundaries over the fluorescence image.

    This image is for QC only, but it's important for the user to see classification was processed as expected 
    """

    gray = stretch_to_8bit(display_plane)

    rgb = np.repeat(gray[..., None], 3, axis=-1)

    edges = mask_boundaries(mask)

    rgb[edges] = np.array([255, 255, 0], dtype=np.uint8)  # specify color for the boundaries to display (yellow0)

    # should we specify a title of the image
    plt.set_title("QC-Cellpose output highlighted YELLOW cell boundaries") 
    plt.imsave(output_path, rgb)


def find_tiffs(input_path,):
    """
    Find .tiff files that are not a preview or mask from other scripts we are specifying for the raw_pngs/
    """

    if input_path.is_file():
        return [input_path]

    return sorted(path for path in input_path.rglob("*.tif") if "PREVIEW" not in path.name.upper() and "MASK" not in path.name.upper())


def find_tissue_mask(tiff_path, tissue_mask_dir):
    """
    This is dir saving function for the tissue mask we have just created from the cellpose predictions 
    """

    if tissue_mask_dir is None:
        return None

    candidate = (tissue_mask_dir / (f"{tiff_path.stem}_tissue_mask.tif"))

    if candidate.exists():
        return candidate

    return None


def process_one(tiff_path, model,output_dir,channels,tissue_mask_dir,diameter,flow_threshold,cellprob_threshold,min_size,overwrite):
    """
    Processing of one .tiff file with cellpose main pipeline that will be called repeatedly for cells, TO DO add a progress of processing in this function.
    """

    #  dir specification 
    output_dir.mkdir(parents=True,exist_ok=True)
    mask_path = (output_dir / (f"{tiff_path.stem}_cellpose_mask.tif"))
    overlay_path = (output_dir / (f"{tiff_path.stem}_cellpose_overlay.png"))
    run_path = (output_dir / (f"{tiff_path.stem}_cellpose_run.json"))
        
    if (mask_path.exists() and not overwrite):
        print(f"[SKIP] {mask_path} since it already exists") 
        return

    #  load tiff file into the cyx format for cellpose formatting 
    stack = load_cyx(tiff_path)

    cp_image, channel_axis = (prepare_cellpose_input(stack,channels))   # extract out specified channel and return cellpose format
    
    tissue_mask_path = (find_tissue_mask(tiff_path,tissue_mask_dir))   # dir helper function for cellpose output 
    tissue_mask = None

    if tissue_mask_path is not None: # if tissue mask is a path 

        tissue_mask = (tifffile.imread(tissue_mask_path).astype(bool)) # read mask output 
        cp_image = apply_tissue_mask(cp_image,tissue_mask) # we apply a tissue mask 
        print(f"Using tissue mask: {tissue_mask_path}")

    elif tissue_mask_dir is not None:

        print(f"[WARN] No tissue mask found for {tiff_path.name}. Using full image.")
        
    # run cell pose eval 
    masks, flows, styles = model.eval(cp_image, channel_axis=channel_axis, diameter=diameter, flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
        normalize=True) # do we need normalisation maybe leave this un-normalized and enable us to this aspect? figure out what this flag does in model.eval 

    masks = np.asarray(masks)

    # Enforce tissue region after segmentation.
    if tissue_mask is not None:
        masks = masks.copy()
        masks[~tissue_mask] = 0

    masks = relabel_sequential(masks)  # output dir with the row-wise segmentation 

    tifffile.imwrite(
        mask_path,
        masks.astype(np.uint32),
        compression="zlib") # output tiff with the mask overlay 

    # First selected channel used only
    # as background for QC overlay.
    display_plane = stack[channels[0] - 1]
    save_overlay(display_plane, masks, overlay_path) # save overlay with yellow boundaries 

    run_info = { # run info for each cellpose run, these are mostly static params for json dump 
        "source_tiff": str(tiff_path),
        "channels_1_based": channels,
        "n_cells": int(masks.max()),
        "diameter": diameter,
        "flow_threshold": flow_threshold,
        "cellprob_threshold": cellprob_threshold,
        "min_size_pixels": min_size,
        "tissue_mask": (str(tissue_mask_path) if tissue_mask_path else None),
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path)}

    run_path.write_text(json.dumps(run_info, indent=2))
        
    print(f"[OK] {tiff_path.name}: {int(masks.max())} detected cell masks")
    print(f"mask: {mask_path}")
    print(f"overlay: {overlay_path}")

def main():

    # specify args 
    parser = argparse.ArgumentParser()
    parser.add_argument("input",type=Path) # one analysis .tiff or a dir of .tiffs
    parser.add_argument("--output-dir", type=Path, default=Path("masks")) # specify what output dir
    parser.add_argument("--tissue-mask-dir", type=Path, required=True) # dir produced by tissue_roi.py (required for our workflow)
    parser.add_argument("--channels", type=int, nargs="+", required=True) # specify channel number you want for our workflow -- channel 5 == vgat 
    parser.add_argument("--model",default="cpsam_v2") # latest cellpose pre-trained model should work well on the well defined vgat boundaries 
    parser.add_argument("--use-gpu", action="store_true") # use MPS acceleration if gpu is avail, would this work on a compute node?
    parser.add_argument("--diameter", type=float, default=None) # we can specify what cell diameters to define as boundaries, this flag can be utilise on second-run if first classification results are not adequate 
    
    # cellpose paramters  
    parser.add_argument("--flow-threshold", type=float, default=0.4) 
    parser.add_argument( "--cellprob-threshold",type=float,default=0.0)
    parser.add_argument("--min-size", type=int, default=15) # min cellpose mask size in PIXELS, might need to adjust 
    parser.add_argument( "--overwrite", action="store_true") # if there is pre-existing masks there are we overwriting this? this flag should be needed since the slide141/139 have diff naming conventions 

    args = parser.parse_args()
    files = find_tiffs(args.input)

    if not files:
        raise SystemExit(f"No TIFF files found under {args.input}")

    print(f"Loading Cellpose model: {args.model}")
    model = models.CellposeModel(gpu=args.use_gpu, pretrained_model=args.model)

    print(f"Found {len(files)} TIFF(s)")

    for i, path in enumerate(files, start=1):
        print(f"run_cellpose.py progress: [{i}/{len(files)}] {path}") # progress output of dir  

        process_one( # process tiff files one by one with them placed in the loop 
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


if __name__ == "__main__":
    main()

