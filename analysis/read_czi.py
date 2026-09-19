"""
read_czi.py

Converts per-scene CZI files into:

1. One losslessly compressed 16-bit multichannel TIFF for Fiji/image analysis.
   Pixel intensities are preserved exactly; no downsampling is performed.

2. One contrast-stretched 8-bit composite PNG for visual inspection only.
   Do NOT use the preview PNG for quantitative measurements.

The original CZI remains the master/raw image.
"""

import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import czifile
import tifffile
from PIL import Image
import glob

import argparse
import csv
import json
import matplotlib.pyplot as plt 

CHANNEL_CONFIG = {
    "DAPI": {
        "index": 0,
        "channel_number": 1,
        "color": (0, 0, 255),        # blue
    },
    "EVX1": {
        "index": 1,
        "channel_number": 2,
        "color": (0, 255, 255),      # cyan
    },
    "PAX2": {
        "index": 2,
        "channel_number": 3,
        "color": (255, 0, 255),      # magenta
    },
    "DBX1": {
        "index": 3,
        "channel_number": 4,
        "color": (255, 255, 0),      # yellow
    },
    "VGAT": {
        "index": 4,
        "channel_number": 5,
        "color": (0, 255, 0),        # green
    },
}

# if enable_background flag is activated, collect fluorescence from each channel x,y 
BACKGROUND_POINTS = {
    "DAPI": {"x": 8101, "y": 5102}, # TO DO COLLECT X,Y COORDS .
    "EVX1": {"x": 7964, "y": 4981},
    "PAX2": {"x": 8180, "y": 5055},
    "DBX1": {"x": 8022, "y": 5110},
    "VGAT": {"x": 8138, "y": 4923}}


CHANNEL_NAMES = list(CHANNEL_CONFIG.keys())
FALLBACK_COLOR = (255, 255, 255)    # fallback color if color in metadata 


# 0 = exactly one pixel
# 1 = 3x3 patch around selected point
# 2 = 5x5 patch around selected point
BACKGROUND_PATCH_RADIUS = 3


def parse_zen_color(hex_str):
    """
    Function to handle ZEN stores channel colors as a hex string, typically ARGB
    (e.g. '#FF00FF00' = full alpha, green) or as a plain RGB ('#00FF00'). 
    """
    if not hex_str:
        return None
    
    h = hex_str.lstrip("#")
    try:
        if len(h) == 8:       # AARRGGBB hex 
            r, g, b = int(h[2:4], 16), int(h[4:6], 16), int(h[6:8], 16)
        elif len(h) == 6:     # RRGGBB rgb 
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        else:
            return None
        
        return (r, g, b)
    
    except ValueError:
        return None


def get_channel_info(czi, num_channels):
    """
    Read (name, color) for each channel straight from metadata, acquisition-time color assignment metadata 
    """

    entries = []  # list of (name, color_rgb_or_None), in first-seen order
    seen_names = set()
    try:
        meta_raw = czi.metadata()
        if isinstance(meta_raw, bytes):
            meta_raw = meta_raw.decode("utf-8", errors="ignore") # read using utf-8 string 
        root = ET.fromstring(meta_raw)  # convert XML tree 
        for ch in root.iter("Channel"): # search for any labels with unique entries channel
            name = ch.get("Name")
            if not name or name in seen_names:
                continue
            color_el = ch.find("Color") # search for child color tag inside the channel 
            color = parse_zen_color(color_el.text) if color_el is not None else None # parse it into an rgb format we can display 
            entries.append((name, color))
            seen_names.add(name)
    except Exception as e:
        print(f"  (Could not parse channel metadata: {e})")

    return entries[:num_channels]


def mirrored_output_dir(file_path, base):
    """
    Preserve input structure, so we can utilise it for the output structure 
    """

    parts = os.path.normpath(file_path).split(os.sep)   
    
    if "raw_czi" in parts:
        idx = parts.index("raw_czi") #  locates source folder 
        sub = parts[idx + 1:-1]  #  grab every subfolder after 'raw czi' 
    else:
        sub = []
    return os.path.join("raw_png", *sub, f"{base}_full") #  construct new path identical to input with raw_png 


def stretch_to_8bit(plane, lo_pct=1, hi_pct=99):
    """
    Enhance the visibility of a 16-bit microscopy slide to 8-bit resolution so we can preview 
    """

    lo, hi = np.percentile(plane, (lo_pct, hi_pct)) # clip the 16-bit to low and high percentile 
    stretched = np.clip((plane.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255) # clip the window to those percentiles 
    
    return stretched.astype(np.uint8) # output a lightweight unit8 png 


def build_lut(color):
    """
    This function builds an imageJ/fjij comptabile LUT (lookup table), tells software exactly how to paint gray pixels with a specific color.
    """
    lut = np.zeros((3, 256), dtype=np.uint8)  #  (3,256) unit array rgb 
    for c in range(3):
        lut[c] = np.linspace(0, color[c], 256, dtype=np.uint8)  #  linspace smooth linear ramp for each color 
    return lut # changes how the image looks in fiji without changing the source czi 


def inspect_and_export(file_path,select_background=False,background_regions=1,overwrite_background=False):
    """
    coordinator function acting as the main processor of czi reading and conversion 
    """


    if not os.path.exists(file_path):
        print(f"Error: file not found: {file_path}")
        return

    base = os.path.splitext(os.path.basename(file_path))[0]
    out_dir = mirrored_output_dir(file_path, base)
    os.makedirs(out_dir, exist_ok=True) #  build output dir 

    print("Reading CZI array...")
   
    with czifile.CziFile(file_path) as czi:  #  with a singular czi file open... 
        arr = czi.asarray()
        squeezed = np.squeeze(arr)
        num_channels = squeezed.shape[0] if squeezed.ndim == 3 else 1  # enforce 3 dimensions 
        
        if num_channels != 5: 
            raise RuntimeError("Expected exactly 5 channels from the .czi file") # expect 5 channels exactly given our hard-coded dict. 
        
        channel_info = get_channel_info(czi, num_channels) #  pull names and colors of channels 

        print(f"[INFO] File: {file_path}")
        print(f"[INFO] Squeezed shape: {squeezed.shape}  dtype: {squeezed.dtype}")

    if squeezed.ndim != 3:  # ensure that we have 3 dimensions in our czi, rest of script assumes we do 
        print(f"Unexpected ndim={squeezed.ndim}, shape={squeezed.shape}. Stopping.")
        return

    planes = [squeezed[i].astype(np.uint16) for i in range(num_channels)]
   
    # Biological channel identity is fixed by acquisition order.
    channel_names = CHANNEL_NAMES.copy()

    channel_colors = [CHANNEL_CONFIG[name]["color"] for name in channel_names]  # pull out channel colour from dict 

    print("\nConfirmed biological channel mapping:")

    for name in channel_names:  # debugging terminal for channel name / rgb 
        config = CHANNEL_CONFIG[name]
        print(f"  Channel {config['channel_number']}: {name} (NumPy index {config['index']}) RGB{config['color']}")


    # Build one full-resolution 16-bit multichannel stack.
    stack = np.stack(planes, axis=0)  # (C, Y, X)

    # Save one losslessly compressed Fiji-ready 16-bit TIFF.
    # LUTs affect display only and do not alter pixel intensities.
    luts = [build_lut(c) for c in channel_colors]

    out_fiji = os.path.join(out_dir, f"{base}_ALLCHANNELS_16bit_FIJI.tif")

    #  build a lossless compressed 16-bit multi-channel stack with a personalised LUT for fiji analysis 
    tifffile.imwrite(out_fiji, stack, imagej=True, compression="zlib", compressionargs={"level": 6}, 
        metadata={
            "axes": "CYX",
            "mode": "composite",
            "LUTs": luts})

    print(f"Fiji-colored stack: {out_fiji}")

    # after writing tiff read it back, so we get a tiff-array and then a czi-derived array to check 
    check = tifffile.imread(out_fiji)

    if not np.array_equal(stack, check):
        raise RuntimeError("TIFF verification failed: saved pixels differ from source!")

    print("Pixel verification: PASS — TIFF matches source exactly.")

    # Preview composite image 8-bit, using ZEN's real colors, display only.
    print(f"Preview overlay (8-bit) (for QC-checks)")
    composite = np.zeros((*planes[0].shape, 3), dtype=np.uint8)
    
    for plane, color in zip(planes, channel_colors):
        stretched8 = stretch_to_8bit(plane)
        tinted = np.zeros((*stretched8.shape, 3), dtype=np.uint8)
        for c in range(3):
            if color[c] > 0:
                tinted[..., c] = (stretched8.astype(np.uint16) * color[c] // 255).astype(np.uint8)
        composite = np.maximum(composite, tinted)
    out_preview = os.path.join(out_dir, f"{base}_PREVIEW_COMPOSITE.png")
    Image.fromarray(composite).save(out_preview)
    print(f"  {out_preview}")

    if select_background:

        print("Measuring channel-specific background fluorescence")
        background_csv = os.path.join(out_dir,f"{base}_BACKGROUND.csv")

        if (os.path.exists(background_csv) and not overwrite_background):
            print("Background already measured, skipping.")

        else:

            measurements = measure_background_from_coordinates(stack=stack, background_points=BACKGROUND_POINTS, patch_radius=BACKGROUND_PATCH_RADIUS)
            save_background_measurements(out_dir=out_dir,base=base,measurements=measurements)

            print("Background measurements from original 16-bit pixels:")

            for row in measurements:
                print(
                    f"  {row['channel_name']:<5} "
                    f"Ch{row['channel_number']}  "
                    f"XY=({row['x_pixel']}, {row['y_pixel']})"
                    f"value/mean={row['background_mean']:.2f}")


def find_czi_files(folder):
    """Recursively find every .czi file under folder, sorted for consistent order for batch processing. """
    return sorted(glob.glob(os.path.join(folder, "**", "*.czi"), recursive=True))


def make_background_display(stack):
    """
    Helper function to create a display-only 8-bit image for selecting background for the user, we take the max projection so the colours are bright to the user for selecting the background. 
    """
    display_channels = []
    for plane in stack:
        display_channels.append(stretch_to_8bit(plane))
    return np.max(np.stack(display_channels,axis=0),axis=0)


def select_background_point(display, title):
    """
    Interactively click ONE background point.
 
    Click once on the image, then close the window (or press any key)
    to confirm. Closing without clicking cancels.
    """
 
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(display, cmap="gray")
    ax.set_title(
        f"{title}\n"
        "Click one point in a representative tissue-background region.\n"
        "One click only — the window will close automatically."
    )
    ax.axis("off")
    plt.tight_layout()
 
    # ginput blocks until n points are clicked or the window is closed
    pts = plt.ginput(n=1, timeout=0)
    plt.close(fig)
 
    if not pts:
        return None
 
    x, y = pts[0]
    return int(round(x)), int(round(y))
 
 
 def measure_background_from_coordinates(stack, background_points, patch_radius=0):
    """
    Measure background fluorescence independently for each biological channel.
    Each channel has its own Fiji-selected X/Y coordinate hard-coded at the start of the script. 
    Returns ->> list of dictionaries containing per-channel background measurements.
    """

    height = stack.shape[1]
    width = stack.shape[2]
    results = []

    for channel_name, config in CHANNEL_CONFIG.items():
        channel_index = config["index"]

        if channel_name not in background_points:
            raise ValueError(f"No background coordinate provided for {channel_name}")

        x = int(background_points[channel_name]["x"]) # collect x,y coordinates definted by user 
        y = int(background_points[channel_name]["y"])

        # Make sure coordinate is inside image.
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(
                f"{channel_name} background point "
                f"(x={x}, y={y}) is outside image dimensions "
                f"{width} x {height}")
        plane = stack[channel_index]

        if patch_radius == 0:
            values = np.array([plane[y, x]],dtype=np.float64)  # radius point we collect one pixel if set to 0

        else:
            x0 = max(0, x - patch_radius) # adjust x,y based on patch_radius and collect min max
            x1 = min(width, x + patch_radius + 1)
            y0 = max(0, y - patch_radius)
            y1 = min(height, y + patch_radius + 1)
            values = plane[y0:y1,x0:x1].astype(np.float64).ravel()

        results.append({ # save all results 
                "channel_number":config["channel_number"],
                "channel_name":channel_name,
                "channel_index":channel_index,
                "x_pixel":x,
                "y_pixel":y,
                "patch_radius":patch_radius,
                "n_pixels":int(values.size),
                "background_mean": float(np.mean(values)), 
                "background_median":float(np.median(values)),
                "background_std":float(np.std(values)),
                "background_min":float(np.min(values)),
                "background_max":float(np.max(values))})

    return results



def save_background_measurements(out_dir, base, measurements):
    """
    Save the Fiji-selected coordinates and corresponding original
    16-bit fluorescence measurements.
    """


    csv_path = os.path.join(out_dir,f"{base}_BACKGROUND.csv")
    json_path = os.path.join(out_dir, f"{base}_BACKGROUND.json")

    fieldnames = [
        "channel_number",
        "channel_name",
        "channel_index",
        "x_pixel",
        "y_pixel",
        "patch_radius",
        "n_pixels",
        "background_mean",
        "background_median",
        "background_std",
        "background_min",
        "background_max"]

    with open(csv_path,"w",newline="",) as handle:  # handle csv writing over 

        writer = csv.DictWriter(handle,fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(measurements)

    with open(json_path,"w") as handle:

        json.dump({
                "source_image":base,
                "background_points":BACKGROUND_POINTS,
                "patch_radius":BACKGROUND_PATCH_RADIUS,
                "measurements": measurements},handle,indent=2)

    print(f"Background CSV:{csv_path}")
    print(f"Background JSON: {json_path}")


# main caller 
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # mac os  /Users/angusgray/Desktop/V0d-Histology/HiPlexUp-V0d-Pipeline/raw_czi
    # windows  /home/gray2/Desktop/HiPlexUp/HiPlexUp-V0d-Pipeline/raw_czi

    parser.add_argument("target",nargs="?",default=("/home/gray2/Desktop/HiPlexUp/HiPlexUp-V0d-Pipeline/raw_czi")) # select czi file location 
    parser.add_argument("--select_background",action="store_true")
    parser.add_argument("--background_regions",type=int,default=1)
    parser.add_argument("--overwrite_background",action="store_true")

    args = parser.parse_args()

    if os.path.isfile(args.target):

        if not args.target.lower().endswith(".czi"):
            raise SystemExit("Input file is not a CZI.")

        files = [args.target]

    else:

        files = find_czi_files(args.target)

    if not files:

        print(f"No .czi files found under {args.target}")
        sys.exit(0)

    print(f"Found {len(files)} CZI file(s)")
    failed = []
    for i, path in enumerate(
        files,
        start=1):

        print(
            f"\n{'#' * 60}"
            f"\n[{i}/{len(files)}] "
            f"{path}"
            f"\n{'#' * 60}"
        )

        try:

            inspect_and_export(path,
                select_background=args.select_background,
                background_regions=args.background_regions,
                overwrite_background=args.overwrite_background)

        except Exception as error:

            print(f"ERROR processing {path}: {error}")
            failed.append(path)

    print(f"\n{'=' * 60}")
    print(
        f"Batch complete: "
        f"{len(files) - len(failed)}/"
        f"{len(files)} succeeded"
    )

    if failed:

        print(
            "Failed files:"
        )

        for path in failed:

            print(
                f"  - {path}"
            )