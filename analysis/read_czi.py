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


# channel_conversion = {
#     "DAPI": "",
#     "AF488": "",
#     "AF555_2": "PAX2",
#     "AF660": "DBX1",
#     "AF751": "VGAT"
# }



# fallback color if color in metadata 
FALLBACK_COLOR = (255, 255, 255)  # white / grayscale

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
   
    with czifile.CziFile(file_path) as czi:
        arr = czi.asarray()
        squeezed = np.squeeze(arr)
        num_channels = squeezed.shape[0] if squeezed.ndim == 3 else 1  # enforce 3 dimensions 
        channel_info = get_channel_info(czi, num_channels) #  pull names and colors of channels 

        print(f"[INFO] File: {file_path}")
        print(f"[INFO] Squeezed shape: {squeezed.shape}  dtype: {squeezed.dtype}")

    if squeezed.ndim != 3:  # ensure that we have 3 dimensions in our czi, rest of script assumes we do 
        print(f"Unexpected ndim={squeezed.ndim}, shape={squeezed.shape}. Stopping.")
        return

    planes = [squeezed[i].astype(np.uint16) for i in range(num_channels)]
    channel_names = []
    channel_colors = []
    for i in range(num_channels):
        if i < len(channel_info):
            name, color = channel_info[i]
        else:
            name, color = None, None
        channel_names.append(name or f"ch{i+1}")
        channel_colors.append(color or FALLBACK_COLOR) 

    print("\nChannel colors (read from ZEN's own saved metadata, not guessed):")
    for i, (name, color) in enumerate(zip(channel_names, channel_colors)):
        if i < len(channel_info) and channel_info[i][1] is not None:
            source = "from metadata"
        else:
            source = "FALLBACK -- not found in metadata"
        print(f"  ch{i+1} ({name}): RGB{color}  [{source}]")

    # Build one full-resolution 16-bit multichannel stack.
    # Pixel values remain unchanged.
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
    print(f"\n--- Preview overlay (8-bit, display only, NOT for measurements) ---")
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

    print("\nSummary:")
    print("  - *_ALLCHANNELS_16bit_FIJI.tif           : same data, opens pre-colored in Fiji.")
    print("  - *_PREVIEW_COMPOSITE.png                : 8-bit, quick visual check only.")
    print("  - All colors above came from ZEN's saved metadata, not from a guess.")

    if select_background: # if we have activated the background flag measurement 
        
        background_csv = os.path.join(out_dir,f"{base}_BACKGROUND.csv")
        
        if (os.path.exists(background_csv) and not overwrite_background): # guard so we dont have to do the entire process again if background csv exists 
            print("  Background already measured, skipping")
        
        else:
            points = select_background_points(stack, base, number_of_points=background_regions)

            if not points:
                print("No background points saved.")

            else:
                measurements = measure_background_points(stack, points, channel_names)
                save_background_results_points(out_dir, base, points, measurements)

                print("Background measurements:")
                for row in measurements:
                    print(
                        f"{row['channel_name']}: "
                        f"mean="
                        f"{row['background_mean']:.2f}, "
                        f"median="
                        f"{row['background_median']:.2f}, "
                        f"std="
                        f"{row['background_std']:.2f}")


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
 
 


def measure_background_points(stack, points, channel_names, patch_radius=0):
    """
    Measure ORIGINAL 16-bit fluorescence values at each clicked point,
    per channel.

    patch_radius=0  -> use the exact single pixel clicked
    patch_radius=N  -> average a (2N+1) x (2N+1) patch centered on the
                       click, which is less noisy than a lone pixel
    """

    results = []
    height, width = stack.shape[-2:]

    for index, (plane, channel_name) in enumerate(
        zip(stack, channel_names), start=1
    ):
        point_values = []

        for x, y in points:
            if patch_radius == 0:
                point_values.append(float(plane[y, x]))
            else:
                y0, y1 = max(0, y - patch_radius), min(height, y + patch_radius + 1)
                x0, x1 = max(0, x - patch_radius), min(width, x + patch_radius + 1)
                patch = plane[y0:y1, x0:x1].astype(np.float64)
                point_values.append(float(np.mean(patch)))

        results.append(
            {
                "channel_index": index,
                "channel_name": channel_name,
                "n_points": len(point_values),
                "point_values": point_values,
                "background_mean": float(np.mean(point_values)),
                "background_median": float(np.median(point_values)),
                "background_std": float(np.std(point_values))})

    return results

def select_background_points(stack, title, number_of_points=1):
    """
    Let the user click one or more background points.
    Returns a list of (x, y) integer pixel coordinates.
    """
 
    display = make_background_display(stack)  # reuse existing helper
    points = []
 
    for point_number in range(1, number_of_points + 1):
        print(f"Click background point {point_number}/{number_of_points}")
 
        point = select_background_point(
            display,
            f"{title} — Background point {point_number}/{number_of_points}",
        )
 
        if point is None:
            print("Background selection cancelled.")
            break
 
        points.append(point)
 
    return points

 
def save_background_results_points(out_dir, base, points, measurements, patch_radius=0):
    """
    Save:
      1. clicked point coordinates (+ patch radius used)
      2. per-channel measurements
    No mask TIFF needed anymore since there's no region to visualize.
    """

    json_path = os.path.join(out_dir, f"{base}_BACKGROUND_POINTS.json")
    csv_path = os.path.join(out_dir, f"{base}_BACKGROUND.csv")

    with open(json_path, "w") as handle:
        json.dump(
            {
                "points_xy": points,
                "number_of_points": len(points),
                "patch_radius": patch_radius,
            },
            handle,
            indent=2,
        )

    fieldnames = [
        "channel_index",
        "channel_name",
        "n_points",
        "background_mean",
        "background_median",
        "background_std",
    ]

    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in measurements:
            writer.writerow({key: row[key] for key in fieldnames})

    print(f"  background points: {json_path}")
    print(f"  background data:   {csv_path}")







def save_background_results(
    out_dir,
    base,
    background_mask,
    vertices,
    measurements,
):
    """
    Save:
      1. background ROI mask
      2. polygon coordinates
      3. per-channel measurements
    """

    mask_path = os.path.join(
        out_dir,
        f"{base}_BACKGROUND_MASK.tif",
    )

    json_path = os.path.join(
        out_dir,
        f"{base}_BACKGROUND_ROI.json",
    )

    csv_path = os.path.join(
        out_dir,
        f"{base}_BACKGROUND.csv",
    )

    tifffile.imwrite(

        mask_path,

        background_mask.astype(
            np.uint8
        ),

        compression="zlib",
    )

    with open(
        json_path,
        "w",
    ) as handle:

        json.dump(
            {
                "regions_xy":
                    vertices,

                "number_of_regions":
                    len(vertices),

                "mask_pixels":
                    int(
                        background_mask.sum()
                    ),
            },
            handle,
            indent=2,
        )

    with open(
        csv_path,
        "w",
        newline="",
    ) as handle:

        writer = csv.DictWriter(

            handle,

            fieldnames=[
                "channel_index",
                "channel_name",
                "n_pixels",
                "background_mean",
                "background_median",
                "background_std",
                "background_min",
                "background_max",
            ],
        )

        writer.writeheader()

        writer.writerows(
            measurements
        )

    print(f"  background mask: {mask_path}")
    print(f"  background data: {csv_path}")
    print(f"  background ROI:  {json_path}")

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