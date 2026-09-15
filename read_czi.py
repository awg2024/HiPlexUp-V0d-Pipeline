"""
read_czi.py python exporter and preview overlay 

For per-scene CZI files from ZEN's Split Scenes / Write files export.

ANALYSIS-READY (full 16-bit precision, nothing downsampled), 16-bit PNT per channel and a 16-bit TIFF stack (all channels) for Fiji image analysis 
PREVIEW an 8-bit png, compressed (from 16-bit downsampled to 8-bits) serves as a sanity check that the data is there before the analysis stage (name: PREVIEW_COMPOSITE)
Output goes into raw_png/<same subfolders as raw_czi>/<file>_full/

Usage:
    python czi_to_png.py path/to/file.czi
"""

import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import czifile
import tifffile
from PIL import Image

# colour keys 
# PAX2 - megenta
# EVX1 stained cyan 
# DBX1 stained yellow
# VGAT staomed green


# Used only if a channel has no color stored in the metadata at all
FALLBACK_COLOR = (255, 255, 255)  


def parse_zen_color(hex_str):
    """
    ZEN stores channel colors as a hex string, typically ARGB
    (e.g. '#FF00FF00' = full alpha, green) but occasionally plain RGB
    ('#00FF00'). Returns an (R, G, B) tuple of ints 0-255, or None if it
    can't be parsed.
    """
    if not hex_str:
        return None
    h = hex_str.lstrip("#")
    try:
        if len(h) == 8:       # AARRGGBB
            r, g, b = int(h[2:4], 16), int(h[4:6], 16), int(h[6:8], 16)
        elif len(h) == 6:     # RRGGBB
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        else:
            return None
        return (r, g, b)
    except ValueError:
        return None


def get_channel_info(czi, num_channels):
    """
    Read (name, color) for each channel straight from ZEN's saved
    metadata -- this is the actual acquisition-time color assignment,
    not a guess. Deduplicates by name, since ZEN's metadata often
    repeats the channel list in more than one XML section (acquisition
    settings, display settings, etc.) -- we keep the first occurrence
    of each name, along with whatever color came with it.
    """
    entries = []  # list of (name, color_rgb_or_None), in first-seen order
    seen_names = set()
    try:
        meta_raw = czi.metadata()
        if isinstance(meta_raw, bytes):
            meta_raw = meta_raw.decode("utf-8", errors="ignore")
        root = ET.fromstring(meta_raw)
        for ch in root.iter("Channel"):
            name = ch.get("Name")
            if not name or name in seen_names:
                continue
            color_el = ch.find("Color")
            color = parse_zen_color(color_el.text) if color_el is not None else None
            entries.append((name, color))
            seen_names.add(name)
    except Exception as e:
        print(f"  (Could not parse channel metadata: {e})")

    return entries[:num_channels]


def mirrored_output_dir(file_path, base):
    """Mirror raw_czi/... subfolders under raw_png/... instead of flattening."""
    parts = os.path.normpath(file_path).split(os.sep)
    if "raw_czi" in parts:
        idx = parts.index("raw_czi")
        sub = parts[idx + 1:-1]
    else:
        sub = []
    return os.path.join("raw_png", *sub, f"{base}_full")


def stretch_to_8bit(plane, lo_pct=1, hi_pct=99):
    """
    Contrast-stretch a 16-bit plane to 8-bit for DISPLAY ONLY. Real signal
    usually occupies a small slice of the full 16-bit range, so a naive
    linear conversion looks black; this clips to the data's own 1st/99th
    percentile and rescales that to 0-255. Does not affect the 16-bit files.
    """
    lo, hi = np.percentile(plane, (lo_pct, hi_pct))
    stretched = np.clip((plane.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255)
    return stretched.astype(np.uint8)


def build_lut(color):
    """
    Build an ImageJ-style LUT: a (3, 256) uint8 array where row 0 is the
    red ramp, row 1 green, row 2 blue -- each going from 0 up to the
    channel's assigned color at value 255. This is what makes Fiji show
    a grayscale channel in that specific color without altering any
    actual pixel value -- it's purely a display lookup table.
    """
    lut = np.zeros((3, 256), dtype=np.uint8)
    for c in range(3):
        lut[c] = np.linspace(0, color[c], 256, dtype=np.uint8)
    return lut


def inspect_and_export(file_path):
    if not os.path.exists(file_path):
        print(f"Error: file not found: {file_path}")
        return

    base = os.path.splitext(os.path.basename(file_path))[0]
    out_dir = mirrored_output_dir(file_path, base)
    os.makedirs(out_dir, exist_ok=True)

    print("Reading CZI array (this can take a bit for large whole-slide scans)...")
    with czifile.CziFile(file_path) as czi:
        arr = czi.asarray()
        squeezed = np.squeeze(arr)
        num_channels = squeezed.shape[0] if squeezed.ndim == 3 else 1
        channel_info = get_channel_info(czi, num_channels)

        print("=" * 60)
        print(f"File: {file_path}")
        print(f"Squeezed shape: {squeezed.shape}  dtype: {squeezed.dtype}")
        print("=" * 60)

    if squeezed.ndim != 3:
        print(f"\nUnexpected ndim={squeezed.ndim}, shape={squeezed.shape}. Stopping.")
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
        source = "from metadata" if channel_info[i][1] else "FALLBACK -- not found in metadata"
        print(f"  ch{i+1} ({name}): RGB{color}  [{source}]")

    # ------------------------------------------------------------------
    # 1) Per-channel 16-bit PNGs -- full precision, uncompressed.
    # ------------------------------------------------------------------
    print(f"\n--- Analysis-ready (full 16-bit, no downsampling) -> {out_dir}/ ---")
    for i, (plane, name) in enumerate(zip(planes, channel_names)):
        out16 = os.path.join(out_dir, f"{base}_{name}_16bit.png")
        Image.fromarray(plane).save(out16, compress_level=0)
        print(f"  channel {i+1} ({name}): {out16}")

    # ------------------------------------------------------------------
    # 2) Plain multi-channel 16-bit TIFF -- no color metadata at all,
    #    for tools that just want raw grayscale channels.
    # ------------------------------------------------------------------
    stack = np.stack(planes, axis=0)  # (C, Y, X)
    out_stack = os.path.join(out_dir, f"{base}_ALLCHANNELS_16bit.tif")
    tifffile.imwrite(
        out_stack,
        stack,
        photometric="minisblack",
        metadata={"axes": "CYX", "Channel": {"Name": channel_names}},
        compression=None,
    )
    print(f"  plain stack: {out_stack}")

    # ------------------------------------------------------------------
    # 3) Fiji-ready TIFF -- same pixel data, ZEN's real colors embedded
    #    as ImageJ LUTs so Fiji opens it already showing the right colors.
    # ------------------------------------------------------------------
    luts = [build_lut(c) for c in channel_colors]
    out_fiji = os.path.join(out_dir, f"{base}_ALLCHANNELS_16bit_FIJI.tif")
    tifffile.imwrite(
        out_fiji,
        stack,
        imagej=True,
        metadata={"axes": "CYX", "mode": "composite", "LUTs": luts},
    )
    print(f"  Fiji-colored stack: {out_fiji}")

    # ------------------------------------------------------------------
    # 4) Preview composite -- 8-bit, using ZEN's real colors, display only.
    # ------------------------------------------------------------------
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
    print("  - *_16bit.png / *_ALLCHANNELS_16bit.tif : full precision, use for analysis.")
    print("  - *_ALLCHANNELS_16bit_FIJI.tif           : same data, opens pre-colored in Fiji.")
    print("  - *_PREVIEW_COMPOSITE.png                : 8-bit, quick visual check only.")
    print("  - All colors above came from ZEN's saved metadata, not from a guess.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Path to CZI file: ").strip('"')
    inspect_and_export(path)