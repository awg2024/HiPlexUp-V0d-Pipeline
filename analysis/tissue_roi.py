
"""
Interactively define the tissue region to analyse in each converted
16-bit TIFF.

This script DOES NOT identify V0d cells.
It simply defines the part of the image in which Cellpose is allowed
to search for cells.

Outputs;;
tissue_masks/<stem>_tissue_mask.tif
tissue_masks/<stem>_tissue_roi.json

Modes;; 
interactive:
    Manually draw a polygon around the tissue region.

full:
    Treat the whole image as the analysis region.

Examples on how to use this script: 

# Draw tissue ROI manually
python tissue_roi.py converted_tiff/image.tif

# Process all TIFFs in a directory
python tissue_roi.py converted_tiff --output-dir tissue_masks

# Analyse the full image -- no tissue exclusion
python tissue_roi.py converted_tiff --mode full

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.widgets import PolygonSelector
import numpy as np
import tifffile


def load_cyx(path: Path) -> np.ndarray:
    """
    Load TIFF and return array with shape:

        C, Y, X
    """

    arr = tifffile.imread(path)

    if arr.ndim == 2:
        return arr[np.newaxis, ...]

    if arr.ndim != 3:
        raise ValueError(
            f"{path}: expected 2D or 3D TIFF, got shape {arr.shape}")

    # read_czi.py writes CYX.
    if arr.shape[0] <= 16:
        return arr

    # Fallback in case TIFF happens to be YXC.
    if arr.shape[-1] <= 16:
        return np.moveaxis(arr, -1, 0)

    raise ValueError(f"{path}: cannot determine channel axis from shape {arr.shape}")

def stretch_to_8bit(plane: np.ndarray,lo_pct: float = 1,hi_pct: float = 99) -> np.ndarray:
    """ 
    Same helper function utilised in read_czi.py to stretch the image out to an 8-bit for the user viewing. 
    """

    plane = plane.astype(np.float32)

    lo, hi = np.percentile(plane,(lo_pct, hi_pct))

    if hi <= lo:
        return np.zeros(plane.shape,dtype=np.uint8)

    stretched = np.clip((plane - lo) / (hi - lo) * 255,0,255)

    return stretched.astype(np.uint8)


def make_display(
    stack: np.ndarray,
    display_channel: int = 0,
) -> np.ndarray:
    """
    Produce a display-only 8-bit image.

    display_channel = 0
        max projection of independently stretched channels

    display_channel = 1, 2, ...
        show one specific channel

    This image is ONLY for drawing the ROI.
    No measurements are taken from it.
    """

    if display_channel == 0:

        stretched = [
            stretch_to_8bit(channel)
            for channel in stack
        ]

        return np.max(
            np.stack(stretched, axis=0),
            axis=0,
        )

    index = display_channel - 1

    if index < 0 or index >= stack.shape[0]:
        raise ValueError(
            f"Requested display channel {display_channel}, "
            f"but TIFF contains {stack.shape[0]} channels."
        )

    return stretch_to_8bit(
        stack[index]
    )


def polygon_to_mask(
    vertices,
    shape,
) -> np.ndarray:
    """
    Convert polygon XY vertices into a boolean YX mask.
    """

    yy, xx = np.mgrid[
        :shape[0],
        :shape[1],
    ]

    points = np.column_stack(
        (
            xx.ravel(),
            yy.ravel(),
        )
    )

    polygon = MplPath(vertices)

    mask = polygon.contains_points(
        points,
        radius=0.5,
    )

    return mask.reshape(shape)


def select_polygon(
    display: np.ndarray,
    title: str,
):
    """
    Interactive polygon selection.

    Controls
    --------
    Mouse:
        place / adjust polygon vertices

    ENTER:
        accept ROI

    ESC:
        cancel this image
    """

    state = {
        "vertices": None,
        "cancelled": False,
    }

    fig, ax = plt.subplots(
        figsize=(11, 8)
    )

    ax.imshow(
        display,
        cmap="gray",
    )

    ax.set_title(
        f"{title}\n"
        "Outline the tissue region. "
        "Press ENTER to accept or ESC to cancel."
    )

    ax.axis("off")

    selector = PolygonSelector(
        ax,
        lambda verts: None,
        useblit=True,
    )

    def on_key(event):

        if event.key in (
            "enter",
            "return",
        ):

            vertices = selector.verts

            if (
                vertices is not None
                and len(vertices) >= 3
            ):

                state["vertices"] = [
                    [float(x), float(y)]
                    for x, y in vertices
                ]

                plt.close(fig)

            else:
                print(
                    "Need at least three polygon "
                    "vertices before accepting."
                )

        elif event.key == "escape":

            state["cancelled"] = True
            plt.close(fig)

    fig.canvas.mpl_connect(
        "key_press_event",
        on_key,
    )

    plt.tight_layout()
    plt.show()

    if state["cancelled"]:
        return None

    return state["vertices"]


def find_tiffs(
    input_path: Path,
):
    """
    Find TIFFs recursively.
    """

    if input_path.is_file():
        return [input_path]

    return sorted(
        path
        for path in input_path.rglob("*.tif")
        if "PREVIEW" not in path.name.upper()
    )


def process_one(
    tiff_path: Path,
    output_dir: Path,
    mode: str,
    display_channel: int,
    overwrite: bool,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = tiff_path.stem

    mask_path = (
        output_dir
        / f"{stem}_tissue_mask.tif"
    )

    json_path = (
        output_dir
        / f"{stem}_tissue_roi.json"
    )

    if (
        mask_path.exists()
        and not overwrite
    ):
        print(
            f"[SKIP] {mask_path} already exists"
        )
        return

    stack = load_cyx(
        tiff_path
    )

    yx_shape = stack.shape[-2:]

    if mode == "full":

        mask = np.ones(
            yx_shape,
            dtype=bool,
        )

        vertices = None

    else:

        display = make_display(
            stack,
            display_channel,
        )

        vertices = select_polygon(
            display,
            tiff_path.name,
        )

        if vertices is None:

            print(
                f"[CANCELLED] {tiff_path.name}"
            )

            return

        mask = polygon_to_mask(
            vertices,
            yx_shape,
        )

    tifffile.imwrite(
        mask_path,
        mask.astype(np.uint8),
        compression="zlib",
    )

    metadata = {

        "source_tiff":
            str(tiff_path),

        "mode":
            mode,

        "shape_yx":
            list(yx_shape),

        "display_channel_1_based":
            display_channel,

        "vertices_xy":
            vertices,

        "mask_pixels":
            int(mask.sum()),
    }

    json_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print(
        f"[OK] Tissue mask: {mask_path}"
    )

    print(
        f"[OK] ROI metadata: {json_path}"
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input",
        type=Path,
        help=(
            "One TIFF or a directory "
            "containing converted TIFFs."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tissue_masks"),
    )

    parser.add_argument(
        "--mode",
        choices=(
            "interactive",
            "full",
        ),
        default="interactive",
    )

    parser.add_argument(
        "--display-channel",
        type=int,
        default=0,
        help=(
            "0 = max projection; "
            "otherwise 1-based channel number."
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
            f"No TIFF files found under "
            f"{args.input}"
        )

    print(
        f"Found {len(files)} TIFF(s)"
    )

    for i, path in enumerate(
        files,
        start=1,
    ):

        print(
            f"\n[{i}/{len(files)}] {path}"
        )

        process_one(
            path,
            output_dir=args.output_dir,
            mode=args.mode,
            display_channel=args.display_channel,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()

    #python scripts/tissue_roi.py converted_tiff --mode full, analyse absolutely everything in these appropriately scenes 
    #python scripts/tissue_roi.py converted_tiff, analyse grey matter 

    

