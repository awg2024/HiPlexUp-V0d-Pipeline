import os
import tifffile


# change as per 
tiff_path = r"/Users/angusgray/Desktop/V0d-Histology/cellpose_masks/masks/Slide_141/141-125-132-128-30JUL26-RESCAN-Split_Scenes_(Write_files)-02-Scene-06-ScanRegion5_ALLCHANNELS_16bit_FIJI_cellpose_QC_16bit_FIJI.tif"

print("\n=== FILE ===")

print("exists:", os.path.exists(tiff_path))

size_bytes = os.path.getsize(tiff_path)

print("size bytes:", size_bytes)
print(f"size GiB {round(size_bytes / 1024**3, 3)} expected to be about ~1.61GiB!")

print("\n=== TIFF STRUCTURE ===")

try:

    with tifffile.TiffFile(tiff_path) as tif:

        print("is_imagej:", tif.is_imagej)
        print("is_bigtiff:", tif.is_bigtiff)

        print(
            "number of visible pages:",
            len(tif.pages)
        )

        print(
            "number of series:",
            len(tif.series)
        )

        for i, series in enumerate(tif.series):

            print(
                f"series {i}: "
                f"shape={series.shape}, "
                f"dtype={series.dtype}, "
                f"axes={series.axes}"
            )

except Exception as e:

    print(
        "FAILED while reading TIFF structure:"
    )

    print(
        type(e).__name__,
        e
    )


print("\n=== FULL ARRAY READ ===")

try:

    x = tifffile.imread(
        tiff_path
    )

    print(
        "SUCCESS"
    )

    print(
        "shape:",
        x.shape
    )

    print(
        "dtype:",
        x.dtype
    )

except Exception as e:

    print(
        "FAILED while reading pixel data:"
    )

    print(
        type(e).__name__,
        e
    )
