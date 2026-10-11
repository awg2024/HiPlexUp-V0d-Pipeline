#
# get_czi_dimensions.py 
#


import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from aicspylibczi import CziFile

def get_dimensions(czi_path: str):
    file_path = Path(czi_path)
    if not file_path.exists():
        print(f"Error: File '{czi_path}' not found.")
        return

    czi = CziFile(file_path)

    # Extract XML metadata safely
    if callable(getattr(czi, "get_meta_xml", None)):
        xml_str = czi.get_meta_xml()
        root = ET.fromstring(xml_str)
    elif isinstance(czi.meta, ET.Element):
        root = czi.meta
    else:
        root = ET.fromstring(str(czi.meta))

    pixel_width_um = 0.0
    pixel_height_um = 0.0

    # Search XML for Distance elements
    for distance in root.findall(".//Distance"):
        dist_id = distance.get("Id")
        value_elem = distance.find("Value")
        if value_elem is not None and value_elem.text:
            val_in_meters = float(value_elem.text)
            val_in_um = val_in_meters * 1e6

            if dist_id == "X":
                pixel_width_um = val_in_um
            elif dist_id == "Y":
                pixel_height_um = val_in_um

    # Extract dimensions and size from CziFile
    dims = czi.dims    # String like 'CZYX' or 'SZYXC'
    size = czi.size    # Tuple like (7, 1, 14104, 10186)

    # Match dimension names to their sizes
    dim_dict = dict(zip(dims, size))
    width_px = dim_dict.get('X', None)
    height_px = dim_dict.get('Y', None)

    width_um = width_px * pixel_width_um if width_px else 0.0
    height_um = height_px * pixel_height_um if height_px else 0.0

    print("\n" + "=" * 45)
    print(f" FILE: {file_path.name}")
    print("=" * 45)
    print(f" Pixel Width  : {pixel_width_um:.4f} µm/pixel")
    print(f" Pixel Height : {pixel_height_um:.4f} µm/pixel")
    print("-" * 45)
    if width_px and height_px:
        print(f" Image Dimensions (Pixels) : {width_px} x {height_px} px")
        print(f" Physical Field of View    : {width_um:.2f} x {height_um:.2f} µm")
    print(f" Full Data Shape ({dims})   : {size}")
    print("=" * 45 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_czi_dimensions.py <path_to_czi_file>")
    else:
        get_dimensions(sys.argv[1])