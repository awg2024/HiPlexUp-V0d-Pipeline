#
# this script generates a row-wise v0d-cell dataset 
#

import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import czifile
import tifffile
from PIL import Image
import glob
import argparse


from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile

from cellpose import models

bg_fluorescence_csv = Path(r"/Users/angusgray/Desktop/V0d-Histology/background_fluorescence/141-125-132-128-30JUL26-RESCAN-Split_Scenes_(Write_files)-02-Scene-09-ScanRegion8_ALLCHANNELS_16bit_FIJI_background.csv")
bg_fluorescence = np.read_csv(bg_fluorescence_csv)


def fluorescence_normalisation(bg_)
    """
    Input the cell id flourescence of a specific channel and normalise it, taking away it's background staining 
    """



def main():

    parser.add_argument("target",nargs="?",default=("/Users/angusgray/Desktop/V0d-Histology/HiPlexUp-V0d-Pipeline/raw_czi")) # select czi file location 
    args = parser.parse_args()



if __name__ == "__main__":
    main()

