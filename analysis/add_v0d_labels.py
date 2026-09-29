from pathlib import Path
import pandas as pd

csv_path = Path(""
    "./cellpose_masks/Slide_139/area_filtered/"
    "139-129-122-134-30JUL26-RESCAN-Split_Scenes_(Write_files)-"
    "01-Scene-09-ScanRegion8_ALLCHANNELS_16bit_FIJI_cellpose_cells.csv"
)

V0D_IDS = {442, 502, 536, 550, 499, 523, 618, 634, 691, 710, 
           719, 659, 669, 744, 751, 767, 669, 646, 556, 539, 
           441, 403, 366, 441, 383, 687, 694, 338, 331, 289, 
           296, 242, 237, 253, 280, 230, 231, 168, 81, 127, 
           280, 155, 104, 76, 99, 41, 86, 110, 168
}

df = pd.read_csv(csv_path)

df["Classification"] = (
    df["Classification"]
    .fillna("")
    .astype(str)
    .str.strip()
)

v0d_cell_ids = {f"cell_{i:04d}" for i in V0D_IDS}

missing_ids = v0d_cell_ids - set(df["Cell_ID"].astype(str))
if missing_ids:
    print("Missing V0d IDs:", sorted(missing_ids))

v0d_mask = df["Cell_ID"].isin(v0d_cell_ids)
df.loc[v0d_mask, "Classification"] = "V0d-cell"

unclassified_mask = df["Classification"].eq("")
df.loc[unclassified_mask, "Classification"] = "non-V0d-cell"

df.to_csv(csv_path, index=False)

print("\nClassification counts:")
print(df["Classification"].value_counts())

n_unclassified = df["Classification"].eq("").sum()
print(f"\nUnclassified remaining: {n_unclassified}")

if n_unclassified != 0:
    raise RuntimeError("Some cells remain unclassified.")

print(f"\nUpdated: {csv_path}")