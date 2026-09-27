from pathlib import Path
import pandas as pd

csv_path = Path(
    "./cellpose_masks/Slide_139/area_filtered/"
    "139-129-122-134-30JUL26-RESCAN-Split_Scenes_(Write_files)-"
    "01-Scene-10-ScanRegion9_ALLCHANNELS_16bit_FIJI_cellpose_cells.csv"
)

V0D_IDS = {
    222, 285, 249, 254, 238, 249, 245, 231, 258, 262, 210, 128, 92, 118, 314, 301, 204, 
    224, 186, 264, 113, 472, 468, 466, 371, 376, 362, 430, 336
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