from pathlib import Path
import pandas as pd


# Configuration
AREA_THRESHOLD_LOW = 1800

# Optional upper bound.
# Set to None for now if you do not want to apply one yet.
AREA_THRESHOLD_HIGH = None

CELLPOSE_CSV_DIR = Path("./cellpose_masks/Slide_139")
OUTPUT_DIR = CELLPOSE_CSV_DIR / "area_filtered"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Find CSV files
csv_files = sorted(CELLPOSE_CSV_DIR.glob("*.csv"))
print(f"Found {len(csv_files)} CSV files")


# Process each file
for csv_file in csv_files:
    print(f"Processing: {csv_file.name}")
    df = pd.read_csv(csv_file)


    # Required-column checks
    if "Area_px" not in df.columns:
        print(
            "  [SKIP] No Area_px column"
        )
        continue

    if "Classification" not in df.columns:
        df["Classification"] = ""

    # Make sure Classification behaves as text
    df["Classification"] = (
        df["Classification"]
        .fillna("")
        .astype(str)
    )

    
    # Only automatically classify cells that have not
    # already been manually labelled.
    unclassified = (
        df["Classification"]
        .str.strip()
        .eq("")
    )

    # Low-area objects
    low_area_mask = (
        df["Area_px"] < AREA_THRESHOLD_LOW
    )

    low_area_unclassified = (
        low_area_mask
        & unclassified
    )

    df.loc[
        low_area_unclassified,
        "Classification"
    ] = "non-V0d-area-low"

    # --------------------------------------------------------
    # Optional high-area objects
    #
    # Useful later for very large merged Cellpose masks.
    # No upper threshold is applied unless
    # AREA_THRESHOLD_HIGH is given a value.
    # --------------------------------------------------------

    if AREA_THRESHOLD_HIGH is not None:

        # Recalculate unclassified because some cells may
        # already have been classified by the low-area rule.
        unclassified = (
            df["Classification"]
            .str.strip()
            .eq("")
        )

        high_area_mask = (
            df["Area_px"] > AREA_THRESHOLD_HIGH
        )

        high_area_unclassified = (
            high_area_mask
            & unclassified
        )

        df.loc[
            high_area_unclassified,
            "Classification"
        ] = "non-V0d-area-high"

    else:

        high_area_mask = pd.Series(
            False,
            index=df.index
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    n_total = len(df)

    n_low = int(
        low_area_mask.sum()
    )

    n_high = int(
        high_area_mask.sum()
    )

    n_remaining = int(
        (
            df["Classification"]
            .str.strip()
            .eq("")
        ).sum()
    )

    # --------------------------------------------------------
    # Save new file
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / csv_file.name
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"  Total objects:              {n_total}"
    )

    print(
        f"  Below {AREA_THRESHOLD_LOW}px:          "
        f"{n_low}"
    )

    if AREA_THRESHOLD_HIGH is not None:
        print(
            f"  Above {AREA_THRESHOLD_HIGH}px:          "
            f"{n_high}"
        )

    print(
        f"  Remaining for review:       "
        f"{n_remaining}"
    )

    print(
        f"  Saved: {output_path}"
    )


print(
    "\nFinished."
)