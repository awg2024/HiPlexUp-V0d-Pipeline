#
# helper function during the manaual classification of V0d/non-V0d cells. 
#
#!/usr/bin/env python3


"""
prepare_labels.py

Join expert biological classifications to the quantitative
cell-level measurements produced by measure_cells.py.

This script DOES NOT decide whether a cell is V0d.

It simply ensures that expert labels are safely and reproducibly
attached to the correct Cellpose cell IDs.

Expected measurement table
--------------------------

At minimum:

    Image_ID
    Cell_ID

Usually also:

    Animal_ID
    X
    Y
    Area_um2
    Circularity
    VGAT_corrected
    PAX2_corrected
    EVX1_corrected
    DBX1_corrected

Expected expert label table
---------------------------

At minimum:

    Image_ID
    Cell_ID
    Classification

Optional:

    Notes

Canonical classifications
-------------------------

    V0d
    non-V0d
    borderline
    exclude

Common variants such as "v0d", "non_v0d", "uncertain" and
"artefact" are automatically converted to the canonical labels.

Examples
--------

Create an empty expert-label template from all detected cells:

    python scripts/prepare_labels.py \
        measurements/all_cells.csv \
        --create-template labels/expert_labels.csv

Merge completed expert labels:

    python scripts/prepare_labels.py \
        measurements/all_cells.csv \
        --labels labels/expert_labels.csv \
        --output measurements/labelled_cells.csv

Require EVERY cell to have been reviewed:

    python scripts/prepare_labels.py \
        measurements/all_cells.csv \
        --labels labels/expert_labels.csv \
        --output measurements/labelled_cells.csv \
        --require-complete

Also write XLSX:

    python scripts/prepare_labels.py \
        measurements/all_cells.csv \
        --labels labels/expert_labels.csv \
        --output measurements/labelled_cells.csv \
        --write-xlsx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Classification definitions
# ---------------------------------------------------------------------------

CANONICAL_LABELS = {
    "V0d",
    "non-V0d",
    "borderline",
    "exclude",
}


LABEL_ALIASES = {
    # V0d
    "v0d":
        "V0d",

    "v0 d":
        "V0d",

    # non-V0d
    "non-v0d":
        "non-V0d",

    "non v0d":
        "non-V0d",

    "non_v0d":
        "non-V0d",

    "nonv0d":
        "non-V0d",

    "not v0d":
        "non-V0d",

    "not-v0d":
        "non-V0d",

    # borderline / uncertain
    "borderline":
        "borderline",

    "uncertain":
        "borderline",

    "borderline/uncertain":
        "borderline",

    "borderline uncertain":
        "borderline",

    "ambiguous":
        "borderline",

    # exclude
    "exclude":
        "exclude",

    "excluded":
        "exclude",

    "artifact":
        "exclude",

    "artefact":
        "exclude",

    "debris":
        "exclude",
}


# ---------------------------------------------------------------------------
# File IO
# ---------------------------------------------------------------------------

def load_table(
    path: Path,
) -> pd.DataFrame:
    """
    Read CSV, TSV or Excel.
    """

    suffix = path.suffix.lower()

    if suffix == ".csv":

        return pd.read_csv(
            path
        )

    if suffix in (
        ".tsv",
        ".txt",
    ):

        return pd.read_csv(
            path,
            sep="\t",
        )

    if suffix in (
        ".xlsx",
        ".xls",
    ):

        return pd.read_excel(
            path
        )

    raise ValueError(
        f"Unsupported table type: {path}"
    )


def save_csv(
    dataframe: pd.DataFrame,
    path: Path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        path,
        index=False,
    )


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------

def clean_column_names(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    dataframe = dataframe.copy()

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    return dataframe


def clean_key_columns(
    dataframe: pd.DataFrame,
    key_columns,
) -> pd.DataFrame:
    """
    Standardise key columns without changing their semantic identity.

    IDs are stored as strings to avoid Excel/CSV type problems such as:

        1
        1.0
    """

    dataframe = dataframe.copy()

    for column in key_columns:

        if column not in dataframe.columns:

            raise ValueError(
                f"Required key column "
                f"'{column}' is missing."
            )

        # Convert IDs like 1.0 to 1 before stringification.
        values = dataframe[
            column
        ]

        def normalise_identifier(value):

            if pd.isna(value):
                return ""

            if isinstance(
                value,
                (
                    float,
                    np.floating,
                ),
            ):

                if float(value).is_integer():
                    return str(
                        int(value)
                    )

            return str(
                value
            ).strip()

        dataframe[
            column
        ] = values.map(
            normalise_identifier
        )

        if (
            dataframe[
                column
            ]
            == ""
        ).any():

            number_missing = int(
                (
                    dataframe[
                        column
                    ]
                    == ""
                ).sum()
            )

            raise ValueError(
                f"Column '{column}' contains "
                f"{number_missing} missing IDs."
            )

    return dataframe


def canonicalise_label(
    value,
):
    """
    Convert common classification spellings into canonical names.
    """

    if pd.isna(value):
        return ""

    raw = str(
        value
    ).strip()

    if raw == "":
        return ""

    lower = raw.lower()

    if lower in LABEL_ALIASES:

        return LABEL_ALIASES[
            lower
        ]

    # Permit exact canonical spellings.
    if raw in CANONICAL_LABELS:
        return raw

    return raw


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def check_duplicate_keys(
    dataframe: pd.DataFrame,
    key_columns,
    table_name: str,
):
    """
    Error if the same cell identifier occurs more than once.
    """

    duplicated = dataframe.duplicated(
        subset=key_columns,
        keep=False,
    )

    if duplicated.any():

        duplicate_rows = dataframe.loc[
            duplicated,
            key_columns,
        ].sort_values(
            key_columns
        )

        print(
            f"\nERROR: duplicate cell IDs "
            f"found in {table_name}:"
        )

        print(
            duplicate_rows.to_string(
                index=False
            )
        )

        raise ValueError(
            f"{table_name} contains "
            f"duplicate key combinations."
        )


def validate_labels(
    labels: pd.DataFrame,
    classification_column: str,
):
    """
    Find unexpected classification values.
    """

    if classification_column not in labels.columns:

        raise ValueError(
            f"Labels table does not contain "
            f"'{classification_column}'."
        )

    labels = labels.copy()

    labels[
        classification_column
    ] = labels[
        classification_column
    ].map(
        canonicalise_label
    )

    observed = set(
        labels.loc[
            labels[
                classification_column
            ]
            != "",
            classification_column,
        ]
    )

    invalid = sorted(
        observed
        - CANONICAL_LABELS
    )

    if invalid:

        raise ValueError(
            "Unknown classification "
            "value(s): "
            + ", ".join(invalid)
            + "\n\nAllowed values are:\n"
            + "\n".join(
                sorted(
                    CANONICAL_LABELS
                )
            )
        )

    return labels


# ---------------------------------------------------------------------------
# Template generation
# ---------------------------------------------------------------------------

def create_label_template(
    measurements: pd.DataFrame,
    key_columns,
    output_path: Path,
):
    """
    Create a spreadsheet/CSV that can be filled during expert review.

    Some useful quantitative columns are carried into the template
    if they are already available.
    """

    helpful_columns = [
        "Animal_ID",
        "X",
        "Y",
        "Area_um2",
        "Circularity",
        "VGAT_corrected",
        "PAX2_corrected",
        "EVX1_corrected",
        "DBX1_corrected",
    ]

    columns = list(
        key_columns
    )

    for column in helpful_columns:

        if (
            column in measurements.columns
            and column not in columns
        ):

            columns.append(
                column
            )

    template = measurements[
        columns
    ].copy()

    template[
        "Classification"
    ] = ""

    template[
        "Notes"
    ] = ""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = output_path.suffix.lower()

    if suffix == ".csv":

        template.to_csv(
            output_path,
            index=False,
        )

    elif suffix == ".xlsx":

        template.to_excel(
            output_path,
            index=False,
        )

    else:

        raise ValueError(
            "Label template must end "
            "in .csv or .xlsx"
        )

    print(
        f"Created expert-label template:\n"
        f"  {output_path}"
    )

    print(
        f"\nRows awaiting classification: "
        f"{len(template)}"
    )


# ---------------------------------------------------------------------------
# Merge labels
# ---------------------------------------------------------------------------

def merge_labels(
    measurements: pd.DataFrame,
    labels: pd.DataFrame,
    key_columns,
    classification_column: str,
):
    """
    Merge expert classifications onto quantitative cell data.
    """

    measurements = measurements.copy()
    labels = labels.copy()

    # We want the labels table to be the authoritative source.
    if classification_column in measurements.columns:

        existing = measurements[
            classification_column
        ].fillna(
            ""
        ).astype(
            str
        ).str.strip()

        if (
            existing != ""
        ).any():

            print(
                "\nWARNING: measurement table already "
                "contains classifications."
            )

            print(
                "They will be replaced by the "
                "expert label table."
            )

        measurements = measurements.drop(
            columns=[
                classification_column
            ]
        )

    # Preserve optional notes.
    label_columns = (
        list(key_columns)
        + [
            classification_column
        ]
    )

    if "Notes" in labels.columns:
        label_columns.append(
            "Notes"
        )

    # ---------------------------------------------------------------
    # Find labels that refer to cells that do not exist.
    # ---------------------------------------------------------------

    measurement_keys = measurements[
        key_columns
    ].copy()

    label_key_check = labels.merge(
        measurement_keys,
        on=key_columns,
        how="left",
        indicator=True,
    )

    unmatched_labels = label_key_check[
        label_key_check[
            "_merge"
        ]
        == "left_only"
    ].copy()

    # ---------------------------------------------------------------
    # Actual merge.
    # ---------------------------------------------------------------

    merged = measurements.merge(
        labels[
            label_columns
        ],
        on=key_columns,
        how="left",
        validate="one_to_one",
    )

    merged[
        classification_column
    ] = merged[
        classification_column
    ].fillna(
        ""
    )

    if "Notes" in merged.columns:

        merged[
            "Notes"
        ] = merged[
            "Notes"
        ].fillna(
            ""
        )

    unlabeled = merged[
        merged[
            classification_column
        ]
        == ""
    ].copy()

    return (
        merged,
        unlabeled,
        unmatched_labels,
    )


# ---------------------------------------------------------------------------
# Summary / QC
# ---------------------------------------------------------------------------

def label_summary(
    dataframe: pd.DataFrame,
    classification_column: str,
):
    """
    Generate counts for each biological label.
    """

    labels = dataframe[
        classification_column
    ].replace(
        "",
        "UNLABELED",
    )

    counts = labels.value_counts(
        dropna=False
    )

    return {
        str(label):
            int(count)
        for label, count in counts.items()
    }


def print_summary(
    measurements,
    merged,
    unlabeled,
    unmatched_labels,
    classification_column,
):
    """
    Print a human-readable merge summary.
    """

    print(
        "\n"
        + "=" * 65
    )

    print(
        "LABEL PREPARATION SUMMARY"
    )

    print(
        "=" * 65
    )

    print(
        f"Total detected cells: "
        f"{len(measurements)}"
    )

    print(
        f"Cells with expert label: "
        f"{len(merged) - len(unlabeled)}"
    )

    print(
        f"Cells still unlabeled: "
        f"{len(unlabeled)}"
    )

    print(
        f"Labels with no matching cell: "
        f"{len(unmatched_labels)}"
    )

    print(
        "\nClassification counts:"
    )

    summary = label_summary(
        merged,
        classification_column,
    )

    for label, count in summary.items():

        print(
            f"  {label:<15} "
            f"{count:>6}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Safely attach expert V0d "
            "classifications to Cellpose "
            "cell measurements."
        )
    )

    parser.add_argument(
        "measurements",
        type=Path,
        help=(
            "Cell-level measurements CSV/XLSX "
            "produced by measure_cells.py."
        ),
    )

    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help=(
            "Completed expert labels "
            "CSV/XLSX."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "measurements/"
            "labelled_cells.csv"
        ),
    )

    parser.add_argument(
        "--create-template",
        type=Path,
        default=None,
        help=(
            "Create an empty expert-label "
            "template and exit."
        ),
    )

    parser.add_argument(
        "--key-columns",
        nargs="+",
        default=[
            "Image_ID",
            "Cell_ID",
        ],
        help=(
            "Columns uniquely identifying "
            "one cell."
        ),
    )

    parser.add_argument(
        "--classification-column",
        default="Classification",
    )

    parser.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "Fail if any detected cell "
            "has not been classified."
        ),
    )

    parser.add_argument(
        "--write-xlsx",
        action="store_true",
        help=(
            "Also save an XLSX version "
            "of the merged dataset."
        ),
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Load cell measurements.
    # ---------------------------------------------------------------

    if not args.measurements.exists():

        raise SystemExit(
            f"Measurements file does not exist: "
            f"{args.measurements}"
        )

    measurements = load_table(
        args.measurements
    )

    measurements = clean_column_names(
        measurements
    )

    measurements = clean_key_columns(
        measurements,
        args.key_columns,
    )

    check_duplicate_keys(
        measurements,
        args.key_columns,
        "measurement table",
    )

    # ---------------------------------------------------------------
    # Template mode.
    # ---------------------------------------------------------------

    if args.create_template is not None:

        create_label_template(
            measurements,
            args.key_columns,
            args.create_template,
        )

        return

    # ---------------------------------------------------------------
    # Label merging requires a label file.
    # ---------------------------------------------------------------

    if args.labels is None:

        raise SystemExit(
            "Supply --labels or use "
            "--create-template."
        )

    if not args.labels.exists():

        raise SystemExit(
            f"Labels file does not exist: "
            f"{args.labels}"
        )

    labels = load_table(
        args.labels
    )

    labels = clean_column_names(
        labels
    )

    labels = clean_key_columns(
        labels,
        args.key_columns,
    )

    labels = validate_labels(
        labels,
        args.classification_column,
    )

    check_duplicate_keys(
        labels,
        args.key_columns,
        "expert labels table",
    )

    # ---------------------------------------------------------------
    # Merge.
    # ---------------------------------------------------------------

    (
        merged,
        unlabeled,
        unmatched_labels,
    ) = merge_labels(
        measurements,
        labels,
        args.key_columns,
        args.classification_column,
    )

    print_summary(
        measurements,
        merged,
        unlabeled,
        unmatched_labels,
        args.classification_column,
    )

    # ---------------------------------------------------------------
    # Save QC files before potentially failing completeness check.
    # ---------------------------------------------------------------

    qc_dir = (
        args.output.parent
        / "label_qc"
    )

    qc_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if len(unlabeled) > 0:

        unlabeled_path = (
            qc_dir
            / "unlabeled_cells.csv"
        )

        unlabeled.to_csv(
            unlabeled_path,
            index=False,
        )

        print(
            f"\nUnlabeled cells written to:\n"
            f"  {unlabeled_path}"
        )

    if len(unmatched_labels) > 0:

        unmatched_path = (
            qc_dir
            / "labels_without_matching_cells.csv"
        )

        unmatched_labels.to_csv(
            unmatched_path,
            index=False,
        )

        print(
            "\nWARNING: Some expert labels "
            "do not match detected cells."
        )

        print(
            f"See:\n"
            f"  {unmatched_path}"
        )

    # ---------------------------------------------------------------
    # Generate JSON QC report.
    # ---------------------------------------------------------------

    report = {
        "Measurements_File":
            str(args.measurements),

        "Labels_File":
            str(args.labels),

        "Output_File":
            str(args.output),

        "Key_Columns":
            args.key_columns,

        "Total_Cells":
            int(len(measurements)),

        "Labelled_Cells":
            int(
                len(merged)
                - len(unlabeled)
            ),

        "Unlabelled_Cells":
            int(len(unlabeled)),

        "Unmatched_Expert_Labels":
            int(
                len(unmatched_labels)
            ),

        "Classification_Counts":
            label_summary(
                merged,
                args.classification_column,
            ),
    }

    report_path = (
        qc_dir
        / "label_merge_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        )
    )

    # ---------------------------------------------------------------
    # Safety checks.
    # ---------------------------------------------------------------

    if len(unmatched_labels) > 0:

        raise SystemExit(
            "\nSTOPPED: expert label table "
            "contains Cell IDs that do not "
            "exist in the measurements table.\n"
            "Resolve these before continuing."
        )

    if (
        args.require_complete
        and len(unlabeled) > 0
    ):

        raise SystemExit(
            "\nSTOPPED: --require-complete "
            "was specified, but some cells "
            "remain unlabeled."
        )

    # ---------------------------------------------------------------
    # Save final merged dataset.
    # ---------------------------------------------------------------

    save_csv(
        merged,
        args.output,
    )

    print(
        f"\nFinal labelled cell dataset:\n"
        f"  {args.output}"
    )

    if args.write_xlsx:

        xlsx_path = (
            args.output.with_suffix(
                ".xlsx"
            )
        )

        merged.to_excel(
            xlsx_path,
            index=False,
        )

        print(
            f"  {xlsx_path}"
        )

    print(
        f"\nQC report:\n"
        f"  {report_path}"
    )

    print(
        "\nLabel preparation complete."
    )


if __name__ == "__main__":
    main()