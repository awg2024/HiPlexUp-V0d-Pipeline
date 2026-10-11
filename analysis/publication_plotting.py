"""
publication_plotting.py

Plot within-slide and between-slide comparisons using R pipeline output tables.

Modes
-----
--within-slide ::: V0d vs non-V0d within one slide.
--compare-slides ::: Slide 139 vs Slide 141 for V0d/non-V0d counts and fluorescence profiles.

Usage
-----
python publication_plotting.py --slide 141 --within-slide
python publication_plotting.py --compare-slides
python publication_plotting.py --slide 141 --within-slide --compare-slides
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu, wilcoxon

# -------------------------------------------------------------------------
# Path Configurations
# -------------------------------------------------------------------------
base_path = Path(r"/Users/angusgray/Desktop/V0d-Histology/r_analysis/tables")

save_path_png = Path(r"/Users/angusgray/Desktop/V0d-Histology/r_analysis/figures/png")
save_path_svg = Path(r"/Users/angusgray/Desktop/V0d-Histology/r_analysis/figures/svg")

count_path = Path(r"/Users/angusgray/Desktop/V0d-Histology/r_analysis/tables/per_mouse_counts.csv")
fluorescence_path = Path(r"/Users/angusgray/Desktop/V0d-Histology/r_analysis/tables/per_mouse_flourescence.csv")

# -------------------------------------------------------------------------
# Headers Reference
# -------------------------------------------------------------------------
# HEADERS FOR FLUORESCENCE:
# Slide,Group,Animal,Sample_key,Classification,Marker,n_cells,Median_corrected_fluorescence

# HEADERS FOR COUNTS:
# Slide,Group,Animal,Sample_key,V0d,nonV0d,Total_classified,V0d_fraction

CHANNELS = ["DAPI", "EVX1", "PAX2", "DBX1", "VGAT"]
GROUPS = ["V0d-cell", "non-V0d-cell"]
SLIDES = [139, 141]


def load_r_fluorescence_data(slide=None):
    """Loads and standardizes per-mouse fluorescence summary data."""
    if not fluorescence_path.exists():
        raise FileNotFoundError(f"Fluorescence table not found at {fluorescence_path}")

    df = pd.read_csv(fluorescence_path)
    df["Classification"] = df["Classification"].fillna("").astype(str).str.strip()
    df["Marker"] = df["Marker"].fillna("").astype(str).str.strip()

    if slide is not None:
        df = df[df["Slide"] == int(slide)].copy()

    return df


def load_r_counts_data(slide=None):
    """Loads and standardizes per-mouse cell count summary data."""
    if not count_path.exists():
        raise FileNotFoundError(f"Counts table not found at {count_path}")

    df = pd.read_csv(count_path)

    if slide is not None:
        df = df[df["Slide"] == int(slide)].copy()

    return df


def save_figure(fig, name, png_dir, svg_dir):
    png_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(png_dir / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(svg_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def dispersion_plot(
    values_a,
    values_b,
    labels,
    ylabel,
    title,
    output_name,
    png_dir,
    svg_dir
):
    values_a = pd.Series(values_a).dropna().to_numpy()
    values_b = pd.Series(values_b).dropna().to_numpy()

    fig, ax = plt.subplots(figsize=(5, 6))

    data = [values_a, values_b]
    positions = [1, 2]

    # Boxplots
    ax.boxplot(
        data,
        positions=positions,
        widths=0.45,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.3},
        whiskerprops={"color": "black", "linewidth": 1.2},
        capprops={"color": "black", "linewidth": 1.2}
    )

    # Raw observations
    rng = np.random.default_rng(42)

    for x, values in zip(positions, data):
        ax.scatter(
            rng.normal(x, 0.06, len(values)),
            values,
            s=25,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.4,
            zorder=3
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)

    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    save_figure(fig, output_name, png_dir, svg_dir)


def violin_plot(
    values_a,
    values_b,
    labels,
    ylabel,
    title,
    output_name,
    png_dir,
    svg_dir
):
    values_a = pd.Series(values_a).dropna().to_numpy()
    values_b = pd.Series(values_b).dropna().to_numpy()

    fig, ax = plt.subplots(figsize=(5, 6))

    data = [values_a, values_b]
    positions = [1, 2]

    # Violin
    parts = ax.violinplot(
        data,
        positions=positions,
        widths=0.8,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    for body in parts["bodies"]:
        body.set_alpha(0.35)
        body.set_edgecolor("black")
        body.set_linewidth(1.2)

    # Median + IQR
    ax.boxplot(
        data,
        positions=positions,
        widths=0.16,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.2},
        whiskerprops={"color": "black", "linewidth": 1},
        capprops={"color": "black", "linewidth": 1}
    )

    # Individual points
    rng = np.random.default_rng(42)

    for x, values in zip(positions, data):
        ax.scatter(
            rng.normal(x, 0.05, len(values)),
            values,
            s=18,
            alpha=0.55,
            edgecolors="none"
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)

    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    save_figure(fig, output_name, png_dir, svg_dir)


def mann_whitney_result(a, b, comparison, channel=None, classification=None):
    a = pd.Series(a).dropna().to_numpy()
    b = pd.Series(b).dropna().to_numpy()

    if len(a) == 0 or len(b) == 0:
        return None

    statistic, p = mannwhitneyu(a, b, alternative="two-sided")

    return {
        "comparison": comparison,
        "channel": channel,
        "classification": classification,
        "n_group_1": len(a),
        "n_group_2": len(b),
        "median_group_1": np.median(a),
        "median_group_2": np.median(b),
        "statistic": statistic,
        "p_value": p
    }


def wilcoxon_result(a, b, comparison, channel=None, classification=None):
    a = pd.Series(a).to_numpy()
    b = pd.Series(b).to_numpy()

    valid = ~(pd.isna(a) | pd.isna(b))

    a = a[valid]
    b = b[valid]

    if len(a) < 2:
        return None

    statistic, p = wilcoxon(a, b, alternative="two-sided")

    return {
        "comparison": comparison,
        "channel": channel,
        "classification": classification,
        "n_group_1": len(a),
        "n_group_2": len(b),
        "median_group_1": np.median(a),
        "median_group_2": np.median(b),
        "statistic": statistic,
        "p_value": p
    }


# -------------------------------------------------------------------------
# Within-slide Analysis
# -------------------------------------------------------------------------
def run_within_slide(slide):
    print(f"\n=== Running Within-Slide Analysis: Slide {slide} ===")

    df_fluor = load_r_fluorescence_data(slide=slide)
    df_counts = load_r_counts_data(slide=slide)

    stats = []

    # 1. Fluorescence comparisons (V0d vs non-V0d per mouse)
    for channel in CHANNELS:
        channel_data = df_fluor[df_fluor["Marker"] == channel]

        pivoted = channel_data.pivot(
            index="Animal",
            columns="Classification",
            values="Median_corrected_fluorescence"
        )

        for group in GROUPS:
            if group not in pivoted.columns:
                pivoted[group] = np.nan

        paired = pivoted.dropna(subset=["V0d-cell", "non-V0d-cell"])
        v0d_mouse = paired["V0d-cell"]
        non_v0d_mouse = paired["non-V0d-cell"]

        violin_plot(
            v0d_mouse,
            non_v0d_mouse,
            ["V0d", "non-V0d"],
            f"{channel} mean fluorescence\n(background corrected)",
            f"{channel} fluorescence — Slide {slide}",
            f"Slide_{slide}_{channel}_V0d_vs_nonV0d",
            save_path_png,
            save_path_svg
        )

        result = wilcoxon_result(
            v0d_mouse,
            non_v0d_mouse,
            comparison="V0d vs non-V0d",
            channel=channel
        )

        if result:
            stats.append(result)

    # 2. Count comparisons (V0d vs non-V0d counts per mouse)
    dispersion_plot(
        df_counts["V0d"],
        df_counts["nonV0d"],
        ["V0d", "non-V0d"],
        "Cell count per mouse",
        f"Cell counts — Slide {slide}",
        f"Slide_{slide}_V0d_vs_nonV0d_counts",
        save_path_png,
        save_path_svg
    )

    if len(df_counts) >= 2:
        statistic, p = wilcoxon(df_counts["V0d"], df_counts["nonV0d"])
        stats.append({
            "comparison": "V0d vs non-V0d counts",
            "channel": None,
            "classification": None,
            "n_group_1": len(df_counts),
            "n_group_2": len(df_counts),
            "median_group_1": df_counts["V0d"].median(),
            "median_group_2": df_counts["nonV0d"].median(),
            "statistic": statistic,
            "p_value": p
        })

    print(f"\n=== Within-slide statistics: Slide {slide} ===")
    for result in stats:
        print(f"\nComparison: {result['comparison']}")
        if result.get("channel") is not None:
            print(f"Channel: {result['channel']}")
        print(f"n group 1: {result['n_group_1']}")
        print(f"n group 2: {result['n_group_2']}")
        print(f"Median group 1: {result['median_group_1']:.3f}")
        print(f"Median group 2: {result['median_group_2']:.3f}")
        print(f"Statistic: {result['statistic']:.3f}")
        print(f"p-value: {result['p_value']:.6g}")


# -------------------------------------------------------------------------
# Between-slide Analysis
# -------------------------------------------------------------------------
def run_slide_comparison():
    print("\n=== Running Slide 139 vs Slide 141 Comparison ===")

    df_fluor = load_r_fluorescence_data()
    df_counts = load_r_counts_data()

    stats = []

    # 1. Fluorescence: Compare Slide 139 vs 141 per Marker & Classification
    for classification in GROUPS:
        label = "V0d" if classification == "V0d-cell" else "non-V0d"
        subset = df_fluor[df_fluor["Classification"] == classification]

        for channel in CHANNELS:
            channel_subset = subset[subset["Marker"] == channel]

            slide_139_mouse = channel_subset.loc[
                channel_subset["Slide"] == 139, "Median_corrected_fluorescence"
            ]
            slide_141_mouse = channel_subset.loc[
                channel_subset["Slide"] == 141, "Median_corrected_fluorescence"
            ]

            result = mann_whitney_result(
                slide_139_mouse,
                slide_141_mouse,
                comparison="Slide 139 vs Slide 141",
                channel=channel,
                classification=classification
            )

            violin_plot(
                slide_139_mouse,
                slide_141_mouse,
                ["Slide 139", "Slide 141"],
                f"{channel} mean fluorescence\n(background corrected)",
                f"{label} {channel} fluorescence",
                f"{label}_{channel}_Slide139_vs_Slide141",
                save_path_png,
                save_path_svg
            )

            if result:
                stats.append(result)

    # 2. Counts: Compare Slide 139 vs 141 for V0d and non-V0d
    count_mappings = {"V0d-cell": "V0d", "non-V0d-cell": "nonV0d"}

    for classification, count_col in count_mappings.items():
        label = "V0d" if classification == "V0d-cell" else "non-V0d"

        slide_139_counts = df_counts.loc[df_counts["Slide"] == 139, count_col]
        slide_141_counts = df_counts.loc[df_counts["Slide"] == 141, count_col]

        dispersion_plot(
            slide_139_counts,
            slide_141_counts,
            ["Slide 139", "Slide 141"],
            f"{label} cell count per mouse",
            f"{label} cell counts",
            f"{label}_counts_Slide139_vs_Slide141",
            save_path_png,
            save_path_svg
        )

        result = mann_whitney_result(
            slide_139_counts,
            slide_141_counts,
            comparison="Slide 139 vs Slide 141 counts",
            classification=classification
        )

        if result:
            stats.append(result)

    print("\nBetween-slide statistics: Slide 139 vs Slide 141")
    for result in stats:
        print(f"\nComparison: {result['comparison']}")
        if result.get("classification") is not None:
            print(f"Classification: {result['classification']}")
        if result.get("channel") is not None:
            print(f"Channel: {result['channel']}")
        print(f"n Slide 139: {result['n_group_1']}")
        print(f"n Slide 141: {result['n_group_2']}")
        print(f"Median Slide 139: {result['median_group_1']:.3f}")
        print(f"Median Slide 141: {result['median_group_2']:.3f}")
        print(f"Statistic: {result['statistic']:.3f}")
        print(f"p-value: {result['p_value']:.6g}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--slide",
        type=int,
        choices=[139, 141]
    )

    parser.add_argument(
        "--within-slide",
        action="store_true",
        help="Generate V0d vs non-V0d plots for one slide"
    )

    parser.add_argument(
        "--compare-slides",
        action="store_true",
        help="Generate Slide 139 vs Slide 141 comparisons"
    )

    args = parser.parse_args()

    if not args.within_slide and not args.compare_slides:
        raise SystemExit("Choose --within-slide and/or --compare-slides")

    if args.within_slide:
        if args.slide is None:
            raise SystemExit("--within-slide requires --slide 139 or --slide 141")

        run_within_slide(args.slide)

    if args.compare_slides:
        run_slide_comparison()

    print("\nFinished.")


if __name__ == "__main__":
    main()