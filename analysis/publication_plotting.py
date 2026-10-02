"""
publication_plotting.py

Plot within-slide and between-slide comparisons from measure_cells.py outputs.

Modes
-----
--within-slide::: V0d vs non-V0d within one slide.

--compare-slides
    Slide 139 vs Slide 141 for: V0d counts, non-v0d counts, v0d/non-v0d fluorescence profiles for all channels 

    

for isolated slide 141 analysis 

python HiPlexUp-V0d-Pipeline/analysis/publication_plotting.py \
    --slide 141 \
    --within-slide

    
for comparative analysis between slide 139 and slide 141 

python HiPlexUp-V0d-Pipeline/analysis/publication_plotting.py \
    --compare-slides

    
or you can do both

python HiPlexUp-V0d-Pipeline/analysis/publication_plotting.py \
    --slide 141 \
    --within-slide \
    --compare-slides

"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu, wilcoxon


CHANNELS = ["DAPI", "EVX1", "PAX2", "DBX1", "VGAT"]
GROUPS = ["V0d-cell", "non-V0d-cell"]
SLIDES = [139, 141]


def load_data(input_dir, slide=None):
    """
    load in csv data from measure cells 
    """
    
    csv_files = sorted(input_dir.glob("*.csv"))

    # Avoid accidentally loading generated statistics files
    csv_files = [p for p in csv_files if "statistics" not in p.name and "counts" not in p.name]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df["Source_file"] = csv_file.name
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    df["Classification"] = (
        df["Classification"].fillna("").astype(str).str.strip())

    df = df[df["Classification"].isin(GROUPS)].copy()

    if slide is not None:
        df["Slide"] = int(slide)
    return df


def get_sample_column(df):
    for column in ["Image_ID", "Sample", "sample"]:
        if column in df.columns:
            return column

    raise ValueError("Could not find sample identifier column. Expected Image_ID or Sample.")


def jitter(n, centre, width=0.08):
    """
    confidence of point 
    """
    
    return np.random.normal(centre, width,n)


def save_figure(fig, name, png_dir, svg_dir):
    png_dir.mkdir(parents=True,exist_ok=True)

    svg_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(png_dir / f"{name}.png", dpi=600, bbox_inches="tight")

    fig.savefig(svg_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def dispersion_plot(values_a, values_b, labels, ylabel, title, output_name, png_dir, svg_dir):
    """
    produce dispersion plots, will be called iteratively depending on mode. 
    """
    
    
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)

    fig, ax = plt.subplots(
        figsize=(5, 6)
    )

    rng = np.random.default_rng(42)

    ax.scatter(
        rng.normal(0, 0.07, len(values_a)),
        values_a,
        alpha=0.6,
        s=25
    )

    ax.scatter(
        rng.normal(1, 0.07, len(values_b)),
        values_b,alpha=0.6,s=25)

    for x, values in enumerate([values_a, values_b]):
        if len(values) == 0:
            continue

        median = np.median(values)
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)

        ax.plot([x - 0.18, x + 0.18],[median, median],linewidth=2)

        ax.vlines(x,q1,q3,linewidth=2)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)

    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_figure(fig,output_name,png_dir,svg_dir)


def mann_whitney_result(a, b, comparison, channel=None, classification=None):
    """
    non-parametric statistical test for classification and channels 
    """

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
        "p_value": p}


# Within-slide plots
def run_within_slide(slide):
    """
    instance to handle within isolated slide plotting will call all helper functions here, main function for this mode 
    """


    input_dir = Path(f"row-wise-{slide}")
    png_dir = input_dir / "raw_png"
    svg_dir = input_dir / "raw_svg"

    png_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)


    df = load_data(input_dir,slide=slide)

    sample_col = get_sample_column(df)

    stats = []

    for channel in CHANNELS:
        value_col = (f"{channel}_mean_corrected")

        v0d = df.loc[df["Classification"] == "V0d-cell", value_col].dropna()

        non_v0d = df.loc[df["Classification"] == "non-V0d-cell", value_col].dropna()

        dispersion_plot(v0d, non_v0d, ["V0d", "non-V0d"],
        f"{channel} mean fluorescence\n(background corrected)",
        f"{channel} fluorescence — Slide {slide}",
        f"{channel}_V0d_vs_nonV0d",
        png_dir,
        svg_dir)


        result = mann_whitney_result(v0d, non_v0d, comparison="V0d vs non-V0d", channel=channel)

        if result:
            stats.append(result)

    counts = (df.groupby([sample_col, "Classification"]).size().unstack(fill_value=0))

    for group in GROUPS:
        if group not in counts.columns:
            counts[group] = 0

    dispersion_plot(
        counts["V0d-cell"],
        counts["non-V0d-cell"],
        ["V0d", "non-V0d"],
        "Cell count per sample",
        f"Cell counts — Slide {slide}",
        "V0d_vs_nonV0d_counts",
        png_dir,
        svg_dir)

    if len(counts) >= 2:
        statistic, p = wilcoxon(
            counts["V0d-cell"],
            counts["non-V0d-cell"])

        stats.append({
            "comparison":"V0d vs non-V0d counts",
            "channel": None,
            "classification": None,
            "n_group_1": len(counts),
            "n_group_2": len(counts),
            "median_group_1": counts["V0d-cell"].median(),
            "median_group_2": counts["non-V0d-cell"].median(),
            "statistic": statistic,
            "p_value":p})
    print(
    f"\n=== Within-slide statistics: Slide {slide} ==="
)

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

   
# Between-slide comparisons
def load_both_slides():
    dfs = []

    for slide in SLIDES:
        df = load_data(
            Path(f"row-wise-{slide}"),
            slide=slide
        )

        dfs.append(df)

    return pd.concat(
        dfs,
        ignore_index=True
    )


def run_slide_comparison():
    df = load_both_slides()

    output_dir = Path(
        "slide-comparison"
    )

    png_dir = output_dir / "raw_png"
    svg_dir = output_dir / "raw_svg"
    stats_dir = output_dir / "statistics"

    stats_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    sample_col = get_sample_column(df)

    stats = []

    # --------------------------------------------------------
    # Fluorescence:
    # compare Slide 139 vs Slide 141 separately for each
    # biological classification.
    # --------------------------------------------------------

    for classification in GROUPS:

        label = ("V0d" if classification == "V0d-cell" else "non-V0d")
        subset = df[df["Classification"] == classification]

        for channel in CHANNELS:
            value_col = (
                f"{channel}_mean_corrected"
            )

            slide_139 = subset.loc[
                subset["Slide"] == 139,
                value_col
            ].dropna()

            slide_141 = subset.loc[
                subset["Slide"] == 141,
                value_col
            ].dropna()

            dispersion_plot(
                slide_139,
                slide_141,
                ["Slide 139", "Slide 141"],
                f"{channel} mean fluorescence\n(background corrected)",
                f"{label} {channel} fluorescence",
                f"{label}_{channel}_Slide139_vs_Slide141",
                png_dir,
                svg_dir
            )

            result = mann_whitney_result(
                slide_139,
                slide_141,
                comparison=
                    "Slide 139 vs Slide 141",
                channel=channel,
                classification=classification
            )

            if result:
                stats.append(result)

  
    # compare V0d counts between slides
    # and non-V0d counts between slides.
    counts = (
        df.groupby(
            ["Slide", sample_col, "Classification"]
        )
        .size()
        .reset_index(
            name="Count"
        )
    )

    for classification in GROUPS:

        label = (
            "V0d"
            if classification == "V0d-cell"
            else "non-V0d"
        )

        group_counts = counts[
            counts["Classification"] == classification
        ]

        slide_139 = group_counts.loc[
            group_counts["Slide"] == 139,
            "Count"
        ]

        slide_141 = group_counts.loc[
            group_counts["Slide"] == 141,
            "Count"
        ]

        dispersion_plot(
            slide_139,
            slide_141,
            ["Slide 139", "Slide 141"],
            f"{label} cell count per sample",
            f"{label} cell counts",
            f"{label}_counts_Slide139_vs_Slide141",
            png_dir,
            svg_dir
        )

        result = mann_whitney_result(
            slide_139,
            slide_141,
            comparison=
                "Slide 139 vs Slide 141 counts",
            classification=classification
        )

        if result:
            stats.append(result)

    print("Between-slide statistics: Slide 139 vs Slide 141")

    for result in stats:

        print(f"\nComparison: {result['comparison']}")

        if result.get("classification") is not None:
            print(f"Classification: {result['classification']}")

        if result.get("channel") is not None:
            print(f"Channel: {result['channel']}")

        print(f"n Slide 139: {result['n_group_1']}")
        print(f"n Slide 141: {result['n_group_2']}")
        print(f"Median Slide 139: "f"{result['median_group_1']:.3f}")
        print(f"Median Slide 141: {result['median_group_2']:.3f}")
        print(f"Statistic: {result['statistic']:.3f}")
        print(f"p-value: {result['p_value']:.6g}")

    print("Counts by sample:")
    print(counts)


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
        raise SystemExit(
            "Choose --within-slide and/or --compare-slides"
        )

    if args.within_slide:
        if args.slide is None:
            raise SystemExit(
                "--within-slide requires --slide 139 or --slide 141"
            )

        print(
            f"\nRunning within-slide analysis: "
            f"Slide {args.slide}"
        )

        run_within_slide(
            args.slide
        )

    if args.compare_slides:
        print(
            "\nRunning Slide 139 vs Slide 141 comparison"
        )

        run_slide_comparison()

    print(
        "\nFinished."
    )


if __name__ == "__main__":
    main()