"""Create grouped lipoprotein plots from existing summary CSVs.

This script does NOT rerun punctum-level or per-cell calculations. It reads:
    peptide_lipoprotein_cell_summary.csv
    peptide_lipoprotein_rep_summary.csv

Output layout:
    one panel each for HDL, LDL, and VLDL
    peptides grouped within each panel
    red shades = peptide identities
    dark-gray dots = individual cells
    black/white-edged dots = replicate means
    black horizontal line = overall cell mean
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

SUMMARY_FOLDER = Path("results/summary_calculations-peptide-lipo-coloc")
PLOT_FOLDER = SUMMARY_FOLDER / "plots_grouped_by_lipoprotein"

CELL_SUMMARY_CSV = SUMMARY_FOLDER / "peptide_lipoprotein_cell_summary.csv"
REP_SUMMARY_CSV = SUMMARY_FOLDER / "peptide_lipoprotein_rep_summary.csv"

# Remove GP30 here if you only want CROT, LL37, and GR30.
PEPTIDE_ORDER = ["CROT", "LL37", "GR30"]
LIPOPROTEIN_ORDER = ["HDL", "LDL", "VLDL"]

# Bright-to-dark peptide-specific reds. Change these hex values if desired.
PEPTIDE_COLORS = {
    "CROT": "#8F1D2C",
    "LL37": "#C83E4D",
    "GR30": "#F05A67",
    "GP30": "#FF8992",
}

CELL_DOT_COLOR = "#A9A9A9"
REPLICATE_DOT_COLOR = "#000000"
PEPTIDE_LABEL_COLOR = "#A51F2D"

SHOW_CELL_DOTS = True
SHOW_REPLICATE_DOTS = True
SHOW_SAMPLE_SIZE = True
SAVE_PNG = True
SAVE_SVG = True
DPI = 300
RANDOM_SEED = 42

# Plot definitions: (column, y-axis label, output filename, y limits)
PLOTS = [
    (
        "fraction_peptide_puncta_coloc_within_1px",
        "+CPP vesicles overlapping with lipoprotein\n(fraction/cell)",
        "grouped_fraction_peptide_puncta_coloc_within_1px",
        (0, 1),
    ),
    (
        "percent_peptide_puncta_coloc_within_1px",
        "+CPP vesicles overlapping with lipoprotein\n(%/cell)",
        "grouped_percent_peptide_puncta_coloc_within_1px",
        (0, 100),
    ),
    (
        "n_peptide_puncta_coloc_within_1px",
        "Colocalized +CPP vesicles/cell",
        "grouped_n_peptide_puncta_coloc_within_1px",
        None,
    ),
    (
        "n_peptide_puncta",
        "+CPP vesicles/cell",
        "grouped_n_peptide_puncta",
        None,
    ),
]


# =============================================================================
# PLOTTING
# =============================================================================

def save_figure(fig: plt.Figure, filename: str) -> None:
    PLOT_FOLDER.mkdir(parents=True, exist_ok=True)
    if SAVE_PNG:
        fig.savefig(PLOT_FOLDER / f"{filename}.png", dpi=DPI, bbox_inches="tight")
    if SAVE_SVG:
        fig.savefig(PLOT_FOLDER / f"{filename}.svg", bbox_inches="tight")


def grouped_violin_plot(
    cell_df: pd.DataFrame,
    rep_df: pd.DataFrame,
    y_col: str,
    ylabel: str,
    filename: str,
    ylim: tuple[float, float] | None,
) -> None:
    if y_col not in cell_df.columns or y_col not in rep_df.columns:
        print(f"Skipping {y_col}: column is missing from a summary CSV.")
        return

    peptides = [p for p in PEPTIDE_ORDER if p in set(cell_df["peptide"].dropna())]
    lipoproteins = [l for l in LIPOPROTEIN_ORDER if l in set(cell_df["lipoprotein"].dropna())]

    if not peptides or not lipoproteins:
        raise ValueError("No configured peptides/lipoproteins were found in the cell summary CSV.")

    fig, axes = plt.subplots(
        1,
        len(lipoproteins),
        figsize=(3.25 * len(lipoproteins), 5.0),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(RANDOM_SEED)

    for panel_index, (ax, lipoprotein) in enumerate(zip(axes, lipoproteins)):
        for peptide_index, peptide in enumerate(peptides):
            cell_values = cell_df.loc[
                (cell_df["lipoprotein"] == lipoprotein)
                & (cell_df["peptide"] == peptide),
                y_col,
            ].dropna()

            rep_values = rep_df.loc[
                (rep_df["lipoprotein"] == lipoprotein)
                & (rep_df["peptide"] == peptide),
                y_col,
            ].dropna()

            if len(cell_values) >= 3 and cell_values.nunique() > 1:
                violin = ax.violinplot(
                    cell_values.to_numpy(),
                    positions=[peptide_index],
                    widths=0.82,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                    bw_method="scott",
                )
                for body in violin["bodies"]:
                    body.set_facecolor(PEPTIDE_COLORS.get(peptide, "#C83E4D"))
                    body.set_edgecolor("#222222")
                    body.set_linewidth(1.8)
                    body.set_alpha(0.82)

            if SHOW_CELL_DOTS and len(cell_values):
                jitter = rng.normal(0, 0.055, len(cell_values))
                ax.scatter(
                    peptide_index + jitter,
                    cell_values,
                    s=14,
                    color=CELL_DOT_COLOR,
                    alpha=0.38,
                    edgecolor="none",
                    zorder=3,
                )

            if SHOW_REPLICATE_DOTS and len(rep_values):
                jitter = rng.normal(0, 0.05, len(rep_values))
                ax.scatter(
                    peptide_index + jitter,
                    rep_values,
                    s=62,
                    color=REPLICATE_DOT_COLOR,
                    edgecolor="white",
                    linewidth=1.0,
                    zorder=6,
                )

            if len(cell_values):
                mean_value = float(cell_values.mean())
                ax.hlines(
                    mean_value,
                    peptide_index - 0.25,
                    peptide_index + 0.25,
                    color="black",
                    linewidth=2.5,
                    zorder=7,
                )

            if SHOW_SAMPLE_SIZE:
                ax.text(
                    peptide_index,
                    -0.225,
                    f"{len(cell_values)}",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=10,
                    color="#222222",
                    clip_on=False,
                )

        ax.set_title(lipoprotein, fontsize=18, fontweight="bold", pad=10)
        ax.set_xticks(range(len(peptides)))
        ax.set_xticklabels(
            peptides,
            fontsize=12,
            fontweight="bold",
            color=PEPTIDE_LABEL_COLOR,
            rotation=35,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_xlabel("")
        ax.tick_params(axis="both", width=1.5, length=5, labelsize=11)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle="--", linewidth=1.0, color="#D9D9D9", alpha=0.85)
        ax.xaxis.grid(False)

        if SHOW_SAMPLE_SIZE:
            ax.text(
                -0.16,
                -0.225,
                "n =",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                color="#222222",
                clip_on=False,
            )

        if ylim is not None:
            ax.set_ylim(*ylim)

        for spine in ax.spines.values():
            spine.set_linewidth(1.6)

        if panel_index > 0:
            ax.set_ylabel("")

    axes[0].set_ylabel(ylabel, fontsize=13, fontweight="bold")
    axes[-1].text(
        0.96, 0.96, f"N = {rep_df['rep'].nunique()}",
        transform=axes[-1].transAxes, ha="right", va="top", fontsize=12,
    )
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.30, top=0.86, wspace=0.18)
    save_figure(fig, filename)
    plt.close(fig)


def main() -> None:
    if not CELL_SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Cell summary not found: {CELL_SUMMARY_CSV}")
    if not REP_SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Replicate summary not found: {REP_SUMMARY_CSV}")

    cell_df = pd.read_csv(CELL_SUMMARY_CSV)
    rep_df = pd.read_csv(REP_SUMMARY_CSV)

    for y_col, ylabel, filename, ylim in PLOTS:
        grouped_violin_plot(cell_df, rep_df, y_col, ylabel, filename, ylim)

    print(f"Finished. Plots saved to: {PLOT_FOLDER}")


if __name__ == "__main__":
    main()