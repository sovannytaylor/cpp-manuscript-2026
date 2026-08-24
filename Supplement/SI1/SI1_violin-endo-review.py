"""Create linear/log violin and dot plots of concentration by uptake status.

Non-TAT observations are black and TAT-containing observations are red.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import seaborn as sns


# -----------------------------------------------------------------------------
# USER SETTINGS
# -----------------------------------------------------------------------------
CSV_PATH = Path(r"C:\Users\u244278\OneDrive - Baylor College of Medicine\Documents\python_projects\VIS_081726_LDLR_SI-figs\cpp_concentration_endocytosis_midpoints_with_tat.csv")

VIOLIN_LINEAR_OUTPUT = "endocytosis_status_midpoint_violin_linear_tat.svg"
VIOLIN_LOG_OUTPUT = "endocytosis_status_midpoint_violin_log_tat.svg"
DOT_LINEAR_OUTPUT = "endocytosis_status_midpoint_dot_linear_tat.svg"
DOT_LOG_OUTPUT = "endocytosis_status_midpoint_dot_log_tat.svg"

FIGURE_SIZE = (5.8, 5.0)
COLORS = {"Endocytosis": "#D95F59", "Not endocytosis": "#A7A9AC"}
ORDER = ["Endocytosis", "Not endocytosis"]

NON_TAT_POINT_COLOR = "#111111"
TAT_POINT_COLOR = "#D62728"
POINT_SIZE = 8
POINT_EDGE_COLOR = "#111111"
POINT_EDGE_WIDTH = 0.6
POINT_ALPHA = 0.85
JITTER = 0.18
RANDOM_SEED = 42

REFERENCE_CONCENTRATIONS = [2.0, 0.125]
REFERENCE_LINE_COLOR = "#404040"
REFERENCE_LINE_WIDTH = 1.2
REFERENCE_LINE_STYLE = "--"


def add_points(ax: plt.Axes, plot_data: pd.DataFrame) -> None:
    """Draw non-TAT observations in black and TAT observations in red."""
    non_tat = plot_data.loc[plot_data["tat_peptide"] == "No"]
    tat = plot_data.loc[plot_data["tat_peptide"] == "Yes"]

    for subset, point_color, zorder in (
        (non_tat, NON_TAT_POINT_COLOR, 3),
        (tat, TAT_POINT_COLOR, 4),
    ):
        if subset.empty:
            continue
        sns.stripplot(
            data=subset,
            x="endocytosis",
            y="concentration_uM",
            order=ORDER,
            color=point_color,
            size=POINT_SIZE,
            alpha=POINT_ALPHA,
            jitter=JITTER,
            edgecolor=POINT_EDGE_COLOR,
            linewidth=POINT_EDGE_WIDTH,
            ax=ax,
            zorder=zorder,
        )


def add_tat_legend(ax: plt.Axes) -> None:
    """Add a color legend outside the axes so it cannot cover observations."""
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none", label="Other CPP",
            markerfacecolor=NON_TAT_POINT_COLOR, markeredgecolor=POINT_EDGE_COLOR,
            markeredgewidth=POINT_EDGE_WIDTH, markersize=POINT_SIZE,
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none", label="TAT-containing",
            markerfacecolor=TAT_POINT_COLOR, markeredgecolor=POINT_EDGE_COLOR,
            markeredgewidth=POINT_EDGE_WIDTH, markersize=POINT_SIZE,
        ),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        fontsize=8,
    )


def format_axis(ax: plt.Axes, plot_data: pd.DataFrame, *, log_scale: bool) -> None:
    """Apply shared labels, scaling, counts, reference lines, and styling."""
    counts = plot_data.groupby("endocytosis").size()
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(
        [f"{status}\n(n = {counts.get(status, 0)})" for status in ORDER]
    )
    ax.set_xlabel("")
    ax.set_ylabel("Concentration (µM)")

    if log_scale:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MultipleLocator(2))

    for concentration in REFERENCE_CONCENTRATIONS:
        ax.axhline(
            y=concentration,
            color=REFERENCE_LINE_COLOR,
            linewidth=REFERENCE_LINE_WIDTH,
            linestyle=REFERENCE_LINE_STYLE,
            zorder=2,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.grid(axis="x", visible=False)
    add_tat_legend(ax)


def make_violin_plot(
    plot_data: pd.DataFrame, output_path: Path, *, log_scale: bool
) -> None:
    """Create and save a violin plot with emphasized TAT observations."""
    np.random.seed(RANDOM_SEED)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    sns.violinplot(
        data=plot_data,
        x="endocytosis",
        y="concentration_uM",
        hue="endocytosis",
        order=ORDER,
        hue_order=ORDER,
        palette=COLORS,
        legend=False,
        inner="quartile",
        cut=0,
        linewidth=1.2,
        density_norm="width",
        ax=ax,
    )
    add_points(ax, plot_data)
    format_axis(ax, plot_data, log_scale=log_scale)
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def make_dot_plot(
    plot_data: pd.DataFrame, output_path: Path, *, log_scale: bool
) -> None:
    """Create and save a dot plot with emphasized TAT observations."""
    np.random.seed(RANDOM_SEED)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    add_points(ax, plot_data)
    format_axis(ax, plot_data, log_scale=log_scale)
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def load_and_validate_csv(csv_path: Path) -> pd.DataFrame:
    """Load and validate concentration, uptake-status, and TAT columns."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip()
    required = {"concentration_uM", "endocytosis", "tat_peptide"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")

    plot_data = data.copy()
    plot_data["concentration_uM"] = pd.to_numeric(
        plot_data["concentration_uM"], errors="coerce"
    )
    plot_data["endocytosis"] = plot_data["endocytosis"].astype(str).str.strip()
    plot_data["tat_peptide"] = (
        plot_data["tat_peptide"].astype(str).str.strip().str.title()
    )
    plot_data = plot_data.dropna(subset=["concentration_uM", "endocytosis"])

    unexpected_status = sorted(set(plot_data["endocytosis"]) - set(ORDER))
    if unexpected_status:
        raise ValueError(f"Unexpected endocytosis values: {unexpected_status}")
    unexpected_tat = sorted(set(plot_data["tat_peptide"]) - {"Yes", "No"})
    if unexpected_tat:
        raise ValueError(f"Unexpected tat_peptide values: {unexpected_tat}")
    if plot_data.empty:
        raise ValueError("No valid rows remained after parsing the CSV.")
    return plot_data


def main() -> None:
    """Load the CSV and generate four SVG plots."""
    csv_path = CSV_PATH.expanduser().resolve()
    plot_data = load_and_validate_csv(csv_path)
    log_data = plot_data.loc[plot_data["concentration_uM"] > 0].copy()
    omitted_count = len(plot_data) - len(log_data)
    if log_data.empty:
        raise ValueError("No positive concentrations are available for log plots.")

    sns.set_theme(style="ticks", context="paper", font_scale=1.15)
    outputs = {
        "linear violin": csv_path.parent / VIOLIN_LINEAR_OUTPUT,
        "log violin": csv_path.parent / VIOLIN_LOG_OUTPUT,
        "linear dot": csv_path.parent / DOT_LINEAR_OUTPUT,
        "log dot": csv_path.parent / DOT_LOG_OUTPUT,
    }
    make_violin_plot(plot_data, outputs["linear violin"], log_scale=False)
    make_violin_plot(log_data, outputs["log violin"], log_scale=True)
    make_dot_plot(plot_data, outputs["linear dot"], log_scale=False)
    make_dot_plot(log_data, outputs["log dot"], log_scale=True)

    print(f"Loaded {len(plot_data)} rows ({(plot_data['tat_peptide'] == 'Yes').sum()} TAT).")
    for label, path in outputs.items():
        print(f"Saved {label}: {path}")
    if omitted_count:
        print(f"Log plots omitted {omitted_count} non-positive row(s).")


if __name__ == "__main__":
    main()