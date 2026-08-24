"""Create one concentration split-violin plot per peptide and parameter.

The script expects the accompanying
``cell_morphology_features_indexed_filtered.csv`` file by default. Each plot
contains one peptide and one parameter. Media types form the two violin halves.
Both SVG and PNG versions are saved.
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd
import seaborn as sns


# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Input/output locations. By default, place this script beside the CSV.
CSV_PATH = Path(__file__).with_name("cell_morphology_features_indexed_filtered.csv")
OUTPUT_DIR = Path(__file__).with_name("morphology_boxplots")

# Choose any entries from the lists printed when the script starts.
PEPTIDES_TO_PLOT = ["BMAP", "LL37","PR39"]
PARAMETERS_TO_PLOT = [
    "area_um2",
    "circularity",
]

# Media is used as the plot hue. This controls both inclusion and display order.
MEDIA_ORDER = ["NORM", "LPD"]

# Split-violin colors and opacity are configured independently by media type.
VIOLIN_COLORS = {
    "NORM": "#E3A6A6",
    "LPD": "#B41F1F",
}
VIOLIN_ALPHA = {
    "NORM": 0.35,
    "LPD": 0.90,
}

# Optional individual-point overlay colors.
DOT_COLORS = {
    "NORM": "#E3A6A6",
    "LPD": "#B41F1F",
}

# Optional filters. Use None to include every value in that column.
TIMEPOINTS_TO_INCLUDE = None
REPLICATES_TO_INCLUDE = None

# Concentrations appear in this order. Use None for ascending numeric order.
CONCENTRATION_ORDER = [0, 4, 8, 16, 32]
CONCENTRATION_UNIT = ""  # e.g. "µM"; leave empty to display "Concentration"

# Figure appearance.
FIGURE_SIZE = (8.0, 5.2)
# Split-violin appearance.
VIOLIN_EDGE_COLOR = "none"  # e.g. "#404040", or "none" for no outline
VIOLIN_LINE_WIDTH = 0.8
VIOLIN_WIDTH = 0.86
VIOLIN_BW_ADJUST = 0.8
VIOLIN_CUT = 0  # 0 prevents the density from extending beyond observed values
VIOLIN_INNER = None  # alternatives supported by seaborn include "quartile" or "box"

# Set this to True if you also want cell-level dots over the split violins.
SHOW_INDIVIDUAL_POINTS = False
DOT_EDGE_COLOR = "#4A4A4A"
DOT_SIZE = 10
DOT_ALPHA = 0.35
DOT_JITTER = 0.045
DOT_EDGE_WIDTH = 0.45

# Sample-size labels: each half-violin gets its own n value.
SHOW_N_ON_PLOT = True
PRINT_N_TO_CONSOLE = True
N_LABEL_FORMAT = "n={n:,}"
N_LABEL_FONT_SIZE = 14
N_LABEL_Y_POSITION = 0.018  # axes fraction: 0 = bottom, 1 = top
N_LABEL_FONT_WEIGHT = "normal"
N_LABEL_USE_MEDIA_COLOR = False
N_LABEL_BACKGROUND_ALPHA = 0.72

# If plots are too dense, set a maximum number of points per concentration.
# Sampling is reproducible and affects dots only; every row remains in the boxplot.
MAX_DOTS_PER_CONCENTRATION = None  # e.g. 500, or None for all dots
RANDOM_SEED = 42

SHOW_OUTLIERS_IN_BOXPLOT = False  # points are already shown by the dot overlay
SHOW_MEAN = True
Y_AXIS_STARTS_AT_ZERO = False

FONT_FAMILY = "Arial"  # Change to "Arial" if Arial is installed.
TITLE_FONT_SIZE = 16
SHOW_TITLE = False
AXIS_LABEL_FONT_SIZE = 20
TICK_FONT_SIZE = 24
AXES_BORDER_COLOR = "#000000"
AXES_BORDER_WIDTH = 1.5
LEGEND_TITLE_FONT_SIZE = 11
LEGEND_FONT_SIZE = 12
LEGEND_LOCATION = "upper right"
SHOW_LEGEND_FRAME = True

PNG_DPI = 600
SAVE_SVG = True
SAVE_PNG = True
TRANSPARENT_BACKGROUND = False


# Optional publication-friendly labels. Unlisted parameters use their CSV name.
PARAMETER_LABELS = {
    "area": "Cell area (pixels²)",
    "area_um2": "Cell area (µm²)",
    "perimeter": "Cell perimeter (pixels)",
    "perimeter_um": "Cell perimeter (µm)",
    "eccentricity": "Eccentricity",
    "solidity": "Solidity",
    "major_axis_length": "Major axis length (pixels)",
    "major_axis_length_um": "Major axis length (µm)",
    "minor_axis_length": "Minor axis length (pixels)",
    "circularity": "Circularity",
    "aspect_ratio": "Aspect ratio",
    "convexity": "Convexity",
    "fractal_dimension": "Fractal dimension",
    "pi_fov_intensity_mean": "PI FOV mean intensity",
    "pi_fov_intensity_median": "PI FOV median intensity",
    "pi_fov_intensity_sum": "PI FOV summed intensity",
    "n_cells_in_fov": "Cells per field of view",
}


def safe_filename(value: object) -> str:
    """Return a filesystem-safe version of a label."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def apply_optional_filter(
    data: pd.DataFrame, column: str, allowed_values: list | None
) -> pd.DataFrame:
    if allowed_values is None:
        return data
    if column not in data.columns:
        raise ValueError(f"Cannot filter on missing column: {column!r}")
    return data[data[column].isin(allowed_values)]


def validate_configuration(data: pd.DataFrame) -> None:
    required = {"peptide", "concentration", "media", *PARAMETERS_TO_PLOT}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"CSV is missing configured column(s): {missing}")

    unknown_peptides = sorted(set(PEPTIDES_TO_PLOT).difference(data["peptide"].dropna()))
    if unknown_peptides:
        raise ValueError(f"Configured peptide(s) not found in CSV: {unknown_peptides}")

    unknown_media = sorted(set(MEDIA_ORDER).difference(data["media"].dropna()))
    if unknown_media:
        raise ValueError(f"Configured media value(s) not found in CSV: {unknown_media}")
    missing_violin_colors = sorted(set(MEDIA_ORDER).difference(VIOLIN_COLORS))
    if missing_violin_colors:
        raise ValueError(f"VIOLIN_COLORS is missing: {missing_violin_colors}")
    missing_violin_alpha = sorted(set(MEDIA_ORDER).difference(VIOLIN_ALPHA))
    if missing_violin_alpha:
        raise ValueError(f"VIOLIN_ALPHA is missing: {missing_violin_alpha}")
    missing_dot_colors = sorted(set(MEDIA_ORDER).difference(DOT_COLORS))
    if missing_dot_colors:
        raise ValueError(f"DOT_COLORS is missing: {missing_dot_colors}")

    nonnumeric = [
        parameter
        for parameter in PARAMETERS_TO_PLOT
        if not pd.api.types.is_numeric_dtype(data[parameter])
    ]
    if nonnumeric:
        raise TypeError(f"Selected parameter(s) are not numeric: {nonnumeric}")


def choose_concentrations(data: pd.DataFrame) -> list:
    observed = list(pd.unique(data["concentration"].dropna()))
    if CONCENTRATION_ORDER is None:
        return sorted(observed)
    return [value for value in CONCENTRATION_ORDER if value in observed]


def sample_dots(data: pd.DataFrame) -> pd.DataFrame:
    if MAX_DOTS_PER_CONCENTRATION is None:
        return data
    return (
        data.groupby(["concentration", "media"], group_keys=False, observed=True)
        .apply(
            lambda group: group.sample(
                n=min(len(group), MAX_DOTS_PER_CONCENTRATION),
                random_state=RANDOM_SEED,
            ),
            include_groups=False,
        )
        .reset_index()
    )


def make_plot(data: pd.DataFrame, peptide: str, parameter: str) -> None:
    plot_data = data.loc[
        data["peptide"].eq(peptide), ["concentration", "media", parameter]
    ].dropna()
    if plot_data.empty:
        print(f"Skipping {peptide} / {parameter}: no rows after filtering.")
        return

    concentration_order = choose_concentrations(plot_data)
    plot_data = plot_data[plot_data["concentration"].isin(concentration_order)]
    dot_data = sample_dots(plot_data)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    sns.violinplot(
        data=plot_data,
        x="concentration",
        y=parameter,
        hue="media",
        order=concentration_order,
        hue_order=MEDIA_ORDER,
        palette=VIOLIN_COLORS,
        split=True,
        inner=VIOLIN_INNER,
        cut=VIOLIN_CUT,
        bw_adjust=VIOLIN_BW_ADJUST,
        density_norm="width",
        common_norm=False,
        width=VIOLIN_WIDTH,
        linewidth=VIOLIN_LINE_WIDTH,
        ax=ax,
        zorder=1,
    )

    # Apply independent opacity to each colored violin half.
    target_rgb = {media: np.asarray(to_rgb(color)) for media, color in VIOLIN_COLORS.items()}
    for collection in ax.collections:
        if not isinstance(collection, PolyCollection) or not len(collection.get_facecolors()):
            continue
        face_rgb = collection.get_facecolors()[0, :3]
        closest_media = min(
            MEDIA_ORDER,
            key=lambda media: np.linalg.norm(face_rgb - target_rgb[media]),
        )
        collection.set_alpha(VIOLIN_ALPHA[closest_media])
        if VIOLIN_EDGE_COLOR == "none":
            collection.set_edgecolor("none")
        else:
            collection.set_edgecolor(VIOLIN_EDGE_COLOR)
            collection.set_linewidth(VIOLIN_LINE_WIDTH)

    # A local RNG makes jitter reproducible without changing global NumPy state.
    rng = np.random.default_rng(RANDOM_SEED)
    position_by_concentration = {
        concentration: position
        for position, concentration in enumerate(concentration_order)
    }
    hue_width = VIOLIN_WIDTH / len(MEDIA_ORDER)
    hue_offsets = {
        media: (index - (len(MEDIA_ORDER) - 1) / 2) * hue_width
        for index, media in enumerate(MEDIA_ORDER)
    }

    # Count the observations contributing to every split half.
    n_by_group = (
        plot_data.groupby(["concentration", "media"], observed=True)
        .size()
        .to_dict()
    )
    if PRINT_N_TO_CONSOLE:
        print(f"\nSample sizes for {peptide} | {parameter}:")
        for concentration in concentration_order:
            group_text = ", ".join(
                f"{media} n={n_by_group.get((concentration, media), 0):,}"
                for media in MEDIA_ORDER
            )
            print(f"  concentration {concentration}: {group_text}")

    if SHOW_N_ON_PLOT:
        for concentration in concentration_order:
            base_x = position_by_concentration[concentration]
            for media in MEDIA_ORDER:
                n_value = n_by_group.get((concentration, media), 0)
                label_color = DOT_COLORS[media] if N_LABEL_USE_MEDIA_COLOR else "#222222"
                ax.text(
                    base_x + hue_offsets[media],
                    N_LABEL_Y_POSITION,
                    N_LABEL_FORMAT.format(n=n_value),
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    fontsize=N_LABEL_FONT_SIZE,
                    fontweight=N_LABEL_FONT_WEIGHT,
                    color=label_color,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": N_LABEL_BACKGROUND_ALPHA,
                        "pad": 0.5,
                    },
                    zorder=4,
                )

    if SHOW_INDIVIDUAL_POINTS:
        for media in MEDIA_ORDER:
            media_dots = dot_data[dot_data["media"].eq(media)]
            x_positions = np.array(
                [
                    position_by_concentration[value] + hue_offsets[media]
                    for value in media_dots["concentration"]
                ],
                dtype=float,
            )
            x_positions += rng.uniform(-DOT_JITTER, DOT_JITTER, size=len(media_dots))
            ax.scatter(
                x_positions,
                media_dots[parameter],
                s=DOT_SIZE,
                c=DOT_COLORS[media],
                alpha=DOT_ALPHA,
                edgecolors=DOT_EDGE_COLOR,
                linewidths=DOT_EDGE_WIDTH,
                zorder=2,
                rasterized=False,
            )

    y_label = PARAMETER_LABELS.get(parameter, parameter.replace("_", " "))
    if SHOW_TITLE:
        ax.set_title(
            f"{peptide} | {parameter}",
            fontsize=TITLE_FONT_SIZE,
            fontweight="normal",
            pad=10,
        )
    x_label = (
        f"Concentration ({CONCENTRATION_UNIT})"
        if CONCENTRATION_UNIT
        else "Concentration"
    )
    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE, width=1.1)
    # Draw a complete black border around the plotting area.
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(AXES_BORDER_COLOR)
        spine.set_linewidth(AXES_BORDER_WIDTH)
    legend_handles = [
        Patch(
            facecolor=VIOLIN_COLORS[media],
            edgecolor=(VIOLIN_EDGE_COLOR if VIOLIN_EDGE_COLOR != "none" else "none"),
            linewidth=VIOLIN_LINE_WIDTH,
            label=media,
            alpha=VIOLIN_ALPHA[media],
        )
        for media in MEDIA_ORDER
    ]
    legend = ax.legend(
        handles=legend_handles,
        title="media",
        loc=LEGEND_LOCATION,
        frameon=SHOW_LEGEND_FRAME,
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_TITLE_FONT_SIZE,
    )
    if SHOW_LEGEND_FRAME:
        legend.get_frame().set_edgecolor("#C8C8C8")
        legend.get_frame().set_linewidth(1.0)
    if Y_AXIS_STARTS_AT_ZERO:
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    output_stem = OUTPUT_DIR / f"{safe_filename(peptide)}__{safe_filename(parameter)}__split_violin"
    if SAVE_SVG:
        fig.savefig(
            output_stem.with_suffix(".svg"),
            bbox_inches="tight",
            transparent=TRANSPARENT_BACKGROUND,
        )
    if SAVE_PNG:
        fig.savefig(
            output_stem.with_suffix(".png"),
            dpi=PNG_DPI,
            bbox_inches="tight",
            transparent=TRANSPARENT_BACKGROUND,
        )
    plt.close(fig)
    print(f"Saved: {output_stem.name}.svg/.png ({len(plot_data):,} total points)")


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find input CSV: {CSV_PATH.resolve()}")

    data = pd.read_csv(CSV_PATH)
    print("Available peptides:", sorted(data["peptide"].dropna().unique()))
    print("Available concentrations:", sorted(data["concentration"].dropna().unique()))
    print("Numeric parameters:", list(data.select_dtypes(include="number").columns))

    validate_configuration(data)
    data = data[data["media"].isin(MEDIA_ORDER)]
    data = apply_optional_filter(data, "timepoint", TIMEPOINTS_TO_INCLUDE)
    data = apply_optional_filter(data, "rep", REPLICATES_TO_INCLUDE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="ticks", font=FONT_FAMILY)

    for peptide in PEPTIDES_TO_PLOT:
        for parameter in PARAMETERS_TO_PLOT:
            make_plot(data, peptide, parameter)


if __name__ == "__main__":
    main()