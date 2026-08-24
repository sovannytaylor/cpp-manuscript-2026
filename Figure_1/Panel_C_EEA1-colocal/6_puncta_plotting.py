"""
Plot peptide–EEA1 puncta overlap from the Yen-only detection output.

UPDATED FIGURE VERSION: cell n below x-axis, no grid, bold x tick labels.

This script DOES NOT rerun puncta detection and DOES NOT calculate or apply a
second GP30 floor. The upstream peptide–EEA1 analysis already:

1. Detects peptide and EEA1 puncta.
2. Calculates the global GP30 mean-punctum-intensity floor.
3. Removes peptide puncta at or below that floor.
4. Saves the retained puncta to peptide_eea1_puncta_features.csv.

This downstream script:

1. Loads the retained peptide puncta.
2. Excludes negative-control peptides from final figures.
3. Recalculates exact and exact-plus-proximity overlap for each cell.
4. Saves per-cell, per-peptide, per-replicate, and per-image summaries.
5. Makes bar and violin plots with individual cells and replicate means.

Primary cell-level metric:

    sum(within_distance_pixels)
    ---------------------------------- x 100
    sum(peptide_punctum_area_px)

This is the percentage of retained peptide-punctum pixels that overlap an
EEA1 punctum exactly or lie within the configured proximity distance used by
the upstream detection script (currently 2 pixels).
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

# Use Arial throughout the figures and preserve editable SVG text.
plt.rcParams["font.family"] = "Arial"
plt.rcParams["svg.fonttype"] = "none"


# ============================================================
# CONFIG
# ============================================================

# This must match OUTPUT_FOLDER in the peptide–EEA1 detection script.
ANALYSIS_FOLDER = Path(
    r"results\peptide_eea1_yen_only_global_gp30_v5"
)

INPUT_PUNCTA_CSV = (
    ANALYSIS_FOLDER / "peptide_eea1_puncta_features.csv"
)

OUTPUT_FOLDER = (
    ANALYSIS_FOLDER / "puncta_overlap_plots_v1"
)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# Negative controls were used upstream to define/filter peptide puncta.
# They are excluded from the experimental peptide figures by default.
NEGATIVE_CONTROLS = {
    "GP30",
    "NEG",
    "PA30",
}

INCLUDE_NEGATIVE_CONTROLS_IN_FINAL_OUTPUT = False


# Publication labels and fixed left-to-right order. The alias map below accepts
# common spellings from the input CSV and converts them to these display names.
PEPTIDE_ORDER = [
    "LL-37", "HTN3", "BMAP", "PR-39", "CAL1", "CAT4", "CROT", "BUFO",
    "OREO", "LT1A", "LT8A", "CU1A", "MAUR", "PARA", "CECR", "MOLL",
    "ARMI", "TRIC",
]

PEPTIDE_NAME_MAP = {
    "LL37": "LL-37",
    "LL-37": "LL-37",
    "HTN3": "HTN3",
    "BMAP": "BMAP",
    "BMAP27": "BMAP",
    "BMAP-27": "BMAP",
    "PR39": "PR-39",
    "CATHL1": "CAL1",
    "CAT4": "CAT4",
    "CATH41": "CAT4",
    "CATH4.1": "CAT4",
    "CROT": "CROT",
    "CROTAMINE": "CROT",
    "BUF": "BUFO",
    "OREO": "OREO",
    "LTC1": "LT1A",
    "LT8A": "LT8A",
    "CU1A": "CU1A",
    "MAUR": "MAUR",
    "MAURI": "MAUR",
    "MAURICIDIN": "MAUR",
    "PARA": "PARA",
    "PARAB": "PARA",
    "PARABUTOPORIN": "PARA",
    "CECR": "CECR",
    "CECRO": "CECR",
    "MOLLUSC": "MOLL",
    "MOLLUSCIDIN": "MOLL",
    "ARMIN": "ARMI",
    "TRICH": "TRIC",
}


# Available calculated metrics:
#   "within_distance_overlap_percent"
#       Percent of peptide-punctum pixels overlapping or within 2 px of EEA1.
#   "overlap_percent"
#       Percent of peptide-punctum pixels exactly overlapping EEA1.
#   "puncta_within_distance_percent"
#       Percent of peptide puncta with any pixel within 2 px of EEA1.
#   "puncta_exact_overlap_percent"
#       Percent of peptide puncta with any exact EEA1 overlap.
PRIMARY_OVERLAP_METRIC = "within_distance_overlap_percent"

Y_LABELS = {
    "within_distance_overlap_percent": (
        "Peptide-punctum pixels overlapping or within 2 px of EEA1 (%)"
    ),
    "overlap_percent": (
        "Peptide-punctum pixels exactly overlapping EEA1 (%)"
    ),
    "puncta_within_distance_percent": (
        "Peptide puncta overlapping or within 2 px of EEA1 (%)"
    ),
    "puncta_exact_overlap_percent": (
        "Peptide puncta exactly overlapping EEA1 (%)"
    ),
}


# ============================================================
# FIGURE SETTINGS
# ============================================================

FIGURE_WIDTH = 20
FIGURE_HEIGHT = 7

BAR_WIDTH = 0.65
VIOLIN_WIDTH = 1.0

CELL_POINT_SIZE = 18
REPLICATE_POINT_SIZE = 90

CELL_JITTER_WIDTH = 0.22
REPLICATE_OFFSET_WIDTH = 0.12

RANDOM_SEED = 42
DPI = 300
SHOW_FIGURES = True

# Text and legend settings
X_TICK_FONT_SIZE = 20
Y_TICK_FONT_SIZE = 20
X_LABEL_FONT_SIZE = 20
Y_LABEL_FONT_SIZE = 20
TITLE_FONT_SIZE = 20
N_LABEL_FONT_SIZE = 18
LEGEND_FONT_SIZE = 12
N_LABEL_Y_POSITION = -0.25
X_LABEL_PADDING = 30

# Change to True to include the plot legend.
SHOW_LEGEND = False

# Custom salmon-to-burgundy gradient matched to the reference figure. This
# avoids the pale pink and extremely dark ends of Matplotlib's "Reds" map.
RED_GRADIENT_LIGHT = "#FAB3B9"
RED_GRADIENT_DARK = "#A63D4D"
CELL_POINT_COLOR = "#4A4A4A"
MEAN_COLOR = "#000000"


# ============================================================
# HELPERS
# ============================================================

def clean_boolean_column(series):
    """Convert common CSV boolean representations to True/False."""

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


def save_figure(figure, filename_stem):
    """Save one figure as both PNG and editable SVG."""

    png_path = OUTPUT_FOLDER / f"{filename_stem}.png"
    svg_path = OUTPUT_FOLDER / f"{filename_stem}.svg"

    figure.savefig(png_path, dpi=DPI, bbox_inches="tight")
    figure.savefig(svg_path, format="svg", bbox_inches="tight")

    print(f"Saved PNG: {png_path}")
    print(f"Saved SVG: {svg_path}")


def sem(series):
    """Return SEM from nonmissing values."""

    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) <= 1:
        return np.nan
    return float(values.std(ddof=1) / np.sqrt(len(values)))


# ============================================================
# LOAD AND VALIDATE RETAINED PUNCTA
# ============================================================

if not INPUT_PUNCTA_CSV.exists():
    raise FileNotFoundError(
        "Could not find the peptide–EEA1 puncta CSV:\n"
        f"  {INPUT_PUNCTA_CSV}\n\n"
        "Run the peptide–EEA1 Yen/global-GP30 analysis first, or update "
        "ANALYSIS_FOLDER at the top of this script."
    )


puncta_df = pd.read_csv(INPUT_PUNCTA_CSV)


required_columns = {
    "image_name",
    "peptide",
    "replicate",
    "field",
    "cell_number",
    "peptide_punctum_label",
    "peptide_punctum_area_px",
    "peptide_punctum_mean_intensity",
    "exact_overlap",
    "within_distance_overlap",
    "exact_overlap_pixels",
    "within_distance_pixels",
    "gp30_floor",
}

missing_columns = sorted(required_columns - set(puncta_df.columns))

if missing_columns:
    raise ValueError(
        "The peptide–EEA1 puncta CSV is missing required columns:\n  "
        + "\n  ".join(missing_columns)
    )

if puncta_df.empty:
    raise RuntimeError(
        "The peptide–EEA1 puncta CSV contains no retained puncta."
    )


# ============================================================
# CLEAN DATA
# ============================================================

puncta_df["peptide"] = (
    puncta_df["peptide"].astype(str).str.strip().str.upper()
)

# Rename CSV peptide codes before filtering, ordering, summaries, and plotting.
puncta_df["peptide"] = puncta_df["peptide"].replace(PEPTIDE_NAME_MAP)

puncta_df["replicate"] = (
    puncta_df["replicate"].astype(str).str.strip().str.upper()
)

puncta_df["image_name"] = (
    puncta_df["image_name"].astype(str).str.strip()
)

puncta_df["exact_overlap"] = clean_boolean_column(
    puncta_df["exact_overlap"]
)

puncta_df["within_distance_overlap"] = clean_boolean_column(
    puncta_df["within_distance_overlap"]
)

numeric_columns = [
    "field",
    "cell_number",
    "peptide_punctum_label",
    "peptide_punctum_area_px",
    "peptide_punctum_mean_intensity",
    "exact_overlap_pixels",
    "within_distance_pixels",
    "gp30_floor",
]

for column in numeric_columns:
    puncta_df[column] = pd.to_numeric(
        puncta_df[column],
        errors="coerce",
    )

puncta_df = puncta_df.dropna(
    subset=[
        "peptide",
        "replicate",
        "image_name",
        "cell_number",
        "peptide_punctum_label",
        "peptide_punctum_area_px",
        "exact_overlap_pixels",
        "within_distance_pixels",
    ]
).copy()

puncta_df = puncta_df.loc[
    (puncta_df["peptide_punctum_area_px"] > 0)
    & (puncta_df["exact_overlap_pixels"] >= 0)
    & (puncta_df["within_distance_pixels"] >= 0)
].copy()


# The upstream analysis should already guarantee these relationships.
# Stop if the CSV is internally inconsistent instead of silently plotting it.
invalid_exact = (
    puncta_df["exact_overlap_pixels"]
    > puncta_df["peptide_punctum_area_px"]
)

invalid_near = (
    puncta_df["within_distance_pixels"]
    > puncta_df["peptide_punctum_area_px"]
)

invalid_order = (
    puncta_df["exact_overlap_pixels"]
    > puncta_df["within_distance_pixels"]
)

if invalid_exact.any() or invalid_near.any() or invalid_order.any():
    raise ValueError(
        "The puncta CSV contains impossible overlap values. Exact and "
        "within-distance overlap pixels must not exceed punctum area, and "
        "exact overlap must not exceed within-distance overlap."
    )


# Record the GP30-floor status for QC. Do not filter again here.
puncta_df["above_saved_gp30_floor"] = (
    puncta_df["peptide_punctum_mean_intensity"]
    > puncta_df["gp30_floor"]
)

unexpected_below_floor = puncta_df.loc[
    puncta_df["gp30_floor"].notna()
    & ~puncta_df["above_saved_gp30_floor"]
].copy()

if not unexpected_below_floor.empty:
    print(
        "WARNING: "
        f"{len(unexpected_below_floor)} saved puncta are at or below their "
        "recorded GP30 floor. They were not automatically removed again. "
        "See puncta_at_or_below_saved_gp30_floor.csv."
    )

unexpected_below_floor.to_csv(
    OUTPUT_FOLDER / "puncta_at_or_below_saved_gp30_floor.csv",
    index=False,
)


# ============================================================
# CHOOSE FINAL PEPTIDES
# ============================================================

if INCLUDE_NEGATIVE_CONTROLS_IN_FINAL_OUTPUT:
    final_puncta_df = puncta_df.copy()
else:
    final_puncta_df = puncta_df.loc[
        ~puncta_df["peptide"].isin(NEGATIVE_CONTROLS)
    ].copy()

if final_puncta_df.empty:
    raise RuntimeError(
        "No experimental peptide puncta remained after excluding controls."
    )


# ============================================================
# RECALCULATE CELL-LEVEL OVERLAP
# ============================================================

cell_group_columns = [
    "peptide",
    "replicate",
    "field",
    "image_name",
    "cell_number",
]

cell_summary = (
    final_puncta_df
    .groupby(cell_group_columns, as_index=False, dropna=False)
    .agg(
        gp30_floor=("gp30_floor", "first"),
        n_retained_peptide_puncta=("peptide_punctum_label", "count"),
        total_retained_peptide_area_px=("peptide_punctum_area_px", "sum"),
        total_exact_overlap_pixels=("exact_overlap_pixels", "sum"),
        total_within_distance_pixels=("within_distance_pixels", "sum"),
        n_puncta_with_exact_overlap=("exact_overlap", "sum"),
        n_puncta_within_distance=("within_distance_overlap", "sum"),
        mean_retained_punctum_intensity=(
            "peptide_punctum_mean_intensity",
            "mean",
        ),
        median_retained_punctum_intensity=(
            "peptide_punctum_mean_intensity",
            "median",
        ),
    )
)


cell_summary["overlap_percent"] = (
    cell_summary["total_exact_overlap_pixels"]
    / cell_summary["total_retained_peptide_area_px"]
    * 100
)

cell_summary["within_distance_overlap_percent"] = (
    cell_summary["total_within_distance_pixels"]
    / cell_summary["total_retained_peptide_area_px"]
    * 100
)

cell_summary["puncta_exact_overlap_percent"] = (
    cell_summary["n_puncta_with_exact_overlap"]
    / cell_summary["n_retained_peptide_puncta"]
    * 100
)

cell_summary["puncta_within_distance_percent"] = (
    cell_summary["n_puncta_within_distance"]
    / cell_summary["n_retained_peptide_puncta"]
    * 100
)

cell_summary = (
    cell_summary
    .replace([np.inf, -np.inf], np.nan)
    .dropna(subset=[PRIMARY_OVERLAP_METRIC])
    .copy()
)

if cell_summary.empty:
    raise RuntimeError(
        "No peptide-positive cells were available for plotting."
    )


# ============================================================
# PEPTIDE ORDER
# ============================================================

observed_peptides = list(
    cell_summary["peptide"].drop_duplicates()
)

if PEPTIDE_ORDER:
    requested_order = list(PEPTIDE_ORDER)

    peptide_order = [
        peptide
        for peptide in requested_order
        if peptide in observed_peptides
    ]

    peptide_order.extend(
        peptide
        for peptide in observed_peptides
        if peptide not in peptide_order
    )
else:
    peptide_order = observed_peptides


for dataframe in (cell_summary, final_puncta_df):
    dataframe["peptide"] = pd.Categorical(
        dataframe["peptide"],
        categories=peptide_order,
        ordered=True,
    )

cell_summary = cell_summary.sort_values(
    [
        "peptide",
        "replicate",
        "image_name",
        "cell_number",
    ]
).reset_index(drop=True)

final_puncta_df = final_puncta_df.sort_values(
    [
        "peptide",
        "replicate",
        "image_name",
        "cell_number",
        "peptide_punctum_label",
    ]
).reset_index(drop=True)


# ============================================================
# SUMMARY TABLES
# ============================================================

peptide_summary = (
    cell_summary
    .groupby("peptide", observed=True, as_index=False)
    .agg(
        mean_overlap_percent=(PRIMARY_OVERLAP_METRIC, "mean"),
        median_overlap_percent=(PRIMARY_OVERLAP_METRIC, "median"),
        std_overlap_percent=(PRIMARY_OVERLAP_METRIC, "std"),
        minimum_overlap_percent=(PRIMARY_OVERLAP_METRIC, "min"),
        maximum_overlap_percent=(PRIMARY_OVERLAP_METRIC, "max"),
        mean_exact_pixel_overlap_percent=("overlap_percent", "mean"),
        mean_within_distance_pixel_overlap_percent=(
            "within_distance_overlap_percent",
            "mean",
        ),
        mean_puncta_exact_overlap_percent=(
            "puncta_exact_overlap_percent",
            "mean",
        ),
        mean_puncta_within_distance_percent=(
            "puncta_within_distance_percent",
            "mean",
        ),
        mean_retained_puncta_per_cell=(
            "n_retained_peptide_puncta",
            "mean",
        ),
        n_cells=("cell_number", "count"),
        n_images=("image_name", "nunique"),
        n_replicates=("replicate", "nunique"),
    )
)

peptide_summary["sem_overlap_percent"] = (
    peptide_summary["std_overlap_percent"]
    / np.sqrt(peptide_summary["n_cells"])
)


replicate_summary = (
    cell_summary
    .groupby(
        ["peptide", "replicate"],
        observed=True,
        as_index=False,
    )
    .agg(
        replicate_mean_overlap_percent=(
            PRIMARY_OVERLAP_METRIC,
            "mean",
        ),
        replicate_median_overlap_percent=(
            PRIMARY_OVERLAP_METRIC,
            "median",
        ),
        replicate_std_overlap_percent=(
            PRIMARY_OVERLAP_METRIC,
            "std",
        ),
        mean_retained_puncta_per_cell=(
            "n_retained_peptide_puncta",
            "mean",
        ),
        n_cells=("cell_number", "count"),
        n_images=("image_name", "nunique"),
    )
)

replicate_summary["replicate_sem_overlap_percent"] = (
    replicate_summary
    .groupby("peptide", observed=True)[
        "replicate_mean_overlap_percent"
    ]
    .transform(sem)
)


image_summary = (
    cell_summary
    .groupby(
        ["peptide", "replicate", "field", "image_name"],
        observed=True,
        as_index=False,
        dropna=False,
    )
    .agg(
        image_mean_overlap_percent=(PRIMARY_OVERLAP_METRIC, "mean"),
        image_median_overlap_percent=(PRIMARY_OVERLAP_METRIC, "median"),
        image_std_overlap_percent=(PRIMARY_OVERLAP_METRIC, "std"),
        mean_retained_puncta_per_cell=(
            "n_retained_peptide_puncta",
            "mean",
        ),
        n_cells=("cell_number", "count"),
    )
)

image_summary["image_sem_overlap_percent"] = (
    image_summary["image_std_overlap_percent"]
    / np.sqrt(image_summary["n_cells"])
)


# ============================================================
# SAVE CSV OUTPUTS
# ============================================================

puncta_df.to_csv(
    OUTPUT_FOLDER / "all_upstream_retained_puncta_qc.csv",
    index=False,
)

final_puncta_df.to_csv(
    OUTPUT_FOLDER / "experimental_peptide_puncta.csv",
    index=False,
)

cell_summary.to_csv(
    OUTPUT_FOLDER / "peptide_eea1_cell_overlap_summary.csv",
    index=False,
)

peptide_summary.to_csv(
    OUTPUT_FOLDER / "peptide_eea1_peptide_summary.csv",
    index=False,
)

replicate_summary.to_csv(
    OUTPUT_FOLDER / "peptide_eea1_replicate_summary.csv",
    index=False,
)

image_summary.to_csv(
    OUTPUT_FOLDER / "peptide_eea1_image_summary.csv",
    index=False,
)


# ============================================================
# PLOT SUPPORT
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)
x_positions = np.arange(len(peptide_order))

red_cmap = LinearSegmentedColormap.from_list(
    "salmon_to_burgundy",
    [RED_GRADIENT_LIGHT, RED_GRADIENT_DARK],
)
peptide_colors = {
    peptide: red_cmap(color_position)
    for peptide, color_position in zip(
        peptide_order,
        np.linspace(0.0, 1.0, len(peptide_order)),
    )
}

unique_replicates = sorted(
    replicate_summary["replicate"].astype(str).unique()
)

if len(unique_replicates) == 1:
    replicate_offsets = {unique_replicates[0]: 0.0}
else:
    replicate_offsets = dict(
        zip(
            unique_replicates,
            np.linspace(
                -REPLICATE_OFFSET_WIDTH,
                REPLICATE_OFFSET_WIDTH,
                len(unique_replicates),
            ),
        )
    )


def add_cell_points(axis):
    """Add one lightly jittered point for each peptide-positive cell."""

    for x_position, peptide in zip(x_positions, peptide_order):
        current_cells = cell_summary.loc[
            cell_summary["peptide"] == peptide
        ]

        jitter = rng.uniform(
            low=-CELL_JITTER_WIDTH,
            high=CELL_JITTER_WIDTH,
            size=len(current_cells),
        )

        axis.scatter(
            x_position + jitter,
            current_cells[PRIMARY_OVERLAP_METRIC],
            s=CELL_POINT_SIZE,
            alpha=0.35,
            color=CELL_POINT_COLOR,
            edgecolors="none",
            zorder=3,
        )


def add_replicate_points(axis):
    """Add one larger dark point per peptide and biological replicate."""

    for x_position, peptide in zip(x_positions, peptide_order):
        current_replicates = replicate_summary.loc[
            replicate_summary["peptide"] == peptide
        ]

        for row in current_replicates.itertuples():
            offset = replicate_offsets[str(row.replicate)]

            axis.scatter(
                x_position + offset,
                row.replicate_mean_overlap_percent,
                s=REPLICATE_POINT_SIZE,
                marker="o",
                facecolor=MEAN_COLOR,
                edgecolor="white",
                linewidth=1.0,
                zorder=6,
            )

def add_n_labels(axis):
    """Add cell counts beneath the x-axis categories."""

    for x_position, peptide in zip(x_positions, peptide_order):
        row = peptide_summary.loc[
            peptide_summary["peptide"] == peptide
        ].iloc[0]

        axis.text(
            x_position,
            N_LABEL_Y_POSITION,
            f"n={int(row['n_cells'])}",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=N_LABEL_FONT_SIZE,
            linespacing=1.15,
            clip_on=False,
        )

def format_axis(axis, title):
    """Apply common formatting to bar and violin plots."""

    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        peptide_order,
        rotation=45,
        ha="right",
        fontsize=X_TICK_FONT_SIZE,
        fontweight="semibold",
    )
    axis.tick_params(axis="y", labelsize=Y_TICK_FONT_SIZE)
    for label in axis.get_yticklabels():
        label.set_fontweight("semibold")
    axis.set_xlabel(
        "Peptide",
        fontsize=X_LABEL_FONT_SIZE,
        labelpad=X_LABEL_PADDING,
    )
    axis.set_ylabel(
        Y_LABELS[PRIMARY_OVERLAP_METRIC],
        fontsize=Y_LABEL_FONT_SIZE,
    )
    axis.set_ylim(0, 105)
    axis.set_title(title, fontsize=TITLE_FONT_SIZE, pad=38)

    # Draw a complete rectangular black border around the plotting area.
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.5)

    axis.grid(False)

    if SHOW_LEGEND:
        axis.scatter(
            [],
            [],
            s=CELL_POINT_SIZE,
            alpha=0.35,
            color=CELL_POINT_COLOR,
            label="Individual cell",
        )

        axis.scatter(
            [],
            [],
            s=REPLICATE_POINT_SIZE,
            facecolor=MEAN_COLOR,
            edgecolor="white",
            linewidth=1.0,
            label="Replicate mean",
        )

        axis.legend(
            frameon=False,
            fontsize=LEGEND_FONT_SIZE,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0,
        )
    add_n_labels(axis)


# ============================================================
# BAR PLOT
# ============================================================

bar_means = []
bar_sems = []

for peptide in peptide_order:
    current = peptide_summary.loc[
        peptide_summary["peptide"] == peptide
    ].iloc[0]

    bar_means.append(float(current["mean_overlap_percent"]))
    bar_sems.append(float(current["sem_overlap_percent"]))


fig, ax = plt.subplots(
    figsize=(FIGURE_WIDTH, FIGURE_HEIGHT)
)

ax.bar(
    x_positions,
    bar_means,
    width=BAR_WIDTH,
    color=[peptide_colors[peptide] for peptide in peptide_order],
    alpha=0.90,
    edgecolor="black",
    linewidth=1.2,
    zorder=1,
)

ax.errorbar(
    x_positions,
    bar_means,
    yerr=bar_sems,
    fmt="none",
    ecolor="black",
    elinewidth=1.2,
    capsize=4,
    capthick=1.2,
    zorder=5,
)

add_cell_points(ax)
add_replicate_points(ax)
format_axis(
    ax,
    "Peptide–EEA1 exact overlap + 2 px proximity",
)

if SHOW_LEGEND:
    fig.tight_layout(rect=[0, 0, 0.90, 0.94])
else:
    fig.tight_layout(rect=[0, 0, 1.0, 0.94])
save_figure(
    fig,
    "peptide_eea1_exact_plus_2px_barplot",
)

if SHOW_FIGURES:
    plt.show()
else:
    plt.close(fig)


# ============================================================
# VIOLIN PLOT
# ============================================================

violin_values = []

for peptide in peptide_order:
    values = (
        cell_summary.loc[
            cell_summary["peptide"] == peptide,
            PRIMARY_OVERLAP_METRIC,
        ]
        .dropna()
        .to_numpy(dtype=float)
    )

    if len(values) == 0:
        raise RuntimeError(
            f"No values were found for peptide {peptide}."
        )

    if len(values) == 1:
        density_values = np.array(
            [values[0] - 1e-6, values[0] + 1e-6]
        )
    elif np.std(values) == 0:
        density_values = values + np.linspace(
            -1e-6,
            1e-6,
            len(values),
        )
    else:
        density_values = values

    violin_values.append(density_values)


fig, ax = plt.subplots(
    figsize=(FIGURE_WIDTH, FIGURE_HEIGHT)
)

violin_parts = ax.violinplot(
    violin_values,
    positions=x_positions,
    widths=VIOLIN_WIDTH,
    showmeans=False,
    showmedians=False,
    showextrema=False,
)

for body, peptide in zip(violin_parts["bodies"], peptide_order):
    body.set_facecolor(peptide_colors[peptide])
    body.set_alpha(0.90)
    body.set_edgecolor("#333333")
    body.set_linewidth(2.0)

for x_position, peptide in zip(x_positions, peptide_order):
    mean_value = float(
        cell_summary.loc[
            cell_summary["peptide"] == peptide,
            PRIMARY_OVERLAP_METRIC,
        ].mean()
    )

    ax.hlines(
        y=mean_value,
        xmin=x_position - 0.20,
        xmax=x_position + 0.20,
        color=MEAN_COLOR,
        linewidth=2.0,
        zorder=5,
    )

add_cell_points(ax)
add_replicate_points(ax)
format_axis(
    ax,
    "Distribution of peptide–EEA1 exact overlap + 2 px proximity",
)

if SHOW_LEGEND:
    ax.plot(
        [],
        [],
        color=MEAN_COLOR,
        linewidth=2.0,
        label="Overall cell mean",
    )

    handles, labels = ax.get_legend_handles_labels()
    unique_legend = dict(zip(labels, handles))

    ax.legend(
        unique_legend.values(),
        unique_legend.keys(),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
    )

if SHOW_LEGEND:
    fig.tight_layout(rect=[0, 0, 0.90, 0.94])
else:
    fig.tight_layout(rect=[0, 0, 1.0, 0.94])
save_figure(
    fig,
    "peptide_eea1_exact_plus_2px_violinplot",
)

if SHOW_FIGURES:
    plt.show()
else:
    plt.close(fig)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("PEPTIDE–EEA1 PUNCTA OVERLAP PLOTTING COMPLETE")
print("=" * 70)

print()
print(
    "Input puncta were already filtered by the upstream global GP30 "
    "punctum-intensity floor."
)

print()
print("Retained peptide-positive cells:")
print(
    cell_summary
    .groupby("peptide", observed=True)
    .size()
    .rename("n_cells")
)

print()
print("Biological replicates:")
print(
    cell_summary
    .groupby("peptide", observed=True)["replicate"]
    .nunique()
    .rename("n_replicates")
)

print()
print("Saved CSV files:")

for csv_path in sorted(OUTPUT_FOLDER.glob("*.csv")):
    print(f"  {csv_path.name}")

print()
print(f"All outputs saved to: {OUTPUT_FOLDER}")