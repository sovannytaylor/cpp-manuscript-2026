"""
Downstream negative-control filtering of detected peptide puncta.

This script DOES NOT rerun puncta detection.

Workflow
--------
1. Load peptide_ldl_puncta_features.csv.
2. Use GP30, NEG, and PA30 puncta to calculate an intensity floor
   separately for each plate.
3. Remove peptide puncta whose mean intensity is less than or equal
   to the plate-specific negative-control floor.
4. Remove cells where no peptide puncta remain after filtering.
5. Recalculate each cell's peptide–LDL overlap.
6. Save:
   - filtered puncta CSV
   - filtered cell CSV
   - peptide summary CSV
   - plate summary CSV
   - control-floor CSVs
   - filtering QC CSVs
   - bar plot as PNG and SVG
   - violin plot as PNG and SVG

Cell-level exact-plus-proximity overlap is recalculated as:

    sum(within_distance_pixels)
    ----------------------------- × 100
    sum(peptide_punctum_area_px)

This represents the percentage of retained peptide-punctum pixels
that overlap detected LDL pixels or fall within the configured
2-pixel LDL proximity region. Exact overlap is still saved separately.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_PUNCTA_CSV = Path(
    r"results\peptide_ldl_overlap_selected_methods"
    r"\peptide_ldl_puncta_features.csv"
)

OUTPUT_FOLDER = Path(
    r"results\peptide_ldl_overlap_selected_methods"
    r"\negative_control_filtered_figures"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# NEGATIVE-CONTROL SETTINGS
# ============================================================

NEGATIVE_CONTROLS = {
    "GP30",
    "NEG",
    "PA30",
}


# Options:
#
# "pooled_mean"
#     Pool every GP30, NEG, and PA30 punctum on a plate and
#     calculate one mean.
#
# "mean_of_control_means"
#     Calculate one mean for each control peptide, then average
#     those control means. Each control contributes equally.
#
# "max_control_mean"
#     Calculate one mean for each control and use the highest.
#     This is the most conservative option.
CONTROL_FLOOR_MODE = "max_control_mean"


# What to do if a plate has no detected negative-control puncta.
#
# "global"
#     Use the global control floor calculated from all plates.
#
# "skip"
#     Remove puncta from plates without a control floor.
#
# "zero"
#     Do not apply an additional intensity floor on those plates.
MISSING_PLATE_FLOOR_ACTION = "global"


# ============================================================
# FINAL OUTPUT SETTINGS
# ============================================================

INCLUDE_GP30_IN_FINAL_OUTPUT = False
INCLUDE_NEG_IN_FINAL_OUTPUT = False
INCLUDE_PA30_IN_FINAL_OUTPUT = False


# Optional manual order.
# Leave empty to preserve the peptide order found in the CSV.
PEPTIDE_ORDER = [
    "LL-37", "HTN3", "BMAP", "PR-39", "CAL1", "CAT4", "CROT", "BUFO",
    "OREO", "LT1A", "LT8A", "CU1A", "MAUR", "PARA", "CECR", "MOLL",
    "ARMI", "TRIC",
]
#============================================================
# FIGURE SETTINGS
# ============================================================

PRIMARY_OVERLAP_METRIC = "within_distance_overlap_percent"
Y_LABEL = "Peptide pixels overlapping or within 2 px of LDL (%)"

FIGURE_WIDTH = 12
FIGURE_HEIGHT = 5.2

BAR_WIDTH = 0.72
VIOLIN_WIDTH = 0.82

CELL_POINT_SIZE = 14
PLATE_POINT_SIZE = 62

# These are the same three anchor reds used in the grouped HDL/LDL/VLDL plot.
# A custom gradient between them gives every peptide a distinct shade without
# introducing the orange-red and nearly brown tones from Matplotlib's "Reds".
RED_PALETTE_ANCHORS = [
    "#F05A67",  # bright light red
    "#C83E4D",  # medium red
    "#8F1D2C",  # dark red
]
CELL_POINT_COLOR = "#A9A9A9"
AVERAGE_POINT_COLOR = "black"

CELL_JITTER_WIDTH = 0.10
PLATE_OFFSET_WIDTH = 0.07

RANDOM_SEED = 42
DPI = 300

SHOW_FIGURES = True


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_boolean_column(series):
    """
    Convert common CSV boolean representations into True/False.
    """

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
    """
    Save one figure as PNG and SVG.
    """

    png_path = OUTPUT_FOLDER / f"{filename_stem}.png"
    svg_path = OUTPUT_FOLDER / f"{filename_stem}.svg"

    figure.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
    )

    figure.savefig(
        svg_path,
        format="svg",
        bbox_inches="tight",
    )

    print(f"Saved PNG: {png_path}")
    print(f"Saved SVG: {svg_path}")


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

puncta_df = pd.read_csv(
    INPUT_PUNCTA_CSV
)


required_columns = {
    "image_name",
    "peptide",
    "plate",
    "well",
    "field",
    "cell_number",
    "peptide_punctum_label",
    "peptide_punctum_area_px",
    "peptide_punctum_mean_intensity",
    "exact_overlap",
    "within_distance_overlap",
    "exact_overlap_pixels",
    "within_distance_pixels",
}

missing_columns = (
    required_columns
    - set(puncta_df.columns)
)

if missing_columns:
    raise ValueError(
        "The punctum CSV is missing these required columns:\n"
        + "\n".join(
            sorted(missing_columns)
        )
    )


# ============================================================
# CLEAN COLUMNS
# ============================================================

puncta_df["peptide"] = (
    puncta_df["peptide"]
    .astype(str)
    .str.strip()
    .str.upper()
)

puncta_df["plate"] = (
    puncta_df["plate"]
    .astype(str)
    .str.strip()
)

puncta_df["well"] = (
    puncta_df["well"]
    .astype(str)
    .str.strip()
    .str.upper()
)


numeric_columns = [
    "field",
    "cell_number",
    "peptide_punctum_label",
    "peptide_punctum_area_px",
    "peptide_punctum_mean_intensity",
    "peptide_punctum_max_intensity",
    "exact_overlap_pixels",
    "within_distance_pixels",
    "nearest_ldl_distance_px",
]

for column in numeric_columns:

    if column in puncta_df.columns:

        puncta_df[column] = pd.to_numeric(
            puncta_df[column],
            errors="coerce",
        )


puncta_df["exact_overlap"] = clean_boolean_column(
    puncta_df["exact_overlap"]
)

puncta_df["within_distance_overlap"] = clean_boolean_column(
    puncta_df["within_distance_overlap"]
)


puncta_df = puncta_df.dropna(
    subset=[
        "peptide",
        "plate",
        "image_name",
        "cell_number",
        "peptide_punctum_label",
        "peptide_punctum_area_px",
        "peptide_punctum_mean_intensity",
        "exact_overlap_pixels",
        "within_distance_pixels",
    ]
).copy()


puncta_df = puncta_df.loc[
    puncta_df["peptide_punctum_area_px"] > 0
].copy()


if puncta_df.empty:
    raise RuntimeError(
        "No valid peptide punctum rows were found."
    )


# ============================================================
# IDENTIFY NEGATIVE-CONTROL PUNCTA
# ============================================================

control_puncta_df = puncta_df.loc[
    puncta_df["peptide"].isin(
        NEGATIVE_CONTROLS
    )
].copy()


if control_puncta_df.empty:
    raise RuntimeError(
        "No GP30, NEG, or PA30 puncta were found in the punctum CSV."
    )


# ============================================================
# CONTROL SUMMARY BY PLATE AND CONTROL PEPTIDE
# ============================================================

control_summary = (
    control_puncta_df
    .groupby(
        [
            "plate",
            "peptide",
        ],
        as_index=False,
    )
    .agg(
        control_mean_punctum_intensity=(
            "peptide_punctum_mean_intensity",
            "mean",
        ),
        control_median_punctum_intensity=(
            "peptide_punctum_mean_intensity",
            "median",
        ),
        control_std_punctum_intensity=(
            "peptide_punctum_mean_intensity",
            "std",
        ),
        control_minimum_punctum_intensity=(
            "peptide_punctum_mean_intensity",
            "min",
        ),
        control_maximum_punctum_intensity=(
            "peptide_punctum_mean_intensity",
            "max",
        ),
        n_control_puncta=(
            "peptide_punctum_mean_intensity",
            "count",
        ),
        n_control_cells=(
            "cell_number",
            "nunique",
        ),
        n_control_images=(
            "image_name",
            "nunique",
        ),
    )
)


# ============================================================
# CALCULATE ONE NEGATIVE-CONTROL FLOOR PER PLATE
# ============================================================

plate_floor_rows = []


for plate, plate_controls in control_puncta_df.groupby(
    "plate"
):

    plate_control_summary = control_summary.loc[
        control_summary["plate"] == plate
    ].copy()

    if CONTROL_FLOOR_MODE == "pooled_mean":

        floor_value = float(
            plate_controls[
                "peptide_punctum_mean_intensity"
            ].mean()
        )

        floor_source = "POOLED_GP30_NEG_PA30"

    elif CONTROL_FLOOR_MODE == "mean_of_control_means":

        floor_value = float(
            plate_control_summary[
                "control_mean_punctum_intensity"
            ].mean()
        )

        floor_source = "MEAN_OF_CONTROL_MEANS"

    elif CONTROL_FLOOR_MODE == "max_control_mean":

        maximum_row_index = (
            plate_control_summary[
                "control_mean_punctum_intensity"
            ]
            .idxmax()
        )

        floor_value = float(
            plate_control_summary.loc[
                maximum_row_index,
                "control_mean_punctum_intensity",
            ]
        )

        floor_source = str(
            plate_control_summary.loc[
                maximum_row_index,
                "peptide",
            ]
        )

    else:
        raise ValueError(
            "CONTROL_FLOOR_MODE must be one of:\n"
            "'pooled_mean', "
            "'mean_of_control_means', or "
            "'max_control_mean'"
        )

    controls_present = sorted(
        plate_controls[
            "peptide"
        ].unique()
    )

    plate_floor_rows.append(
        {
            "plate": str(plate),
            "control_floor_mode": CONTROL_FLOOR_MODE,
            "control_intensity_floor": floor_value,
            "floor_source": floor_source,
            "controls_present": ",".join(
                controls_present
            ),
            "n_control_types_present": len(
                controls_present
            ),
            "n_control_puncta": len(
                plate_controls
            ),
            "n_control_cells": (
                plate_controls[
                    [
                        "image_name",
                        "cell_number",
                    ]
                ]
                .drop_duplicates()
                .shape[0]
            ),
            "n_control_images": (
                plate_controls[
                    "image_name"
                ].nunique()
            ),
        }
    )


plate_floor_df = pd.DataFrame(
    plate_floor_rows
)


plate_floor_lookup = {
    str(row.plate): float(
        row.control_intensity_floor
    )
    for row in plate_floor_df.itertuples()
}


# ============================================================
# CALCULATE GLOBAL FALLBACK FLOOR
# ============================================================

if CONTROL_FLOOR_MODE == "pooled_mean":

    global_control_floor = float(
        control_puncta_df[
            "peptide_punctum_mean_intensity"
        ].mean()
    )

elif CONTROL_FLOOR_MODE == "mean_of_control_means":

    global_control_means = (
        control_puncta_df
        .groupby(
            "peptide",
            as_index=False,
        )
        .agg(
            control_mean=(
                "peptide_punctum_mean_intensity",
                "mean",
            )
        )
    )

    global_control_floor = float(
        global_control_means[
            "control_mean"
        ].mean()
    )

elif CONTROL_FLOOR_MODE == "max_control_mean":

    global_control_means = (
        control_puncta_df
        .groupby(
            "peptide",
            as_index=False,
        )
        .agg(
            control_mean=(
                "peptide_punctum_mean_intensity",
                "mean",
            )
        )
    )

    global_control_floor = float(
        global_control_means[
            "control_mean"
        ].max()
    )

else:
    raise ValueError(
        f"Unsupported CONTROL_FLOOR_MODE: "
        f"{CONTROL_FLOOR_MODE}"
    )


plate_floor_df[
    "global_fallback_floor"
] = global_control_floor


# ============================================================
# ASSIGN A CONTROL FLOOR TO EVERY PUNCTUM
# ============================================================

def choose_floor_for_plate(plate):
    """
    Return the plate-specific floor or the configured fallback.
    """

    plate = str(plate)

    if plate in plate_floor_lookup:

        return (
            plate_floor_lookup[plate],
            "plate_specific",
        )

    if MISSING_PLATE_FLOOR_ACTION == "global":

        return (
            global_control_floor,
            "global_fallback",
        )

    if MISSING_PLATE_FLOOR_ACTION == "zero":

        return (
            0.0,
            "zero_fallback",
        )

    if MISSING_PLATE_FLOOR_ACTION == "skip":

        return (
            np.nan,
            "missing_skip",
        )

    raise ValueError(
        "MISSING_PLATE_FLOOR_ACTION must be "
        "'global', 'zero', or 'skip'"
    )


floor_assignments = puncta_df[
    "plate"
].map(
    choose_floor_for_plate
)


puncta_df[
    "negative_control_intensity_floor"
] = floor_assignments.map(
    lambda value: value[0]
)

puncta_df[
    "negative_control_floor_source"
] = floor_assignments.map(
    lambda value: value[1]
)


# ============================================================
# APPLY NEGATIVE-CONTROL FILTER
# ============================================================

# A punctum is retained only when its mean intensity is strictly
# greater than the control floor.
#
# Therefore, puncta equal to the floor are removed.
puncta_df[
    "passes_negative_control_filter"
] = (
    puncta_df[
        "peptide_punctum_mean_intensity"
    ]
    >
    puncta_df[
        "negative_control_intensity_floor"
    ]
)


filtered_puncta_df = puncta_df.loc[
    puncta_df[
        "passes_negative_control_filter"
    ]
].copy()


# ============================================================
# FILTERING QC SUMMARY
# ============================================================

filtering_summary = (
    puncta_df
    .groupby(
        [
            "peptide",
            "plate",
        ],
        as_index=False,
    )
    .agg(
        negative_control_intensity_floor=(
            "negative_control_intensity_floor",
            "first",
        ),
        floor_source=(
            "negative_control_floor_source",
            "first",
        ),
        n_puncta_before_filter=(
            "peptide_punctum_label",
            "count",
        ),
        n_puncta_after_filter=(
            "passes_negative_control_filter",
            "sum",
        ),
        mean_punctum_intensity_before_filter=(
            "peptide_punctum_mean_intensity",
            "mean",
        ),
        median_punctum_intensity_before_filter=(
            "peptide_punctum_mean_intensity",
            "median",
        ),
    )
)


filtering_summary[
    "n_puncta_removed"
] = (
    filtering_summary[
        "n_puncta_before_filter"
    ]
    -
    filtering_summary[
        "n_puncta_after_filter"
    ]
)


filtering_summary[
    "percent_puncta_removed"
] = (
    filtering_summary[
        "n_puncta_removed"
    ]
    /
    filtering_summary[
        "n_puncta_before_filter"
    ]
    * 100
)


after_filter_summary = (
    filtered_puncta_df
    .groupby(
        [
            "peptide",
            "plate",
        ],
        as_index=False,
    )
    .agg(
        mean_punctum_intensity_after_filter=(
            "peptide_punctum_mean_intensity",
            "mean",
        ),
        median_punctum_intensity_after_filter=(
            "peptide_punctum_mean_intensity",
            "median",
        ),
    )
)


filtering_summary = filtering_summary.merge(
    after_filter_summary,
    on=[
        "peptide",
        "plate",
    ],
    how="left",
)


# ============================================================
# REMOVE CONTROLS FROM FINAL BIOLOGICAL OUTPUT
# ============================================================

final_puncta_df = (
    filtered_puncta_df.copy()
)


if not INCLUDE_GP30_IN_FINAL_OUTPUT:

    final_puncta_df = final_puncta_df.loc[
        final_puncta_df["peptide"] != "GP30"
    ].copy()


if not INCLUDE_NEG_IN_FINAL_OUTPUT:

    final_puncta_df = final_puncta_df.loc[
        final_puncta_df["peptide"] != "NEG"
    ].copy()


if not INCLUDE_PA30_IN_FINAL_OUTPUT:

    final_puncta_df = final_puncta_df.loc[
        final_puncta_df["peptide"] != "PA30"
    ].copy()


if final_puncta_df.empty:
    raise RuntimeError(
        "No experimental peptide puncta remained after filtering."
    )


# ============================================================
# RECALCULATE CELL-LEVEL OVERLAP
# ============================================================

cell_group_columns = [
    "peptide",
    "plate",
    "well",
    "field",
    "image_name",
    "cell_number",
]


cell_summary = (
    final_puncta_df
    .groupby(
        cell_group_columns,
        as_index=False,
    )
    .agg(
        negative_control_intensity_floor=(
            "negative_control_intensity_floor",
            "first",
        ),
        negative_control_floor_source=(
            "negative_control_floor_source",
            "first",
        ),
        n_retained_peptide_puncta=(
            "peptide_punctum_label",
            "count",
        ),
        total_retained_peptide_area_px=(
            "peptide_punctum_area_px",
            "sum",
        ),
        total_exact_overlap_pixels=(
            "exact_overlap_pixels",
            "sum",
        ),
        total_within_distance_pixels=(
            "within_distance_pixels",
            "sum",
        ),
        n_puncta_with_exact_overlap=(
            "exact_overlap",
            "sum",
        ),
        n_puncta_within_distance=(
            "within_distance_overlap",
            "sum",
        ),
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


# Percentage of retained peptide pixels exactly overlapping LDL.
cell_summary[
    "overlap_percent"
] = (
    cell_summary[
        "total_exact_overlap_pixels"
    ]
    /
    cell_summary[
        "total_retained_peptide_area_px"
    ]
    * 100
)


# PRIMARY FIGURE METRIC:
# Percentage of retained peptide pixels that overlap LDL exactly
# or fall within the configured 2-pixel LDL proximity region.
cell_summary[
    "within_distance_overlap_percent"
] = (
    cell_summary[
        "total_within_distance_pixels"
    ]
    /
    cell_summary[
        "total_retained_peptide_area_px"
    ]
    * 100
)


# Percentage of retained peptide puncta with any exact overlap.
cell_summary[
    "puncta_exact_overlap_percent"
] = (
    cell_summary[
        "n_puncta_with_exact_overlap"
    ]
    /
    cell_summary[
        "n_retained_peptide_puncta"
    ]
    * 100
)


# Percentage of retained peptide puncta within the configured
# LDL distance.
cell_summary[
    "puncta_within_distance_percent"
] = (
    cell_summary[
        "n_puncta_within_distance"
    ]
    /
    cell_summary[
        "n_retained_peptide_puncta"
    ]
    * 100
)


cell_summary = cell_summary.replace(
    [
        np.inf,
        -np.inf,
    ],
    np.nan,
)


cell_summary = cell_summary.dropna(
    subset=[
        PRIMARY_OVERLAP_METRIC,
    ]
).copy()


if cell_summary.empty:
    raise RuntimeError(
        "No peptide-positive cells remained after filtering."
    )


# ============================================================
# CHOOSE PEPTIDE ORDER
# ============================================================

# Rename the preprocessing labels only after negative-control filtering.
# This preserves the original names while controls and intensity floors are
# being identified, but uses the requested names in final CSVs and figures.
PEPTIDE_DISPLAY_NAMES = {
    "LL37": "LL-37",
    "HTN3": "HTN3",
    "BMAP": "BMAP",
    "PR39": "PR-39",
    "CATHL1": "CAL1",
    "CATH41": "CAT4",
    "CROT": "CROT",
    "BUF": "BUFO",
    "OREO": "OREO",
    "LTC1": "LT1A",
    "LT8A": "LT8A",
    "CU1A": "CU1A",
    "MAURI": "MAUR",
    "PARAB": "PARA",
    "CECRO": "CECR",
    "MOLLUSC": "MOLL",
    "ARMIN": "ARMI",
    "TRICH": "TRIC",
}

cell_summary["peptide"] = (
    cell_summary["peptide"].astype(str).replace(PEPTIDE_DISPLAY_NAMES)
)

final_puncta_df["peptide"] = (
    final_puncta_df["peptide"].astype(str).replace(PEPTIDE_DISPLAY_NAMES)
)

observed_peptides = list(
    cell_summary[
        "peptide"
    ].drop_duplicates()
)


if PEPTIDE_ORDER:

    requested_order = [
        str(peptide)
        .strip()
        for peptide in PEPTIDE_ORDER
    ]

    peptide_order = [
        peptide
        for peptide in requested_order
        if peptide in observed_peptides
    ]

    peptide_order.extend(
        [
            peptide
            for peptide in observed_peptides
            if peptide not in peptide_order
        ]
    )

else:

    peptide_order = observed_peptides


cell_summary["peptide"] = pd.Categorical(
    cell_summary["peptide"],
    categories=peptide_order,
    ordered=True,
)


cell_summary = cell_summary.sort_values(
    [
        "peptide",
        "plate",
        "well",
        "field",
        "cell_number",
    ]
).reset_index(drop=True)


final_puncta_df["peptide"] = pd.Categorical(
    final_puncta_df["peptide"],
    categories=peptide_order,
    ordered=True,
)


final_puncta_df = final_puncta_df.sort_values(
    [
        "peptide",
        "plate",
        "well",
        "field",
        "cell_number",
        "peptide_punctum_label",
    ]
).reset_index(drop=True)


# ============================================================
# PEPTIDE-LEVEL SUMMARY
# ============================================================

peptide_summary = (
    cell_summary
    .groupby(
        "peptide",
        observed=True,
        as_index=False,
    )
    .agg(
        mean_overlap_percent=(
            PRIMARY_OVERLAP_METRIC,
            "mean",
        ),
        median_overlap_percent=(
            PRIMARY_OVERLAP_METRIC,
            "median",
        ),
        std_overlap_percent=(
            PRIMARY_OVERLAP_METRIC,
            "std",
        ),
        minimum_overlap_percent=(
            PRIMARY_OVERLAP_METRIC,
            "min",
        ),
        maximum_overlap_percent=(
            PRIMARY_OVERLAP_METRIC,
            "max",
        ),
        mean_within_distance_overlap_percent=(
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
        n_cells=(
            "cell_number",
            "count",
        ),
        n_images=(
            "image_name",
            "nunique",
        ),
        n_wells=(
            "well",
            "nunique",
        ),
        n_plates=(
            "plate",
            "nunique",
        ),
    )
)


peptide_summary[
    "sem_overlap_percent"
] = (
    peptide_summary[
        "std_overlap_percent"
    ]
    /
    np.sqrt(
        peptide_summary[
            "n_cells"
        ]
    )
)


# ============================================================
# PLATE-LEVEL SUMMARY
# ============================================================

plate_summary = (
    cell_summary
    .groupby(
        [
            "peptide",
            "plate",
        ],
        observed=True,
        as_index=False,
    )
    .agg(
        plate_mean_overlap_percent=(
            PRIMARY_OVERLAP_METRIC,
            "mean",
        ),
        plate_median_overlap_percent=(
            PRIMARY_OVERLAP_METRIC,
            "median",
        ),
        plate_std_overlap_percent=(
            PRIMARY_OVERLAP_METRIC,
            "std",
        ),
        plate_mean_within_distance_overlap_percent=(
            "within_distance_overlap_percent",
            "mean",
        ),
        mean_retained_puncta_per_cell=(
            "n_retained_peptide_puncta",
            "mean",
        ),
        n_cells=(
            "cell_number",
            "count",
        ),
        n_images=(
            "image_name",
            "nunique",
        ),
        n_wells=(
            "well",
            "nunique",
        ),
    )
)


plate_summary[
    "plate_sem_overlap_percent"
] = (
    plate_summary[
        "plate_std_overlap_percent"
    ]
    /
    np.sqrt(
        plate_summary[
            "n_cells"
        ]
    )
)


# ============================================================
# IMAGE-LEVEL SUMMARY
# ============================================================

image_summary = (
    cell_summary
    .groupby(
        [
            "peptide",
            "plate",
            "well",
            "field",
            "image_name",
        ],
        observed=True,
        as_index=False,
    )
    .agg(
        image_mean_overlap_percent=(
            PRIMARY_OVERLAP_METRIC,
            "mean",
        ),
        image_median_overlap_percent=(
            PRIMARY_OVERLAP_METRIC,
            "median",
        ),
        image_std_overlap_percent=(
            PRIMARY_OVERLAP_METRIC,
            "std",
        ),
        image_mean_within_distance_overlap_percent=(
            "within_distance_overlap_percent",
            "mean",
        ),
        mean_retained_puncta_per_cell=(
            "n_retained_peptide_puncta",
            "mean",
        ),
        n_cells=(
            "cell_number",
            "count",
        ),
    )
)


image_summary[
    "image_sem_overlap_percent"
] = (
    image_summary[
        "image_std_overlap_percent"
    ]
    /
    np.sqrt(
        image_summary[
            "n_cells"
        ]
    )
)


# ============================================================
# SAVE CSV OUTPUTS
# ============================================================

control_puncta_path = (
    OUTPUT_FOLDER
    / "negative_control_puncta.csv"
)

control_summary_path = (
    OUTPUT_FOLDER
    / "negative_control_summary_by_plate.csv"
)

plate_floor_path = (
    OUTPUT_FOLDER
    / "negative_control_intensity_floors.csv"
)

all_puncta_qc_path = (
    OUTPUT_FOLDER
    / "all_puncta_with_control_filter_status.csv"
)

filtering_summary_path = (
    OUTPUT_FOLDER
    / "puncta_filtering_summary.csv"
)

retained_puncta_path = (
    OUTPUT_FOLDER
    / "retained_peptide_puncta.csv"
)

cell_summary_path = (
    OUTPUT_FOLDER
    / "filtered_cell_overlap_summary.csv"
)

peptide_summary_path = (
    OUTPUT_FOLDER
    / "filtered_peptide_summary.csv"
)

plate_summary_path = (
    OUTPUT_FOLDER
    / "filtered_plate_summary.csv"
)

image_summary_path = (
    OUTPUT_FOLDER
    / "filtered_image_summary.csv"
)


control_puncta_df.to_csv(
    control_puncta_path,
    index=False,
)

control_summary.to_csv(
    control_summary_path,
    index=False,
)

plate_floor_df.to_csv(
    plate_floor_path,
    index=False,
)

puncta_df.to_csv(
    all_puncta_qc_path,
    index=False,
)

filtering_summary.to_csv(
    filtering_summary_path,
    index=False,
)

final_puncta_df.to_csv(
    retained_puncta_path,
    index=False,
)

cell_summary.to_csv(
    cell_summary_path,
    index=False,
)

peptide_summary.to_csv(
    peptide_summary_path,
    index=False,
)

plate_summary.to_csv(
    plate_summary_path,
    index=False,
)

image_summary.to_csv(
    image_summary_path,
    index=False,
)


# ============================================================
# PLOT SUPPORT
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

x_positions = np.arange(
    len(peptide_order)
)

# Assign progressively darker shades from the exact palette used by the
# grouped lipoprotein figure.
custom_red_cmap = LinearSegmentedColormap.from_list(
    "grouped_figure_reds",
    RED_PALETTE_ANCHORS,
)

PEPTIDE_COLORS = custom_red_cmap(
    np.linspace(0, 1, len(peptide_order))
)


unique_plates = sorted(
    plate_summary[
        "plate"
    ]
    .astype(str)
    .unique()
)


if len(unique_plates) == 1:

    plate_offsets = {
        unique_plates[0]: 0.0
    }

else:

    plate_offsets = dict(
        zip(
            unique_plates,
            np.linspace(
                -PLATE_OFFSET_WIDTH,
                PLATE_OFFSET_WIDTH,
                len(unique_plates),
            ),
        )
    )


def add_cell_points(axis):
    """
    Add one jittered point for every retained peptide-positive cell.
    """

    for x_position, peptide in zip(
        x_positions,
        peptide_order,
    ):

        current_cells = cell_summary.loc[
            cell_summary["peptide"]
            == peptide
        ]

        jitter = rng.uniform(
            low=-CELL_JITTER_WIDTH,
            high=CELL_JITTER_WIDTH,
            size=len(current_cells),
        )

        axis.scatter(
            x_position + jitter,
            current_cells[
                PRIMARY_OVERLAP_METRIC
            ],
            s=CELL_POINT_SIZE,
            color=CELL_POINT_COLOR,
            alpha=0.38,
            edgecolors="none",
            zorder=3,
        )


def add_plate_points(axis):
    """
    Add one larger dark point per peptide and plate.
    """

    for x_position, peptide in zip(
        x_positions,
        peptide_order,
    ):

        current_plates = plate_summary.loc[
            plate_summary["peptide"]
            == peptide
        ]

        for row in current_plates.itertuples():

            offset = plate_offsets[
                str(row.plate)
            ]

            axis.scatter(
                x_position + offset,
                row.plate_mean_overlap_percent,
                s=PLATE_POINT_SIZE,
                marker="o",
                facecolor=AVERAGE_POINT_COLOR,
                edgecolor="white",
                linewidth=1.0,
                zorder=6,
            )


def format_axis(axis, title):
    """
    Apply common formatting to bar and violin plots.
    """

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        peptide_order,
        rotation=35,
        ha="right",
        rotation_mode="anchor",
        fontsize=10,
        fontweight="bold",
    )

    axis.set_xlabel(
        ""
    )

    axis.set_ylabel(
        Y_LABEL
    )

    axis.set_ylim(
        bottom=0,
        top=100,
    )

    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.6)

    axis.tick_params(
        axis="both",
        width=1.5,
        length=5,
        labelsize=10,
    )

    axis.set_axisbelow(True)

    axis.grid(
        axis="y",
        linestyle="--",
        color="#D9D9D9",
        linewidth=1.0,
        alpha=0.85,
        zorder=0,
    )

    # One sample-size row, styled like the grouped HDL/LDL/VLDL figure.
    axis.text(
        -0.025,
        -0.28,
        "n =",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        clip_on=False,
    )

    for x_position, peptide in zip(x_positions, peptide_order):
        n_cells = int((cell_summary["peptide"] == peptide).sum())
        axis.text(
            x_position,
            -0.28,
            str(n_cells),
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            clip_on=False,
        )

    axis.text(
        0.96,
        0.96,
        f"N = {len(unique_plates)}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=12,
    )


# ============================================================
# BAR PLOT
# ============================================================

bar_means = []


for peptide in peptide_order:

    current_summary = peptide_summary.loc[
        peptide_summary["peptide"]
        == peptide
    ]

    bar_means.append(
        float(
            current_summary[
                "mean_overlap_percent"
            ].iloc[0]
        )
    )

fig, ax = plt.subplots(
    figsize=(
        FIGURE_WIDTH,
        FIGURE_HEIGHT,
    )
)


ax.bar(
    x_positions,
    bar_means,
    width=BAR_WIDTH,
    color=PEPTIDE_COLORS,
    alpha=0.82,
    edgecolor="#222222",
    linewidth=1.8,
    zorder=1,
)

# Black horizontal line = overall cell mean; no SEM or SD bars.
for x_position, mean_value in zip(x_positions, bar_means):
    ax.hlines(
        mean_value,
        x_position - 0.25,
        x_position + 0.25,
        color="black",
        linewidth=2.5,
        zorder=7,
    )


add_cell_points(
    ax
)

add_plate_points(
    ax
)


format_axis(
    ax,
    "Peptide–LDL exact overlap + 2 px proximity after negative-control filtering",
)


fig.subplots_adjust(left=0.08, right=0.99, bottom=0.31, top=0.96)


save_figure(
    fig,
    "negative_control_filtered_exact_plus_2px_barplot",
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
            cell_summary["peptide"]
            == peptide,
            PRIMARY_OVERLAP_METRIC,
        ]
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    if len(values) == 0:
        raise RuntimeError(
            f"No values were found for peptide {peptide}."
        )

    # Matplotlib violinplot needs at least two nonidentical
    # values to estimate a density.
    if len(values) == 1:

        value = values[0]

        density_values = np.array(
            [
                value - 1e-6,
                value + 1e-6,
            ]
        )

    elif np.std(values) == 0:

        density_values = (
            values
            + np.linspace(
                -1e-6,
                1e-6,
                len(values),
            )
        )

    else:

        density_values = values

    violin_values.append(
        density_values
    )


fig, ax = plt.subplots(
    figsize=(
        FIGURE_WIDTH,
        FIGURE_HEIGHT,
    )
)


violin_parts = ax.violinplot(
    violin_values,
    positions=x_positions,
    widths=VIOLIN_WIDTH,
    showmeans=False,
    showmedians=False,
    showextrema=False,
)


for body, peptide_color in zip(
    violin_parts["bodies"],
    PEPTIDE_COLORS,
):

    body.set_facecolor(
        peptide_color
    )

    body.set_alpha(
        0.82
    )

    body.set_edgecolor(
        "#222222"
    )

    body.set_linewidth(
        1.8
    )


# Add a horizontal mean line to each violin.
for x_position, peptide in zip(
    x_positions,
    peptide_order,
):

    mean_value = float(
        cell_summary.loc[
            cell_summary["peptide"]
            == peptide,
            PRIMARY_OVERLAP_METRIC,
        ].mean()
    )

    ax.hlines(
        y=mean_value,
        xmin=x_position - 0.20,
        xmax=x_position + 0.20,
        color="black",
        linewidth=2.5,
        zorder=7,
    )


add_cell_points(
    ax
)

add_plate_points(
    ax
)


format_axis(
    ax,
    "Distribution of peptide–LDL exact overlap + 2 px proximity after control filtering",
)


fig.subplots_adjust(left=0.08, right=0.99, bottom=0.31, top=0.96)


save_figure(
    fig,
    "negative_control_filtered_exact_plus_2px_violinplot",
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
print("NEGATIVE-CONTROL FILTERING COMPLETE")
print("=" * 70)

print()
print(
    f"Control-floor mode: "
    f"{CONTROL_FLOOR_MODE}"
)

print(
    f"Global fallback floor: "
    f"{global_control_floor:.3f}"
)

print()
print("Plate-specific negative-control floors:")

print(
    plate_floor_df[
        [
            "plate",
            "control_intensity_floor",
            "floor_source",
            "controls_present",
            "n_control_puncta",
        ]
    ].to_string(
        index=False
    )
)

print()
print("Retained peptide-positive cells:")

print(
    cell_summary
    .groupby(
        "peptide",
        observed=True,
    )
    .size()
    .rename(
        "n_cells"
    )
)

print()
print("Saved CSV files:")

for csv_path in sorted(
    OUTPUT_FOLDER.glob("*.csv")
):
    print(
        f"  {csv_path.name}"
    )

print()
print(
    f"All outputs saved to: "
    f"{OUTPUT_FOLDER}"
)