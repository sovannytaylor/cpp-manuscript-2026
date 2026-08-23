"""
Plot Naegleria morphology features grouped by peptide and timepoint.

Replicate definition:
    - each image = one replicate
    - dark dots = image-level averages
    - PR39 can have 9 image reps per condition
    - LL37/BMAP can have 3 image reps per condition

PI positivity:
    - threshold is calculated per plate/timepoint from:
        NORM media + 0 concentration
    - threshold = mean(NORM 0) + 8 * SD(NORM 0)

Outputs:
    - cell-level plots with image-rep mean dots
    - PI-positive barplots using image reps
    - replicate_summary_by_image.csv
"""

import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger

logger.info("import ok")

# -------------------------
# configuration
# -------------------------

input_folder = "concat_calculations/"
output_folder = "concat_calculations/morphology_plots/greater-n/"

os.makedirs(output_folder, exist_ok=True)

input_csv = os.path.join(input_folder, "cell_morphology_features.csv")

plt.rcParams.update({
    "font.size": 14,
    "svg.fonttype": "none"
})

sns.set_palette("Paired")

PI_INTENSITY_FEATURE = "pi_cell_intensity_mean"
PI_SD_MULTIPLIER = 8

CELL_LEVEL_FEATURES_TO_PLOT = [
    "area_um2",
    "perimeter_um",
    "circularity",
    "aspect_ratio",
    "eccentricity",
    "solidity",
    "pi_cell_intensity_mean",
    "pi_cell_intensity_median",
    "pi_fov_intensity_mean",
]

PI_BAR_FEATURES_TO_PLOT = [
    "pi_positive_percent",
    "percent_pi_positive_in_fov_recalc",
]

FEATURES_TO_SUMMARIZE = (
    CELL_LEVEL_FEATURES_TO_PLOT
    + [
        "pi_positive",
        "n_pi_positive_in_fov_recalc",
        "n_cells_in_fov_recalc",
        "pi_positive_percent",
        "percent_pi_positive_in_fov_recalc",
    ]
)

PEPTIDE_ORDER = None
MEDIA_ORDER = ["NORM", "LPD"]
TIMEPOINT_ORDER = ["4HR", "12HR"]

FILTER_OUTLIERS = True
IQR_MULTIPLIER = 1.5
MIN_CELL_LENGTH_UM = 10
SCALE_PX = 0.693


# -------------------------
# metadata parsing
# -------------------------

WELL_METADATA_26037B = {
    "B02": ("NORM", 0),  "C02": ("NORM", 0),  "D02": ("NORM", 0),
    "E02": ("LPD", 0),   "F02": ("LPD", 0),   "G02": ("LPD", 0),
    "B03": ("NORM", 4),  "C03": ("NORM", 4),  "D03": ("NORM", 4),
    "E03": ("LPD", 4),   "F03": ("LPD", 4),   "G03": ("LPD", 4),
    "B04": ("NORM", 8),  "C04": ("NORM", 8),  "D04": ("NORM", 8),
    "E04": ("LPD", 8),   "F04": ("LPD", 8),   "G04": ("LPD", 8),
    "B05": ("NORM", 16), "C05": ("NORM", 16), "D05": ("NORM", 16),
    "E05": ("LPD", 16),  "F05": ("LPD", 16),  "G05": ("LPD", 16),
    "B06": ("NORM", 32), "C06": ("NORM", 32), "D06": ("NORM", 32),
    "E06": ("LPD", 32),  "F06": ("LPD", 32),  "G06": ("LPD", 32),
}

WELL_METADATA_26037C = {
    "B02": ("NORM", 0),   "C02": ("NORM", 0),   "D02": ("NORM", 0),
    "B03": ("NORM", 4),   "C03": ("NORM", 4),   "D03": ("NORM", 4),
    "B04": ("NORM", 8),   "C04": ("NORM", 8),   "D04": ("NORM", 8),
    "B05": ("NORM", 16),  "C05": ("NORM", 16),  "D05": ("NORM", 16),
    "B06": ("NORM", 32),  "C06": ("NORM", 32),  "D06": ("NORM", 32),
    "B07": ("LPD", 0),    "C07": ("LPD", 0),    "D07": ("LPD", 0),
    "B08": ("LPD", 4),    "C08": ("LPD", 4),    "D08": ("LPD", 4),
    "B09": ("LPD", 8),    "C09": ("LPD", 8),    "D09": ("LPD", 8),
    "B10": ("LPD", 16),   "C10": ("LPD", 16),   "D10": ("LPD", 16),
    "B11": ("LPD", 32),   "C11": ("LPD", 32),   "D11": ("LPD", 32),
}


def parse_filename_metadata(image_name):
    base = os.path.basename(str(image_name))
    base = os.path.splitext(base)[0]

    pattern1 = (
        r"(?P<plate>NAEG)-(?P<timepoint>\d+HR)_"
        r"(?P<media>NORM|LPD)_"
        r"(?P<peptide>.+?)_"
        r"(?P<concentration>\d+)_"
        r"(?P<rep>REP\d+)"
    )

    match1 = re.match(pattern1, base)

    if match1:
        d = match1.groupdict()
        return pd.Series({
            "plate": d["plate"],
            "timepoint": d["timepoint"],
            "media": d["media"],
            "peptide": d["peptide"],
            "concentration": int(d["concentration"]),
            "rep": d["rep"],
            "well": "not_applicable",
        })

    parts = base.split("_")

    if len(parts) < 2:
        return pd.Series({
            "plate": "unknown",
            "timepoint": "unknown",
            "media": "unknown",
            "peptide": "unknown",
            "concentration": np.nan,
            "rep": "unknown",
            "well": "unknown",
        })

    plate = parts[0]
    info = parts[1].split("-")

    if len(info) < 5:
        return pd.Series({
            "plate": plate,
            "timepoint": "unknown",
            "media": "unknown",
            "peptide": "unknown",
            "concentration": np.nan,
            "rep": "unknown",
            "well": "unknown",
        })

    peptide = info[1]
    timepoint = info[2]
    rep = info[3]
    well = info[4]

    if plate == "26037B":
        well_metadata = WELL_METADATA_26037B
    elif plate == "26037C":
        well_metadata = WELL_METADATA_26037C
    else:
        well_metadata = {}

    if well in well_metadata:
        media, concentration = well_metadata[well]
    else:
        logger.warning(f"Well not found for plate {plate}: {well}")
        media = "unknown"
        concentration = np.nan

    return pd.Series({
        "plate": plate,
        "timepoint": timepoint,
        "media": media,
        "peptide": peptide,
        "concentration": concentration,
        "rep": rep,
        "well": well,
    })


def add_metadata(df):
    metadata = df["image_name"].apply(parse_filename_metadata)
    df = pd.concat([df, metadata], axis=1)

    df["concentration"] = pd.to_numeric(df["concentration"], errors="coerce")

    df["image_rep"] = df["image_name"].astype(str)

    df["condition"] = (
        df["timepoint"].astype(str)
        + "_"
        + df["media"].astype(str)
        + "_"
        + df["peptide"].astype(str)
        + "_"
        + df["concentration"].astype(str)
    )

    return df


# -------------------------
# PI thresholding
# -------------------------

def add_pi_threshold_from_norm0(df):
    if PI_INTENSITY_FEATURE not in df.columns:
        logger.warning(f"{PI_INTENSITY_FEATURE} not found. Skipping PI recalculation.")
        return df

    df = df.copy()

    threshold_group_cols = ["plate", "timepoint"]

    negative_control_df = df[
        (df["media"] == "NORM")
        & (df["concentration"] == 0)
        & (df[PI_INTENSITY_FEATURE].notna())
    ].copy()

    pi_thresholds = (
        negative_control_df
        .groupby(threshold_group_cols)[PI_INTENSITY_FEATURE]
        .agg(
            norm0_mean="mean",
            norm0_std="std",
            norm0_median="median",
            norm0_p95=lambda x: np.percentile(x.dropna(), 95),
            norm0_p99=lambda x: np.percentile(x.dropna(), 99),
            norm0_p995=lambda x: np.percentile(x.dropna(), 99.5),
        )
        .reset_index()
    )

    pi_thresholds["pi_threshold_from_norm0"] = (
        pi_thresholds["norm0_mean"]
        + PI_SD_MULTIPLIER * pi_thresholds["norm0_std"]
    )

    df = df.merge(pi_thresholds, on=threshold_group_cols, how="left")

    df["pi_positive"] = (
        df[PI_INTENSITY_FEATURE] > df["pi_threshold_from_norm0"]
    )

    df.loc[df["pi_threshold_from_norm0"].isna(), "pi_positive"] = np.nan

    image_pi_summary = (
        df
        .groupby("image_name", as_index=False)
        .agg(
            n_cells_in_fov_recalc=("pi_positive", "count"),
            n_pi_positive_in_fov_recalc=("pi_positive", "sum"),
            pi_threshold_from_norm0_image=("pi_threshold_from_norm0", "first"),
        )
    )

    image_pi_summary["percent_pi_positive_in_fov_recalc"] = (
        image_pi_summary["n_pi_positive_in_fov_recalc"]
        / image_pi_summary["n_cells_in_fov_recalc"]
        * 100
    )

    df = df.merge(
        image_pi_summary[
            [
                "image_name",
                "n_cells_in_fov_recalc",
                "n_pi_positive_in_fov_recalc",
                "percent_pi_positive_in_fov_recalc",
            ]
        ],
        on="image_name",
        how="left"
    )

    df["pi_positive_percent"] = df["pi_positive"] * 100

    pi_thresholds.to_csv(
        os.path.join(output_folder, "pi_thresholds_from_norm0_mean_plus_8sd.csv"),
        index=False
    )

    return df


# -------------------------
# filtering
# -------------------------

def filter_small_masks(df, min_length_um=MIN_CELL_LENGTH_UM):
    if "major_axis_length" not in df.columns:
        logger.warning("major_axis_length column not found. Skipping small mask filtering.")
        return df

    if "major_axis_length_um" not in df.columns:
        df["major_axis_length_um"] = df["major_axis_length"] * SCALE_PX

    before = len(df)
    df = df[df["major_axis_length_um"] >= min_length_um].copy()
    after = len(df)

    logger.info(f"Removed {before - after} masks smaller than {min_length_um} um.")
    logger.info(f"Kept {after} masks.")

    return df


def filter_outliers_by_group(df, features, group_cols, iqr_multiplier=1.5):
    filtered = df.copy()

    skip_features = [
        "pi_positive",
        "pi_positive_percent",
        "percent_pi_positive_in_fov_recalc",
        "n_pi_positive_in_fov_recalc",
        "n_cells_in_fov_recalc",
        "pi_threshold_from_norm0",
    ]

    for feature in features:
        if feature not in filtered.columns:
            continue

        if feature in skip_features:
            continue

        keep_mask = pd.Series(True, index=filtered.index)

        for _, subdf in filtered.groupby(group_cols):
            values = subdf[feature].dropna()

            if len(values) < 5:
                continue

            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1

            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr

            idx = subdf.index
            keep_mask.loc[idx] = (
                filtered.loc[idx, feature].between(lower, upper)
                | filtered.loc[idx, feature].isna()
            )

        filtered = filtered[keep_mask].copy()

    return filtered


# -------------------------
# replicate summary
# -------------------------

def make_image_replicate_summary(df):
    group_cols = [
        "image_name",
        "plate",
        "rep",
        "well",
        "timepoint",
        "media",
        "peptide",
        "concentration",
    ]

    agg_dict = {}

    for feature in CELL_LEVEL_FEATURES_TO_PLOT:
        if feature in df.columns:
            agg_dict[f"{feature}_mean"] = (feature, "mean")
            agg_dict[f"{feature}_median"] = (feature, "median")
            agg_dict[f"{feature}_std"] = (feature, "std")
            agg_dict[f"{feature}_n_cells"] = (feature, "count")

    if "pi_positive" in df.columns:
        agg_dict["pi_positive_fraction"] = ("pi_positive", "mean")
        agg_dict["pi_positive_percent"] = ("pi_positive", lambda x: x.mean() * 100)
        agg_dict["n_pi_positive"] = ("pi_positive", "sum")
        agg_dict["n_cells"] = ("pi_positive", "count")

    rep_summary = (
        df
        .groupby(group_cols, as_index=False)
        .agg(**agg_dict)
    )

    return rep_summary


# -------------------------
# helpers
# -------------------------

def get_concentration_order(df):
    return sorted(df["concentration"].dropna().unique())


def get_timepoints(df):
    timepoints = sorted(df["timepoint"].dropna().unique())
    return [t for t in TIMEPOINT_ORDER if t in timepoints]


def get_peptides(df):
    if PEPTIDE_ORDER is None:
        return sorted(df["peptide"].dropna().unique())
    return [p for p in PEPTIDE_ORDER if p in df["peptide"].dropna().unique()]


def add_n_labels(ax, plot_df, feature, concentration_order):
    y_min = plot_df[feature].min()
    y_max = plot_df[feature].max()
    y_range = y_max - y_min if y_max > y_min else 1
    label_y = y_min - 0.15 * y_range

    for i, conc in enumerate(concentration_order):
        for media in MEDIA_ORDER:
            sub = plot_df[
                (plot_df["concentration"] == conc)
                & (plot_df["media"] == media)
            ]

            if sub.empty:
                continue

            n_cells = len(sub)
            n_reps = sub["image_name"].nunique()

            x_offset = -0.2 if media == "NORM" else 0.2

            ax.text(
                i + x_offset,
                label_y,
                f"n={n_cells}\nN={n_reps}",
                ha="center",
                va="top",
                fontsize=8
            )

    ax.set_ylim(label_y - 0.08 * y_range, y_max + 0.08 * y_range)


# -------------------------
# plotting
# -------------------------

def plot_cells_and_image_rep_means(df, rep_summary, features):
    plot_output_folder = os.path.join(output_folder, "cell_and_image_rep_mean_plots")
    os.makedirs(plot_output_folder, exist_ok=True)

    for peptide in get_peptides(df):
        for timepoint in get_timepoints(df):

            peptide_df = df[
                (df["peptide"] == peptide)
                & (df["timepoint"] == timepoint)
            ].copy()

            peptide_rep_df = rep_summary[
                (rep_summary["peptide"] == peptide)
                & (rep_summary["timepoint"] == timepoint)
            ].copy()

            if peptide_df.empty or peptide_rep_df.empty:
                continue

            concentration_order = get_concentration_order(peptide_df)

            for feature in features:
                if feature not in peptide_df.columns:
                    continue

                rep_feature = f"{feature}_mean"
                if rep_feature not in peptide_rep_df.columns:
                    continue

                plot_df = peptide_df.dropna(
                    subset=[feature, "concentration", "media", "image_name"]
                ).copy()

                rep_plot_df = peptide_rep_df.dropna(
                    subset=[rep_feature, "concentration", "media", "image_name"]
                ).copy()

                if plot_df.empty or rep_plot_df.empty:
                    continue

                plt.figure(figsize=(8, 5))

                ax = sns.boxplot(
                    data=plot_df,
                    x="concentration",
                    y=feature,
                    hue="media",
                    order=concentration_order,
                    hue_order=MEDIA_ORDER,
                    showfliers=False,
                    zorder=1,
                    boxprops={"alpha": 0.35},
                )

                sns.stripplot(
                    data=plot_df,
                    x="concentration",
                    y=feature,
                    hue="media",
                    order=concentration_order,
                    hue_order=MEDIA_ORDER,
                    dodge=True,
                    edgecolor="none",
                    linewidth=0,
                    size=3,
                    alpha=0.18,
                    jitter=True,
                    zorder=2,
                    ax=ax,
                )

                sns.stripplot(
                    data=rep_plot_df,
                    x="concentration",
                    y=rep_feature,
                    hue="media",
                    order=concentration_order,
                    hue_order=MEDIA_ORDER,
                    dodge=True,
                    edgecolor="black",
                    linewidth=0.7,
                    size=8,
                    alpha=0.95,
                    jitter=0.08,
                    zorder=3,
                    ax=ax,
                )

                handles, labels = ax.get_legend_handles_labels()
                ax.legend(
                    handles[:len(MEDIA_ORDER)],
                    labels[:len(MEDIA_ORDER)],
                    title="media",
                    frameon=False
                )

                add_n_labels(ax, plot_df, feature, concentration_order)

                sns.despine()
                plt.title(f"{peptide} | {timepoint} | {feature}")
                plt.xlabel("Concentration")
                plt.ylabel(feature)
                plt.tight_layout()

                save_base = f"{peptide}_{timepoint}_{feature}_cells_imageRepMeans"

                plt.savefig(
                    os.path.join(plot_output_folder, f"{save_base}.png"),
                    dpi=300,
                    bbox_inches="tight"
                )

                plt.savefig(
                    os.path.join(plot_output_folder, f"{save_base}.svg"),
                    bbox_inches="tight"
                )

                plt.close()


def plot_pi_barplots(rep_summary):
    plot_output_folder = os.path.join(output_folder, "pi_barplots_image_reps")
    os.makedirs(plot_output_folder, exist_ok=True)

    for peptide in get_peptides(rep_summary):
        for timepoint in get_timepoints(rep_summary):

            plot_df = rep_summary[
                (rep_summary["peptide"] == peptide)
                & (rep_summary["timepoint"] == timepoint)
            ].copy()

            if plot_df.empty:
                continue

            concentration_order = get_concentration_order(plot_df)

            for feature in PI_BAR_FEATURES_TO_PLOT:

                if feature not in plot_df.columns:
                    continue

                use_df = plot_df.dropna(
                    subset=[feature, "concentration", "media", "image_name"]
                ).copy()

                if use_df.empty:
                    continue

                plt.figure(figsize=(8, 5))

                ax = sns.barplot(
                    data=use_df,
                    x="concentration",
                    y=feature,
                    hue="media",
                    order=concentration_order,
                    hue_order=MEDIA_ORDER,
                    errorbar="se",
                    capsize=0.15,
                    alpha=0.55,
                    zorder=1,
                )

                sns.stripplot(
                    data=use_df,
                    x="concentration",
                    y=feature,
                    hue="media",
                    order=concentration_order,
                    hue_order=MEDIA_ORDER,
                    dodge=True,
                    edgecolor="black",
                    linewidth=0.7,
                    size=8,
                    alpha=0.95,
                    jitter=0.08,
                    zorder=2,
                    ax=ax,
                )

                handles, labels = ax.get_legend_handles_labels()
                ax.legend(
                    handles[:len(MEDIA_ORDER)],
                    labels[:len(MEDIA_ORDER)],
                    title="media",
                    frameon=False
                )

                ax.set_ylim(0, 105)

                sns.despine()
                plt.title(f"{peptide} | {timepoint} | {feature}")
                plt.xlabel("Concentration")
                plt.ylabel(feature)
                plt.tight_layout()

                save_base = f"{peptide}_{timepoint}_{feature}_barplot_imageReps"

                plt.savefig(
                    os.path.join(plot_output_folder, f"{save_base}.png"),
                    dpi=300,
                    bbox_inches="tight"
                )

                plt.savefig(
                    os.path.join(plot_output_folder, f"{save_base}.svg"),
                    bbox_inches="tight"
                )

                plt.close()


# -------------------------
# main
# -------------------------

if __name__ == "__main__":

    logger.info("Loading morphology features...")
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df)} rows.")

    df = add_metadata(df)

    df = df[
        (df["peptide"] != "unknown")
        & (df["media"] != "unknown")
        & (~df["concentration"].isna())
        & (df["timepoint"] != "unknown")
    ].copy()

    df = add_pi_threshold_from_norm0(df)

    df.to_csv(
        os.path.join(
            output_folder,
            "cell_morphology_features_indexed_with_recalculated_PI.csv"
        ),
        index=False
    )

    df = filter_small_masks(df, min_length_um=MIN_CELL_LENGTH_UM)

    if FILTER_OUTLIERS:
        logger.info("Filtering outliers...")
        plot_df = filter_outliers_by_group(
            df,
            CELL_LEVEL_FEATURES_TO_PLOT,
            group_cols=[
                "peptide",
                "timepoint",
                "media",
                "concentration",
            ],
            iqr_multiplier=IQR_MULTIPLIER
        )
    else:
        plot_df = df.copy()

    plot_df.to_csv(
        os.path.join(output_folder, "cell_morphology_features_indexed_filtered.csv"),
        index=False
    )

    logger.info("Creating image-level replicate summary...")

    rep_summary = make_image_replicate_summary(plot_df)

    rep_summary.to_csv(
        os.path.join(output_folder, "replicate_summary_by_image.csv"),
        index=False
    )

    logger.info(f"Saved image replicate summary: {len(rep_summary)} rows")

    logger.info("Making cell + image replicate mean plots...")
    plot_cells_and_image_rep_means(
        plot_df,
        rep_summary,
        CELL_LEVEL_FEATURES_TO_PLOT
    )

    logger.info("Making PI barplots...")
    plot_pi_barplots(rep_summary)

    logger.info("done.")