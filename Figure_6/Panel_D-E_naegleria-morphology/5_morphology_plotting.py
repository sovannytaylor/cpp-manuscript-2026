"""
Plot Naegleria morphology features grouped by timepoint.

Outputs:
    - plots split by peptide AND timepoint
"""

import os
import math
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
output_folder = "concat_calculations/morphology_plots/"
fov_output_folder = os.path.join(output_folder, "fov_average")

os.makedirs(output_folder, exist_ok=True)
os.makedirs(fov_output_folder, exist_ok=True)

input_csv = os.path.join(input_folder, "cell_morphology_features.csv")

plt.rcParams.update({
    "font.size": 14,
    "svg.fonttype": "none"
})

sns.set_palette("Paired")

FEATURES_TO_PLOT = [
    "area_um2",
    "perimeter_um",
    "circularity",
    "aspect_ratio",
    "eccentricity",
    "solidity",
    "convexity",
    "fractal_dimension",

    # PI features
    "pi_cell_intensity_mean",
    "pi_cell_intensity_median",
    "pi_fov_intensity_mean",
    "percent_pi_positive_in_fov",
    "n_pi_positive_in_fov",
    "n_cells_in_fov",
]

PEPTIDE_ORDER = ["PR39"]
MEDIA_ORDER = ["NORM", "LPD"]

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
    # NORM
    "B02": ("NORM", 0),   "C02": ("NORM", 0),   "D02": ("NORM", 0),
    "B03": ("NORM", 4),   "C03": ("NORM", 4),   "D03": ("NORM", 4),
    "B04": ("NORM", 8),   "C04": ("NORM", 8),   "D04": ("NORM", 8),
    "B05": ("NORM", 16),  "C05": ("NORM", 16),  "D05": ("NORM", 16),
    "B06": ("NORM", 32),  "C06": ("NORM", 32),  "D06": ("NORM", 32),

    # LPD
    "B07": ("LPD", 0),    "C07": ("LPD", 0),    "D07": ("LPD", 0),
    "B08": ("LPD", 4),    "C08": ("LPD", 4),    "D08": ("LPD", 4),
    "B09": ("LPD", 8),    "C09": ("LPD", 8),    "D09": ("LPD", 8),
    "B10": ("LPD", 16),   "C10": ("LPD", 16),   "D10": ("LPD", 16),
    "B11": ("LPD", 32),   "C11": ("LPD", 32),   "D11": ("LPD", 32),
}


def parse_filename_metadata(image_name):
    """
    Expected filename:
        26037B_NAEG-PR39-4HR-01-B04
    """

    base = os.path.basename(str(image_name))
    base = os.path.splitext(base)[0]

    parts = base.split("_")

    if len(parts) < 2:
        logger.warning(f"Filename does not match expected format: {image_name}")
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
        logger.warning(f"Filename does not match expected format: {image_name}")
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
        logger.warning(f"Plate not recognized: {plate}")
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

    df["concentration"] = pd.to_numeric(
        df["concentration"],
        errors="coerce"
    )

    df["condition"] = (
        df["plate"].astype(str)
        + "_"
        + df["timepoint"].astype(str)
        + "_"
        + df["media"].astype(str)
        + "_"
        + df["peptide"].astype(str)
        + "_"
        + df["concentration"].astype(str)
    )

    return df


# -------------------------
# filtering
# -------------------------

def filter_small_masks(df, min_length_um=MIN_CELL_LENGTH_UM):
    if "major_axis_length" not in df.columns:
        raise ValueError("major_axis_length column not found.")

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

    for feature in features:
        if feature not in filtered.columns:
            logger.warning(f"Skipping outlier filtering for missing feature: {feature}")
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


def make_fov_average_df(df, features):
    fov_level_features = [
        "pi_fov_intensity_mean",
        "pi_fov_intensity_median",
        "pi_fov_intensity_sum",
        "n_cells_in_fov",
        "n_pi_positive_in_fov",
        "percent_pi_positive_in_fov",
        "pi_positive_threshold",
    ]

    morphology_features = [
        f for f in features
        if f in df.columns and f not in fov_level_features
    ]

    agg_dict = {f: "mean" for f in morphology_features}

    for f in fov_level_features:
        if f in df.columns:
            agg_dict[f] = "first"

    group_cols = [
        "image_name",
        "plate",
        "timepoint",
        "media",
        "peptide",
        "concentration",
        "rep",
        "well",
        "condition",
    ]

    return df.groupby(group_cols, as_index=False).agg(agg_dict)


# -------------------------
# helpers
# -------------------------

def get_concentration_order(df):
    return sorted(df["concentration"].dropna().unique())


def get_timepoints(df):
    return sorted(df["timepoint"].dropna().unique())


# -------------------------
# plotting
# -------------------------

def plot_stripplots_by_peptide(df, features):
    for peptide in PEPTIDE_ORDER:
        for timepoint in get_timepoints(df):

            peptide_df = df[
                (df["peptide"] == peptide)
                & (df["timepoint"] == timepoint)
            ].copy()

            if peptide_df.empty:
                continue

            concentration_order = get_concentration_order(peptide_df)

            for feature in features:
                if feature not in peptide_df.columns:
                    continue

                plot_df = peptide_df.dropna(
                    subset=[feature, "concentration", "media"]
                ).copy()

                if plot_df.empty:
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
                )

                sns.stripplot(
                    data=plot_df,
                    x="concentration",
                    y=feature,
                    hue="media",
                    order=concentration_order,
                    hue_order=MEDIA_ORDER,
                    dodge=True,
                    edgecolor="black",
                    linewidth=0.3,
                    size=4,
                    alpha=0.45,
                    jitter=True,
                    zorder=2,
                    ax=ax,
                )

                handles, labels = ax.get_legend_handles_labels()
                ax.legend(
                    handles[:len(MEDIA_ORDER)],
                    labels[:len(MEDIA_ORDER)],
                    title="media"
                )

                sns.despine()
                plt.title(f"{peptide} | {timepoint} | {feature}")
                plt.xlabel("Concentration")
                plt.ylabel(feature)
                plt.tight_layout()

                save_base = f"{peptide}_{timepoint}_{feature}_strip_box"
                plt.savefig(os.path.join(output_folder, f"{save_base}.png"), dpi=300, bbox_inches="tight")
                plt.savefig(os.path.join(output_folder, f"{save_base}.svg"), bbox_inches="tight")
                plt.close()


def plot_histograms_by_peptide(df, features):
    for peptide in PEPTIDE_ORDER:
        for timepoint in get_timepoints(df):

            peptide_df = df[
                (df["peptide"] == peptide)
                & (df["timepoint"] == timepoint)
            ].copy()

            if peptide_df.empty:
                continue

            concentration_order = get_concentration_order(peptide_df)

            for feature in features:
                if feature not in peptide_df.columns:
                    continue

                feature_df = peptide_df.dropna(
                    subset=[feature, "media", "concentration"]
                ).copy()

                if feature_df.empty:
                    continue

                n_concs = len(concentration_order)
                ncols = 3
                nrows = math.ceil(n_concs / ncols)

                fig, axes = plt.subplots(
                    nrows=nrows,
                    ncols=ncols,
                    figsize=(5 * ncols, 4 * nrows),
                    sharex=False,
                    sharey=False
                )

                axes = np.array(axes).flatten()

                for i, conc in enumerate(concentration_order):
                    ax = axes[i]
                    conc_df = feature_df[feature_df["concentration"] == conc].copy()

                    if conc_df.empty:
                        ax.axis("off")
                        continue

                    sns.histplot(
                        data=conc_df,
                        x=feature,
                        hue="media",
                        hue_order=MEDIA_ORDER,
                        element="step",
                        stat="density",
                        common_norm=False,
                        bins=25,
                        alpha=0.35,
                        ax=ax
                    )

                    ax.set_title(f"{conc:g}")
                    ax.set_xlabel(feature)
                    ax.set_ylabel("Density")
                    sns.despine(ax=ax)

                for ax in axes[n_concs:]:
                    ax.axis("off")

                fig.suptitle(
                    f"{peptide} | {timepoint} | {feature} distribution by concentration",
                    y=1.02
                )

                fig.tight_layout()

                save_base = f"{peptide}_{timepoint}_{feature}_histograms_by_concentration"
                fig.savefig(os.path.join(output_folder, f"{save_base}.png"), dpi=300, bbox_inches="tight")
                fig.savefig(os.path.join(output_folder, f"{save_base}.svg"), bbox_inches="tight")
                plt.close(fig)


def plot_fov_barcharts_by_peptide(fov_df, features):
    bar_output_folder = os.path.join(fov_output_folder, "barplots")
    os.makedirs(bar_output_folder, exist_ok=True)

    for peptide in PEPTIDE_ORDER:
        for timepoint in get_timepoints(fov_df):

            peptide_df = fov_df[
                (fov_df["peptide"] == peptide)
                & (fov_df["timepoint"] == timepoint)
            ].copy()

            if peptide_df.empty:
                continue

            concentration_order = get_concentration_order(peptide_df)

            for feature in features:
                if feature not in peptide_df.columns:
                    continue

                plot_df = peptide_df.dropna(
                    subset=[feature, "concentration", "media"]
                ).copy()

                if plot_df.empty:
                    continue

                plt.figure(figsize=(8, 5))

                ax = sns.barplot(
                    data=plot_df,
                    x="concentration",
                    y=feature,
                    hue="media",
                    order=concentration_order,
                    hue_order=MEDIA_ORDER,
                    errorbar="se",
                    capsize=0.2,
                    err_kws={"linewidth": 1.5},
                    zorder=1,
                )

                sns.stripplot(
                    data=plot_df,
                    x="concentration",
                    y=feature,
                    hue="media",
                    order=concentration_order,
                    hue_order=MEDIA_ORDER,
                    dodge=True,
                    edgecolor="black",
                    linewidth=0.4,
                    size=5,
                    alpha=0.8,
                    jitter=True,
                    zorder=2,
                    ax=ax,
                )

                handles, labels = ax.get_legend_handles_labels()
                ax.legend(
                    handles[:len(MEDIA_ORDER)],
                    labels[:len(MEDIA_ORDER)],
                    title="media"
                )

                sns.despine()
                plt.title(f"{peptide} | {timepoint} | FOV average | {feature}")
                plt.xlabel("Concentration")
                plt.ylabel(f"FOV average {feature}")
                plt.tight_layout()

                save_base = f"{peptide}_{timepoint}_{feature}_FOV_average_barplot"
                plt.savefig(os.path.join(bar_output_folder, f"{save_base}.png"), dpi=300, bbox_inches="tight")
                plt.savefig(os.path.join(bar_output_folder, f"{save_base}.svg"), bbox_inches="tight")
                plt.close()


def plot_fov_barchart_grid_by_peptide(fov_df, features):
    bar_output_folder = os.path.join(fov_output_folder, "barplots")
    os.makedirs(bar_output_folder, exist_ok=True)

    valid_features = [f for f in features if f in fov_df.columns]

    for peptide in PEPTIDE_ORDER:
        for timepoint in get_timepoints(fov_df):

            peptide_df = fov_df[
                (fov_df["peptide"] == peptide)
                & (fov_df["timepoint"] == timepoint)
            ].copy()

            if peptide_df.empty:
                continue

            concentration_order = get_concentration_order(peptide_df)

            n_features = len(valid_features)
            ncols = 3
            nrows = math.ceil(n_features / ncols)

            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(5 * ncols, 4.5 * nrows)
            )

            axes = np.array(axes).flatten()

            for i, feature in enumerate(valid_features):
                ax = axes[i]

                plot_df = peptide_df.dropna(
                    subset=[feature, "concentration", "media"]
                ).copy()

                if plot_df.empty:
                    ax.axis("off")
                    continue

                sns.barplot(
                    data=plot_df,
                    x="concentration",
                    y=feature,
                    hue="media",
                    order=concentration_order,
                    hue_order=MEDIA_ORDER,
                    errorbar="se",
                    capsize=0.2,
                    err_kws={"linewidth": 1.2},
                    ax=ax,
                    zorder=1,
                )

                sns.stripplot(
                    data=plot_df,
                    x="concentration",
                    y=feature,
                    hue="media",
                    order=concentration_order,
                    hue_order=MEDIA_ORDER,
                    dodge=True,
                    edgecolor="black",
                    linewidth=0.3,
                    size=4,
                    alpha=0.8,
                    jitter=True,
                    ax=ax,
                    zorder=2,
                )

                handles, labels = ax.get_legend_handles_labels()
                ax.legend(
                    handles[:len(MEDIA_ORDER)],
                    labels[:len(MEDIA_ORDER)],
                    title="media"
                )

                ax.set_title(feature)
                ax.set_xlabel("Concentration")
                ax.set_ylabel("")
                ax.tick_params(axis="x", rotation=35)
                sns.despine(ax=ax)

            for ax in axes[n_features:]:
                ax.axis("off")

            fig.suptitle(
                f"{peptide} | {timepoint} morphology features | FOV average barplots",
                y=1.02
            )

            fig.tight_layout()

            save_base = f"{peptide}_{timepoint}_all_features_FOV_average_barplot_grid"
            fig.savefig(os.path.join(bar_output_folder, f"{save_base}.png"), dpi=300, bbox_inches="tight")
            fig.savefig(os.path.join(bar_output_folder, f"{save_base}.svg"), bbox_inches="tight")
            plt.close(fig)


def plot_fov_stripplots_by_peptide(fov_df, features):
    for peptide in PEPTIDE_ORDER:
        for timepoint in get_timepoints(fov_df):

            peptide_df = fov_df[
                (fov_df["peptide"] == peptide)
                & (fov_df["timepoint"] == timepoint)
            ].copy()

            if peptide_df.empty:
                continue

            concentration_order = get_concentration_order(peptide_df)

            for feature in features:
                if feature not in peptide_df.columns:
                    continue

                plot_df = peptide_df.dropna(
                    subset=[feature, "concentration", "media"]
                ).copy()

                if plot_df.empty:
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
                )

                sns.stripplot(
                    data=plot_df,
                    x="concentration",
                    y=feature,
                    hue="media",
                    order=concentration_order,
                    hue_order=MEDIA_ORDER,
                    dodge=True,
                    edgecolor="black",
                    linewidth=0.5,
                    size=7,
                    alpha=0.8,
                    jitter=True,
                    zorder=2,
                    ax=ax,
                )

                handles, labels = ax.get_legend_handles_labels()
                ax.legend(
                    handles[:len(MEDIA_ORDER)],
                    labels[:len(MEDIA_ORDER)],
                    title="media"
                )

                sns.despine()
                plt.title(f"{peptide} | {timepoint} | FOV average | {feature}")
                plt.xlabel("Concentration")
                plt.ylabel(f"FOV average {feature}")
                plt.tight_layout()

                save_base = f"{peptide}_{timepoint}_{feature}_FOV_average_strip_box"
                plt.savefig(os.path.join(fov_output_folder, f"{save_base}.png"), dpi=300, bbox_inches="tight")
                plt.savefig(os.path.join(fov_output_folder, f"{save_base}.svg"), bbox_inches="tight")
                plt.close()


def plot_violinplots_by_peptide(df, features):
    violin_output_folder = os.path.join(output_folder, "violinplots")
    os.makedirs(violin_output_folder, exist_ok=True)

    for peptide in PEPTIDE_ORDER:
        for timepoint in get_timepoints(df):

            peptide_df = df[
                (df["peptide"] == peptide)
                & (df["timepoint"] == timepoint)
            ].copy()

            if peptide_df.empty:
                continue

            concentration_order = get_concentration_order(peptide_df)

            for feature in features:
                if feature not in peptide_df.columns:
                    continue

                plot_df = peptide_df.dropna(
                    subset=[feature, "concentration", "media"]
                ).copy()

                if plot_df.empty:
                    continue

                plt.figure(figsize=(8, 5))

                ax = sns.violinplot(
                    data=plot_df,
                    x="concentration",
                    y=feature,
                    hue="media",
                    order=concentration_order,
                    hue_order=MEDIA_ORDER,
                    cut=0,
                    inner="quartile",
                    linewidth=1,
                    density_norm="width",
                )

                sns.despine()
                plt.title(f"{peptide} | {timepoint} | {feature} violin")
                plt.xlabel("Concentration")
                plt.ylabel(feature)

                handles, labels = ax.get_legend_handles_labels()
                ax.legend(
                    handles[:len(MEDIA_ORDER)],
                    labels[:len(MEDIA_ORDER)],
                    title="media"
                )

                plt.tight_layout()

                save_base = f"{peptide}_{timepoint}_{feature}_violinplot"
                plt.savefig(os.path.join(violin_output_folder, f"{save_base}.png"), dpi=300, bbox_inches="tight")
                plt.savefig(os.path.join(violin_output_folder, f"{save_base}.svg"), bbox_inches="tight")
                plt.close()


def plot_feature_grid_by_peptide(df, features, plot_type="strip"):
    valid_features = [f for f in features if f in df.columns]

    for peptide in PEPTIDE_ORDER:
        for timepoint in get_timepoints(df):

            peptide_df = df[
                (df["peptide"] == peptide)
                & (df["timepoint"] == timepoint)
            ].copy()

            if peptide_df.empty:
                continue

            concentration_order = get_concentration_order(peptide_df)

            n_features = len(valid_features)
            ncols = 3
            nrows = math.ceil(n_features / ncols)

            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(5 * ncols, 4.5 * nrows)
            )

            axes = np.array(axes).flatten()

            for i, feature in enumerate(valid_features):
                ax = axes[i]
                plot_df = peptide_df.dropna(subset=[feature]).copy()

                if plot_df.empty:
                    ax.axis("off")
                    continue

                if plot_type == "hist":
                    sns.histplot(
                        data=plot_df,
                        x=feature,
                        hue="media",
                        hue_order=MEDIA_ORDER,
                        element="step",
                        stat="density",
                        common_norm=False,
                        bins=30,
                        alpha=0.35,
                        ax=ax,
                    )
                    ax.set_ylabel("Density")

                else:
                    sns.boxplot(
                        data=plot_df,
                        x="concentration",
                        y=feature,
                        hue="media",
                        order=concentration_order,
                        hue_order=MEDIA_ORDER,
                        showfliers=False,
                        ax=ax,
                        zorder=1,
                    )

                    sns.stripplot(
                        data=plot_df,
                        x="concentration",
                        y=feature,
                        hue="media",
                        order=concentration_order,
                        hue_order=MEDIA_ORDER,
                        dodge=True,
                        edgecolor="black",
                        linewidth=0.2,
                        size=2.5,
                        alpha=0.35,
                        jitter=True,
                        ax=ax,
                        zorder=2,
                    )

                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(
                        handles[:len(MEDIA_ORDER)],
                        labels[:len(MEDIA_ORDER)],
                        title="media"
                    )

                    ax.set_xlabel("Concentration")
                    ax.tick_params(axis="x", rotation=35)

                ax.set_title(feature)
                sns.despine(ax=ax)

            for ax in axes[n_features:]:
                ax.axis("off")

            fig.suptitle(
                f"{peptide} | {timepoint} morphology features",
                y=1.02
            )

            fig.tight_layout()

            save_base = f"{peptide}_{timepoint}_all_features_{plot_type}_grid"
            fig.savefig(os.path.join(output_folder, f"{save_base}.png"), dpi=300, bbox_inches="tight")
            fig.savefig(os.path.join(output_folder, f"{save_base}.svg"), bbox_inches="tight")
            plt.close(fig)


def plot_fov_feature_grid_by_peptide(fov_df, features):
    valid_features = [f for f in features if f in fov_df.columns]

    for peptide in PEPTIDE_ORDER:
        for timepoint in get_timepoints(fov_df):

            peptide_df = fov_df[
                (fov_df["peptide"] == peptide)
                & (fov_df["timepoint"] == timepoint)
            ].copy()

            if peptide_df.empty:
                continue

            concentration_order = get_concentration_order(peptide_df)

            n_features = len(valid_features)
            ncols = 3
            nrows = math.ceil(n_features / ncols)

            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(5 * ncols, 4.5 * nrows)
            )

            axes = np.array(axes).flatten()

            for i, feature in enumerate(valid_features):
                ax = axes[i]

                plot_df = peptide_df.dropna(
                    subset=[feature, "concentration", "media"]
                ).copy()

                if plot_df.empty:
                    ax.axis("off")
                    continue

                sns.boxplot(
                    data=plot_df,
                    x="concentration",
                    y=feature,
                    hue="media",
                    order=concentration_order,
                    hue_order=MEDIA_ORDER,
                    showfliers=False,
                    ax=ax,
                    zorder=1,
                )

                sns.stripplot(
                    data=plot_df,
                    x="concentration",
                    y=feature,
                    hue="media",
                    order=concentration_order,
                    hue_order=MEDIA_ORDER,
                    dodge=True,
                    edgecolor="black",
                    linewidth=0.4,
                    size=5,
                    alpha=0.8,
                    jitter=True,
                    ax=ax,
                    zorder=2,
                )

                handles, labels = ax.get_legend_handles_labels()
                ax.legend(
                    handles[:len(MEDIA_ORDER)],
                    labels[:len(MEDIA_ORDER)],
                    title="media"
                )

                ax.set_title(feature)
                ax.set_xlabel("Concentration")
                ax.set_ylabel("")
                ax.tick_params(axis="x", rotation=35)
                sns.despine(ax=ax)

            for ax in axes[n_features:]:
                ax.axis("off")

            fig.suptitle(
                f"{peptide} | {timepoint} morphology features | FOV averages",
                y=1.02
            )

            fig.tight_layout()

            save_base = f"{peptide}_{timepoint}_all_features_FOV_average_grid"
            fig.savefig(os.path.join(fov_output_folder, f"{save_base}.png"), dpi=300, bbox_inches="tight")
            fig.savefig(os.path.join(fov_output_folder, f"{save_base}.svg"), bbox_inches="tight")
            plt.close(fig)


# -------------------------
# main
# -------------------------

if __name__ == "__main__":

    logger.info("Loading morphology features...")
    df = pd.read_csv(input_csv)

    logger.info(f"Loaded {len(df)} rows.")

    df = add_metadata(df)

    df.to_csv(
        os.path.join(output_folder, "cell_morphology_features_indexed.csv"),
        index=False
    )

    logger.info("Metadata parsed.")
    logger.info(
        df[
            [
                "image_name",
                "plate",
                "timepoint",
                "media",
                "peptide",
                "concentration",
                "rep",
                "well",
            ]
        ].drop_duplicates().head(20)
    )

    df = df[
        (df["peptide"] != "unknown")
        & (df["media"] != "unknown")
        & (~df["concentration"].isna())
        & (df["timepoint"] != "unknown")
    ].copy()

    df = filter_small_masks(df, min_length_um=MIN_CELL_LENGTH_UM)

    if FILTER_OUTLIERS:
        logger.info("Filtering outliers...")
        plot_df = filter_outliers_by_group(
            df,
            FEATURES_TO_PLOT,
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

    logger.info(f"Plotting {len(plot_df)} rows after filtering.")

    plot_df.to_csv(
        os.path.join(output_folder, "cell_morphology_features_indexed_filtered.csv"),
        index=False
    )

    logger.info("Making FOV average dataframe...")

    fov_df = make_fov_average_df(plot_df, FEATURES_TO_PLOT)

    fov_df.to_csv(
        os.path.join(fov_output_folder, "fov_average_morphology_features.csv"),
        index=False
    )

    logger.info(f"FOV average rows: {len(fov_df)}")

    logger.info("Making FOV average plots...")
    plot_fov_stripplots_by_peptide(fov_df, FEATURES_TO_PLOT)
    plot_fov_feature_grid_by_peptide(fov_df, FEATURES_TO_PLOT)

    logger.info("Making FOV average barplots...")
    plot_fov_barcharts_by_peptide(fov_df, FEATURES_TO_PLOT)
    plot_fov_barchart_grid_by_peptide(fov_df, FEATURES_TO_PLOT)

    logger.info("Making strip/box plots...")
    plot_stripplots_by_peptide(plot_df, FEATURES_TO_PLOT)

    logger.info("Making violin plots...")
    plot_violinplots_by_peptide(plot_df, FEATURES_TO_PLOT)

    logger.info("Making histograms...")
    plot_histograms_by_peptide(plot_df, FEATURES_TO_PLOT)

    logger.info("Making summary grids...")
    plot_feature_grid_by_peptide(plot_df, FEATURES_TO_PLOT, plot_type="strip")
    plot_feature_grid_by_peptide(plot_df, FEATURES_TO_PLOT, plot_type="hist")

    logger.info("done.")