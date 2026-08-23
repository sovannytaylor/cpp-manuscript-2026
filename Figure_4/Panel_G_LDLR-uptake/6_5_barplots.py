import os
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger

logger.info("import ok")

plt.rcParams.update({"font.size": 14})
sns.set_palette("Paired")

input_folder = "results/summary_calculations-02/"
output_folder = "results/plotting_bar_sem-02/"

os.makedirs(output_folder, exist_ok=True)


def load_summary_data(input_folder):
    return {
        "percell_reps": pd.read_csv(
            f"{input_folder}percell_puncta_features_reps.csv"
        )
    }


def clean_names(dfs):
    for df in dfs.values():
        if "cell" in df.columns:
            df["cell"] = df["cell"].astype(str).str.strip()
            df["cell"] = df["cell"].replace({
                "OE-EGFP": "OE",
                "OE-eGFP": "OE",
                "OE-eGFP0": "OE",
                "OE-GFP": "OE",
            })

        if "peptide" in df.columns:
            df["peptide"] = df["peptide"].astype(str).str.strip()

    return dfs


def plot_bar_sem_points(
    data_agg,
    feature,
    save_name,
    order=None,
    hue_order=None,
    x="peptide",
    hue="cell",
):
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=data_agg,
        x=x,
        y=feature,
        hue=hue,
        order=order,
        hue_order=hue_order,
        errorbar="se",
        capsize=0.15,
        err_kws={"linewidth": 1.5},
        alpha=0.7,
        ax=ax,
    )

    sns.stripplot(
        data=data_agg,
        x=x,
        y=feature,
        hue=hue,
        order=order,
        hue_order=hue_order,
        dodge=True,
        size=8,
        edgecolor="black",
        linewidth=1,
        ax=ax,
    )

    ax.set_title(feature)
    ax.set_xlabel("Peptide")
    ax.set_ylabel(feature)
    ax.tick_params(axis="x", rotation=45)
    sns.despine(ax=ax)

    handles, labels = ax.get_legend_handles_labels()
    n = len(hue_order) if hue_order else data_agg[hue].nunique()

    ax.legend(
        handles[:n],
        labels[:n],
        title=hue,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_folder, save_name),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


if __name__ == "__main__":
    logger.info("Loading data...")

    dfs = load_summary_data(input_folder)
    dfs = clean_names(dfs)

    percell_features = [
        "cell_size",
        "mean_puncta_area",
        "puncta_area_proportion",
        "puncta_count",
        "puncta_mean_minor_axis",
        "puncta_mean_major_axis",
        "puncta_mean_aspect_ratio",
        "puncta_mean_circularity",
        "avg_eccentricity",
        "puncta_cv_mean",
        "puncta_skew_mean",
        "cell_std",
        "cell_cv",
        "cell_skew",
        "cell_coi1_intensity_mean",
        "cell_coi2_intensity_mean",
        "puncta_coi1_intensity_mean",
        "puncta_coi2_intensity_mean",
    ]

    peptide_order = sorted(
        dfs["percell_reps"]["peptide"].dropna().unique().tolist()
    )

    desired_cell_order = ["KO", "WT", "OE"]
    cell_order = [
        c for c in desired_cell_order
        if c in dfs["percell_reps"]["cell"].dropna().unique()
    ]

    features_to_plot = [
        f for f in percell_features
        if f in dfs["percell_reps"].columns
    ]

    logger.info("Generating barplots with SEM and replicate points...")

    for feature in features_to_plot:
        plot_bar_sem_points(
            data_agg=dfs["percell_reps"],
            feature=feature,
            save_name=f"{feature}_bar_sem_points.png",
            order=peptide_order,
            hue_order=cell_order,
        )

    logger.info("plotting complete.")