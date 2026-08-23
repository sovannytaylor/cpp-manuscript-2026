import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger

logger.info("import ok")

plt.rcParams.update({"font.size": 14})
sns.set_palette("Paired")

input_folder = "results/summary_calculations-02/"
output_folder = "results/plotting_log2fc-02/"

os.makedirs(output_folder, exist_ok=True)


def clean_names(df):
    df = df.copy()

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

    return df


def find_replicate_column(df):
    possible_cols = [
        "replicate",
        "rep",
        "Replicate",
        "REP",
        "biological_replicate",
        "experiment",
        "plate",
    ]

    for col in possible_cols:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not find a replicate column. "
        "Check your percell_reps columns and add the correct name."
    )


def calculate_puncta_count_log2fc(df, rep_col):
    feature = "puncta_count"

    required_cols = ["peptide", "cell", rep_col, feature]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    wt_lookup = (
        df[df["cell"] == "WT"]
        [["peptide", rep_col, feature]]
        .rename(columns={feature: "wt_puncta_count"})
    )

    df_fc = df.merge(
        wt_lookup,
        on=["peptide", rep_col],
        how="left"
    )

    missing_wt = df_fc["wt_puncta_count"].isna().sum()
    if missing_wt > 0:
        logger.warning(
            f"{missing_wt} rows do not have a matched WT value. "
            "Those rows will have NaN log2FC."
        )

    df_fc["puncta_count_log2fc_vs_WT"] = np.log2(
        (df_fc["puncta_count"] + 1) /
        (df_fc["wt_puncta_count"] + 1)
    )

    return df_fc


def plot_log2fc(df_fc, peptide_order, cell_order):
    fig, ax = plt.subplots(figsize=(11, 6))

    sns.boxplot(
        data=df_fc,
        x="peptide",
        y="puncta_count_log2fc_vs_WT",
        hue="cell",
        order=peptide_order,
        hue_order=cell_order,
        dodge=True,
        showfliers=False,
        boxprops={"facecolor": "none"},
        linewidth=1.5,
        ax=ax,
    )

    sns.stripplot(
        data=df_fc,
        x="peptide",
        y="puncta_count_log2fc_vs_WT",
        hue="cell",
        order=peptide_order,
        hue_order=cell_order,
        dodge=True,
        size=8,
        edgecolor="black",
        linewidth=1,
        ax=ax,
    )

    ax.axhline(
        y=0,
        color="black",
        linestyle="--",
        linewidth=1
    )

    ax.set_title("Puncta count log2FC vs WT")
    ax.set_ylabel("log2FC puncta count vs matched WT")
    ax.set_xlabel("Peptide")
    ax.tick_params(axis="x", rotation=45)
    sns.despine(ax=ax)

    handles, labels = ax.get_legend_handles_labels()
    n = len(cell_order)

    ax.legend(
        handles[:n],
        labels[:n],
        title="cell",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(output_folder, "puncta_count_log2fc_vs_WT.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


if __name__ == "__main__":
    logger.info("Loading replicate data...")

    df = pd.read_csv(
        os.path.join(input_folder, "percell_puncta_features_reps.csv")
    )

    df = clean_names(df)

    rep_col = find_replicate_column(df)
    logger.info(f"Using replicate column: {rep_col}")

    df_fc = calculate_puncta_count_log2fc(df, rep_col)

    df_fc.to_csv(
        os.path.join(output_folder, "percell_reps_puncta_count_log2fc_vs_WT.csv"),
        index=False,
    )

    peptide_order = sorted(df_fc["peptide"].dropna().unique().tolist())

    desired_cell_order = ["KO", "WT", "OE"]
    cell_order = [
        c for c in desired_cell_order
        if c in df_fc["cell"].dropna().unique()
    ]

    plot_log2fc(df_fc, peptide_order, cell_order)

    logger.info("plotting complete.")



def plot_log2fc(df_fc, peptide_order, cell_order):
    fig, ax = plt.subplots(figsize=(9, 5))

    plot_df = df_fc[df_fc["cell"].isin(["KO", "OE"])].copy()
    plot_cell_order = [c for c in ["KO", "OE"] if c in plot_df["cell"].unique()]

    sns.pointplot(
        data=plot_df,
        x="peptide",
        y="puncta_count_log2fc_vs_WT",
        hue="cell",
        order=peptide_order,
        hue_order=plot_cell_order,
        dodge=0.35,
        errorbar="sd",
        markers="_",
        linestyles="none",
        capsize=0.15,
        err_kws={"linewidth": 1.5},
        ax=ax,
    )

    sns.stripplot(
        data=plot_df,
        x="peptide",
        y="puncta_count_log2fc_vs_WT",
        hue="cell",
        order=peptide_order,
        hue_order=plot_cell_order,
        dodge=True,
        size=7,
        edgecolor="black",
        linewidth=1,
        ax=ax,
    )

    ax.axhline(0, color="black", linestyle="--", linewidth=1)

    ax.set_title("Puncta count relative to WT")
    ax.set_ylabel("log2FC vs matched WT")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)
    sns.despine(ax=ax)

    handles, labels = ax.get_legend_handles_labels()
    n = len(plot_cell_order)
    ax.legend(
        handles[:n],
        labels[:n],
        title="cell",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_folder, "puncta_count_log2fc_vs_WT_clean.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)