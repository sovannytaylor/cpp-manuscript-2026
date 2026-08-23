import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger

logger.info("import ok")

plt.rcParams.update({"font.size": 14})
sns.set_palette("Paired")

input_folder = "results/summary_calculations-02/"
output_folder = "results/plotting_histograms_capped-02/"

os.makedirs(output_folder, exist_ok=True)


def clean_names(df):
    df = df.copy()

    df["cell"] = df["cell"].astype(str).str.strip().replace({
        "OE-EGFP": "OE",
        "OE-eGFP": "OE",
        "OE-eGFP0": "OE",
        "OE-GFP": "OE",
    })

    df["peptide"] = df["peptide"].astype(str).str.strip()

    return df


def plot_kde_by_peptide(df):
    feature = "puncta_count"

    plot_df = df.dropna(subset=[feature, "peptide", "cell"]).copy()
    plot_df = plot_df[plot_df[feature] <= 50]

    cell_order = [c for c in ["KO", "WT", "OE"] if c in plot_df["cell"].unique()]
    peptide_order = sorted(plot_df["peptide"].unique())

    for peptide in peptide_order:
        peptide_df = plot_df[plot_df["peptide"] == peptide].copy()

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.kdeplot(
            data=peptide_df,
            x=feature,
            hue="cell",
            hue_order=cell_order,
            common_norm=False,
            fill=False,
            linewidth=3,
            bw_adjust=1.0,
            ax=ax,
        )

        ax.set_xlim(0, 50)
        ax.set_title(f"{peptide} puncta count KDE, ≤50 puncta")
        ax.set_xlabel("Puncta count per cell")
        ax.set_ylabel("Density")
        sns.despine(ax=ax)

        fig.tight_layout()
        fig.savefig(
            os.path.join(output_folder, f"{peptide}_puncta_count_kde_capped50.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_step_hist_by_peptide(df):
    feature = "puncta_count"

    plot_df = df.dropna(subset=[feature, "peptide", "cell"]).copy()
    plot_df = plot_df[plot_df[feature] <= 50]

    cell_order = [c for c in ["KO", "WT", "OE"] if c in plot_df["cell"].unique()]
    peptide_order = sorted(plot_df["peptide"].unique())

    bins = range(0, 52)

    for peptide in peptide_order:
        peptide_df = plot_df[plot_df["peptide"] == peptide].copy()

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.histplot(
            data=peptide_df,
            x=feature,
            hue="cell",
            hue_order=cell_order,
            bins=bins,
            stat="count",
            common_norm=False,
            element="step",
            fill=False,
            linewidth=3,
            ax=ax,
        )

        ax.set_xlim(0, 50)
        ax.set_title(f"{peptide} puncta count histogram, ≤50 puncta")
        ax.set_xlabel("Puncta count per cell")
        ax.set_ylabel("Number of cells")
        sns.despine(ax=ax)

        fig.tight_layout()
        fig.savefig(
            os.path.join(output_folder, f"{peptide}_puncta_count_step_hist_capped50.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


if __name__ == "__main__":
    logger.info("Loading data...")

    df = pd.read_csv(
        os.path.join(input_folder, "percell_puncta_features.csv")
    )

    df = clean_names(df)

    logger.info("Generating KDE plots...")
    plot_kde_by_peptide(df)

    logger.info("Generating step histogram plots...")
    plot_step_hist_by_peptide(df)

    logger.info("plotting complete.")