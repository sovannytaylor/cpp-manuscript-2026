import os
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

logger.info("import ok")

plt.rcParams.update({"font.size": 14})

input_folder = "results/summary_calculations-02/"
output_folder = "results/plotting_uptake_positive-02/"

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


def calculate_uptake_percentages(df):
    df = df.copy()

    df["uptake_status"] = df["puncta_count"].apply(
        lambda x: "uptake-positive" if x > 0 else "no uptake"
    )

    counts = (
        df
        .groupby(["peptide", "cell", "uptake_status"])
        .size()
        .reset_index(name="n_cells")
    )

    totals = (
        df
        .groupby(["peptide", "cell"])
        .size()
        .reset_index(name="total_cells")
    )

    summary = counts.merge(
        totals,
        on=["peptide", "cell"],
        how="left"
    )

    summary["percent"] = (
        summary["n_cells"] / summary["total_cells"] * 100
    )

    return summary


def plot_stacked_uptake(summary):
    cell_order = ["KO", "WT", "OE"]
    status_order = ["uptake-positive", "no uptake"]

    peptide_order = sorted(summary["peptide"].unique())

    for peptide in peptide_order:
        peptide_df = summary[
            summary["peptide"] == peptide
        ].copy()

        pivot = peptide_df.pivot_table(
            index="cell",
            columns="uptake_status",
            values="percent",
            fill_value=0
        )

        n_cells = (
            peptide_df.groupby("cell")["total_cells"]
            .first()
            .reindex(cell_order)
        )

        pivot = pivot.reindex(cell_order)
        pivot = pivot[[c for c in status_order if c in pivot.columns]]

        fig, ax = plt.subplots(figsize=(6, 5))

        bottom = None

        for status in status_order:
            if status not in pivot.columns:
                continue

            values = pivot[status]

            ax.bar(
                pivot.index,
                values,
                bottom=bottom,
                label=status
            )

            if bottom is None:
                bottom = values
            else:
                bottom = bottom + values

        ax.set_ylim(0, 100)
        ax.set_ylabel("Cells (%)")
        ax.set_xlabel("Cell type")
        ax.set_title(f"{peptide}: uptake-positive vs no uptake")

        xticklabels = [
            f"{cell}\n(n={int(n_cells[cell])})"
            if pd.notna(n_cells[cell])
            else cell
            for cell in pivot.index
        ]

        ax.set_xticks(range(len(pivot.index)))
        ax.set_xticklabels(xticklabels)

        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            title="Status"
        )

        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="%.1f%%",
                label_type="center",
                fontsize=10
            )

        fig.tight_layout()

        fig.savefig(
            os.path.join(
                output_folder,
                f"{peptide}_uptake_positive_stacked_bar.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)


if __name__ == "__main__":
    logger.info("Loading data...")

    df = pd.read_csv(
        os.path.join(
            input_folder,
            "percell_puncta_features.csv"
        )
    )

    df = clean_names(df)

    df = df.dropna(subset=["puncta_count", "peptide", "cell"])

    summary = calculate_uptake_percentages(df)

    summary.to_csv(
        os.path.join(
            output_folder,
            "uptake_positive_summary.csv"
        ),
        index=False
    )

    plot_stacked_uptake(summary)

    logger.info("plotting complete.")