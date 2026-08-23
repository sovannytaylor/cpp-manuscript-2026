"""
Make per-cell and per-replicate colocalization summaries.

Plots:
    x = peptide
    hue = lipoprotein
    y = % peptide puncta colocalized with lipoprotein

Also makes CDF plots:
    one plot per peptide
    each line = lipoprotein

Saves:
    PNG + SVG for every plot
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger


# -------------------------
# CONFIG
# -------------------------

input_folder = "results/summary_calculations-peptide-lipo-coloc/"
output_folder = "results/summary_calculations-peptide-lipo-coloc/"
plot_folder = os.path.join(output_folder, "plots")

os.makedirs(output_folder, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)

input_file = os.path.join(input_folder, "peptide_lipoprotein_puncta_features.csv")

PEPTIDE_ORDER = ["GP30", "GR30", "CROT", "LL37"]
LIPOPROTEIN_ORDER = ["HDL", "LDL", "VLDL"]


# -------------------------
# HELPERS
# -------------------------

def parse_rep_from_image_name(name):
    base = os.path.basename(str(name))
    base = base.removesuffix(".npy").removesuffix("_mask")

    m = re.search(r"-(\d+)$", base)
    if m:
        return f"REP{m.group(1)}"

    return "REP_UNKNOWN"


def save_figure(fig, filename_base):
    png_path = os.path.join(plot_folder, f"{filename_base}.png")
    svg_path = os.path.join(plot_folder, f"{filename_base}.svg")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")

    logger.info(f"Saved plot: {png_path}")
    logger.info(f"Saved plot: {svg_path}")


# -------------------------
# SUMMARY
# -------------------------

def make_cell_summary(df):
    rows = []

    group_cols = [
        "image_name",
        "lipoprotein",
        "peptide",
        "concentration_nM",
        "cell_number",
    ]

    for group_vals, g in df.groupby(group_cols, dropna=False):
        image_name, lipoprotein, peptide, concentration_nM, cell_number = group_vals

        peptide_rows = g[g["puncta_area"] > 0].copy()
        n_peptide_puncta = len(peptide_rows)

        if n_peptide_puncta > 0:
            n_coloc_exact = int(peptide_rows["peptide_punctum_coloc_exact"].sum())
            n_coloc_1px = int(peptide_rows["peptide_punctum_coloc_within_1px"].sum())

            fraction_coloc_exact = n_coloc_exact / n_peptide_puncta
            fraction_coloc_1px = n_coloc_1px / n_peptide_puncta

            percent_coloc_exact = fraction_coloc_exact * 100
            percent_coloc_1px = fraction_coloc_1px * 100

            mean_peptide_punctum_intensity = peptide_rows["peptide_punctum_intensity_mean"].mean()
            mean_lipo_inside_peptide = peptide_rows["lipo_intensity_inside_peptide_punctum_mean"].mean()
            mean_puncta_area = peptide_rows["puncta_area"].mean()
            total_puncta_area = peptide_rows["puncta_area"].sum()

        else:
            n_coloc_exact = 0
            n_coloc_1px = 0
            fraction_coloc_exact = np.nan
            fraction_coloc_1px = np.nan
            percent_coloc_exact = np.nan
            percent_coloc_1px = np.nan
            mean_peptide_punctum_intensity = np.nan
            mean_lipo_inside_peptide = np.nan
            mean_puncta_area = np.nan
            total_puncta_area = 0

        cell_size = g["cell_size"].iloc[0]

        rows.append({
            "image_name": image_name,
            "rep": parse_rep_from_image_name(image_name),
            "lipoprotein": lipoprotein,
            "peptide": peptide,
            "concentration_nM": concentration_nM,
            "cell_number": cell_number,
            "cell_size": cell_size,

            "n_peptide_puncta": n_peptide_puncta,
            "n_lipoprotein_puncta": g["n_lipoprotein_puncta"].iloc[0],

            "n_peptide_puncta_coloc_exact": n_coloc_exact,
            "n_peptide_puncta_coloc_within_1px": n_coloc_1px,

            "fraction_peptide_puncta_coloc_exact": fraction_coloc_exact,
            "fraction_peptide_puncta_coloc_within_1px": fraction_coloc_1px,

            "percent_peptide_puncta_coloc_exact": percent_coloc_exact,
            "percent_peptide_puncta_coloc_within_1px": percent_coloc_1px,

            "mean_peptide_punctum_intensity": mean_peptide_punctum_intensity,
            "mean_lipo_intensity_inside_peptide_punctum": mean_lipo_inside_peptide,
            "mean_puncta_area": mean_puncta_area,
            "total_puncta_area": total_puncta_area,
            "puncta_area_proportion": (total_puncta_area / cell_size) * 100 if cell_size > 0 else np.nan,

            "cell_peptide_intensity_mean": g["cell_peptide_intensity_mean"].iloc[0],
            "cell_lipoprotein_intensity_mean": g["cell_lipoprotein_intensity_mean"].iloc[0],
        })

    return pd.DataFrame(rows)


def make_rep_summary(cell_df):
    features = [
        "n_peptide_puncta",
        "n_lipoprotein_puncta",
        "n_peptide_puncta_coloc_exact",
        "n_peptide_puncta_coloc_within_1px",
        "fraction_peptide_puncta_coloc_exact",
        "fraction_peptide_puncta_coloc_within_1px",
        "percent_peptide_puncta_coloc_exact",
        "percent_peptide_puncta_coloc_within_1px",
        "mean_peptide_punctum_intensity",
        "mean_lipo_intensity_inside_peptide_punctum",
        "mean_puncta_area",
        "total_puncta_area",
        "puncta_area_proportion",
        "cell_peptide_intensity_mean",
        "cell_lipoprotein_intensity_mean",
    ]

    group_cols = ["lipoprotein", "peptide", "concentration_nM", "rep"]

    rep_df = (
        cell_df
        .groupby(group_cols, dropna=False)[features]
        .mean()
        .reset_index()
    )

    rep_df["n_cells"] = (
        cell_df
        .groupby(group_cols, dropna=False)
        .size()
        .values
    )

    return rep_df


# -------------------------
# PLOTTING HELPERS
# -------------------------

def get_plot_dfs(cell_df, rep_df):
    plot_df = cell_df.copy()
    rep_plot_df = rep_df.copy()

    plot_df = plot_df[
        plot_df["peptide"].isin(PEPTIDE_ORDER) &
        plot_df["lipoprotein"].isin(LIPOPROTEIN_ORDER)
    ]

    rep_plot_df = rep_plot_df[
        rep_plot_df["peptide"].isin(PEPTIDE_ORDER) &
        rep_plot_df["lipoprotein"].isin(LIPOPROTEIN_ORDER)
    ]

    x_labels = [p for p in PEPTIDE_ORDER if p in plot_df["peptide"].unique()]
    hue_labels = [l for l in LIPOPROTEIN_ORDER if l in plot_df["lipoprotein"].unique()]

    return plot_df, rep_plot_df, x_labels, hue_labels


# -------------------------
# BAR PLOTS
# -------------------------

def barplot_cells_and_reps(cell_df, rep_df, y_col, ylabel, filename_base, force_percent_axis=False):
    plot_df, rep_plot_df, x_labels, hue_labels = get_plot_dfs(cell_df, rep_df)

    x_pos = np.arange(len(x_labels))
    bar_width = 0.8 / max(len(hue_labels), 1)

    offsets = np.linspace(
        -0.4 + bar_width / 2,
        0.4 - bar_width / 2,
        len(hue_labels)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    rng = np.random.default_rng(42)

    for h_idx, lipoprotein in enumerate(hue_labels):
        means = []
        sems = []

        for peptide in x_labels:
            vals = plot_df.loc[
                (plot_df["peptide"] == peptide) &
                (plot_df["lipoprotein"] == lipoprotein),
                y_col
            ].dropna()

            means.append(vals.mean())
            sems.append(vals.sem() if len(vals) > 1 else 0)

        bar_positions = x_pos + offsets[h_idx]

        ax.bar(
            bar_positions,
            means,
            width=bar_width,
            yerr=sems,
            capsize=4,
            alpha=0.55,
            edgecolor="black",
            linewidth=1,
            label=lipoprotein,
        )

        for i, peptide in enumerate(x_labels):
            vals = plot_df.loc[
                (plot_df["peptide"] == peptide) &
                (plot_df["lipoprotein"] == lipoprotein),
                y_col
            ].dropna()

            jitter = rng.normal(0, bar_width * 0.18, size=len(vals))

            ax.scatter(
                np.full(len(vals), x_pos[i] + offsets[h_idx]) + jitter,
                vals,
                alpha=0.18,
                s=16,
                color="gray",
            )

        for i, peptide in enumerate(x_labels):
            vals = rep_plot_df.loc[
                (rep_plot_df["peptide"] == peptide) &
                (rep_plot_df["lipoprotein"] == lipoprotein),
                y_col
            ].dropna()

            jitter = rng.normal(0, bar_width * 0.10, size=len(vals))

            ax.scatter(
                np.full(len(vals), x_pos[i] + offsets[h_idx]) + jitter,
                vals,
                alpha=0.95,
                s=65,
                color="black",
                edgecolor="white",
                linewidth=0.5,
            )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Peptide")

    if force_percent_axis:
        ax.set_ylim(0, 100)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Lipoprotein", frameon=False)

    fig.tight_layout()
    save_figure(fig, filename_base)
    plt.close(fig)


# -------------------------
# VIOLIN PLOTS
# -------------------------

def violinplot_cells_and_reps(cell_df, rep_df, y_col, ylabel, filename_base, force_percent_axis=False):
    plot_df, rep_plot_df, x_labels, hue_labels = get_plot_dfs(cell_df, rep_df)

    x_pos = np.arange(len(x_labels))
    violin_width = 0.8 / max(len(hue_labels), 1)

    offsets = np.linspace(
        -0.4 + violin_width / 2,
        0.4 - violin_width / 2,
        len(hue_labels)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    rng = np.random.default_rng(42)

    for h_idx, lipoprotein in enumerate(hue_labels):
        for i, peptide in enumerate(x_labels):
            vals = plot_df.loc[
                (plot_df["peptide"] == peptide) &
                (plot_df["lipoprotein"] == lipoprotein),
                y_col
            ].dropna()

            if len(vals) == 0:
                continue

            pos = x_pos[i] + offsets[h_idx]

            if len(vals) >= 3:
                parts = ax.violinplot(
                    vals,
                    positions=[pos],
                    widths=violin_width * 0.9,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )

                for body in parts["bodies"]:
                    body.set_alpha(0.35)
                    body.set_edgecolor("black")
                    body.set_linewidth(0.8)

            jitter = rng.normal(0, violin_width * 0.12, size=len(vals))

            ax.scatter(
                np.full(len(vals), pos) + jitter,
                vals,
                alpha=0.15,
                s=14,
                color="gray",
            )

            rep_vals = rep_plot_df.loc[
                (rep_plot_df["peptide"] == peptide) &
                (rep_plot_df["lipoprotein"] == lipoprotein),
                y_col
            ].dropna()

            rep_jitter = rng.normal(0, violin_width * 0.08, size=len(rep_vals))

            ax.scatter(
                np.full(len(rep_vals), pos) + rep_jitter,
                rep_vals,
                alpha=0.95,
                s=65,
                color="black",
                edgecolor="white",
                linewidth=0.5,
            )

    for lipoprotein in hue_labels:
        ax.scatter([], [], label=lipoprotein)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Peptide")

    if force_percent_axis:
        ax.set_ylim(0, 100)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Lipoprotein", frameon=False)

    fig.tight_layout()
    save_figure(fig, filename_base)
    plt.close(fig)


# -------------------------
# CDF PLOTS
# -------------------------

def cdfplot_by_peptide(cell_df, y_col, xlabel, filename_base, force_percent_axis=False):
    plot_df = cell_df.copy()

    plot_df = plot_df[
        plot_df["peptide"].isin(PEPTIDE_ORDER) &
        plot_df["lipoprotein"].isin(LIPOPROTEIN_ORDER)
    ]

    peptides = [p for p in PEPTIDE_ORDER if p in plot_df["peptide"].unique()]
    lipoproteins = [l for l in LIPOPROTEIN_ORDER if l in plot_df["lipoprotein"].unique()]

    for peptide in peptides:
        fig, ax = plt.subplots(figsize=(6, 5))

        for lipoprotein in lipoproteins:
            vals = plot_df.loc[
                (plot_df["peptide"] == peptide) &
                (plot_df["lipoprotein"] == lipoprotein),
                y_col
            ].dropna().sort_values()

            if len(vals) == 0:
                continue

            x = vals.to_numpy()
            y = np.arange(1, len(x) + 1) / len(x)

            ax.step(
                x,
                y,
                where="post",
                linewidth=2,
                label=lipoprotein,
            )

        ax.set_title(peptide)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Cumulative fraction of cells")

        if force_percent_axis:
            ax.set_xlim(0, 100)

        ax.set_ylim(0, 1)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(title="Lipoprotein", frameon=False)

        fig.tight_layout()

        safe_peptide = peptide.areplace("/", "-")
        save_figure(fig, f"{filename_base}_{safe_peptide}")

        plt.close(fig)


# -------------------------
# MAKE ALL PLOTS
# -------------------------

def make_all_plots(cell_df, rep_df):
    barplot_cells_and_reps(
        cell_df=cell_df,
        rep_df=rep_df,
        y_col="percent_peptide_puncta_coloc_within_1px",
        ylabel="% peptide puncta colocalized with lipoprotein",
        filename_base="barplot_percent_peptide_puncta_coloc_within_1px_by_lipoprotein",
        force_percent_axis=True,
    )

    violinplot_cells_and_reps(
        cell_df=cell_df,
        rep_df=rep_df,
        y_col="percent_peptide_puncta_coloc_within_1px",
        ylabel="% peptide puncta colocalized with lipoprotein",
        filename_base="violinplot_percent_peptide_puncta_coloc_within_1px_by_lipoprotein",
        force_percent_axis=True,
    )

    cdfplot_by_peptide(
        cell_df=cell_df,
        y_col="percent_peptide_puncta_coloc_within_1px",
        xlabel="% peptide puncta colocalized with lipoprotein",
        filename_base="cdf_percent_peptide_puncta_coloc_within_1px_by_peptide",
        force_percent_axis=True,
    )

    barplot_cells_and_reps(
        cell_df=cell_df,
        rep_df=rep_df,
        y_col="n_peptide_puncta_coloc_within_1px",
        ylabel="Colocalized peptide puncta per cell",
        filename_base="barplot_n_peptide_puncta_coloc_within_1px_by_lipoprotein",
    )

    violinplot_cells_and_reps(
        cell_df=cell_df,
        rep_df=rep_df,
        y_col="n_peptide_puncta_coloc_within_1px",
        ylabel="Colocalized peptide puncta per cell",
        filename_base="violinplot_n_peptide_puncta_coloc_within_1px_by_lipoprotein",
    )

    cdfplot_by_peptide(
        cell_df=cell_df,
        y_col="n_peptide_puncta_coloc_within_1px",
        xlabel="Colocalized peptide puncta per cell",
        filename_base="cdf_n_peptide_puncta_coloc_within_1px_by_peptide",
    )

    barplot_cells_and_reps(
        cell_df=cell_df,
        rep_df=rep_df,
        y_col="n_peptide_puncta",
        ylabel="Peptide puncta per cell",
        filename_base="barplot_n_peptide_puncta_by_lipoprotein",
    )

    violinplot_cells_and_reps(
        cell_df=cell_df,
        rep_df=rep_df,
        y_col="n_peptide_puncta",
        ylabel="Peptide puncta per cell",
        filename_base="violinplot_n_peptide_puncta_by_lipoprotein",
    )

    cdfplot_by_peptide(
        cell_df=cell_df,
        y_col="n_peptide_puncta",
        xlabel="Peptide puncta per cell",
        filename_base="cdf_n_peptide_puncta_by_peptide",
    )


# -------------------------
# MAIN
# -------------------------

if __name__ == "__main__":
    logger.info("Loading puncta-level features...")

    df = pd.read_csv(input_file)

    logger.info(f"Loaded {len(df)} puncta-level rows")

    cell_summary = make_cell_summary(df)

    cell_summary.to_csv(
        os.path.join(output_folder, "peptide_lipoprotein_cell_summary.csv"),
        index=False,
    )

    logger.info("Saved peptide_lipoprotein_cell_summary.csv")

    rep_summary = make_rep_summary(cell_summary)

    rep_summary.to_csv(
        os.path.join(output_folder, "peptide_lipoprotein_rep_summary.csv"),
        index=False,
    )

    logger.info("Saved peptide_lipoprotein_rep_summary.csv")

    make_all_plots(cell_summary, rep_summary)

    logger.info("Done.")