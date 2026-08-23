import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from loguru import logger

logger.info("import ok")

# -------------------------
# CONFIG
# -------------------------

files = [
    "percell_puncta_features.csv",
    "percell_puncta_features(1).csv",
]

input_folder = r"C:\Users\u244278\OneDrive - Baylor College of Medicine\Documents\python_projects\ANA_26039_LDLR_HEPG2-uptake\results\concat"
output_folder = "cumulative_KO_WT_OE_plots"
os.makedirs(output_folder, exist_ok=True)

PEPTIDE_ORDER = [
    "GR30",
    "CROT",
    "LL37",
    "MOLLUSC",
    "LDL",
]

CELL_ORDER = ["KO", "WT", "OE"]

FEATURES_TO_PLOT = [
    "puncta_count",
    "puncta_area_sum_um2",
    "puncta_area_proportion",
    "puncta_coi1_intensity_mean",
    "cell_coi1_intensity_mean",
]

SCALE_PX = 0.69

plt.rcParams.update({
    "font.size": 14,
    "svg.fonttype": "none"
})

CELL_COLORS = {
    "KO": "#1f77b4",
    "WT": "#2ca02c",
    "OE": "#d62728",
}


# -------------------------
# LOAD / CONCAT
# -------------------------

def read_table(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".xlsx") or path.endswith(".xls"):
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")


dfs = []

for file in files:
    path = os.path.join(input_folder, file)
    temp = read_table(path)
    temp["source_file"] = file
    dfs.append(temp)

df = pd.concat(dfs, ignore_index=True)

logger.info(f"Combined rows: {len(df)}")
logger.info(f"Columns: {df.columns.tolist()}")


# -------------------------
# CLEAN
# -------------------------

df["cell"] = df["cell"].astype(str).str.strip().replace({
    "OE-EGFP": "OE",
    "OE-eGFP": "OE",
    "OE-eGFP0": "OE",
    "OE-GFP": "OE",
})

df["peptide"] = df["peptide"].astype(str).str.strip().replace({
    "GR30-594": "GR30",
    "GP30-594": "GP30",
    "LL37-594": "LL37",
    "MOLLUSC-594": "MOLLUSC",
    "LDL-DIL": "LDL",
})

df["rep"] = df["rep"].astype(str).str.strip()

df = df[
    df["cell"].isin(CELL_ORDER)
    & df["peptide"].isin(PEPTIDE_ORDER)
].copy()

df["cell"] = pd.Categorical(df["cell"], categories=CELL_ORDER, ordered=True)
df["peptide"] = pd.Categorical(df["peptide"], categories=PEPTIDE_ORDER, ordered=True)

if "puncta_area_sum" in df.columns:
    df["puncta_area_sum_um2"] = df["puncta_area_sum"] * (SCALE_PX ** 2)


from scipy.stats import ks_2samp

# -------------------------
# KS TEST PER REP
# -------------------------

ks_results = []

COMPARISONS = [
    ("KO", "WT"),
    ("OE", "WT"),
    ("KO", "OE"),
]

for peptide in PEPTIDE_ORDER:
    peptide_df = df[df["peptide"] == peptide].copy()

    for feature in FEATURES_TO_PLOT:
        if feature not in peptide_df.columns:
            logger.warning(f"Skipping missing feature for KS: {feature}")
            continue

        for rep in sorted(peptide_df["rep"].dropna().unique()):
            rep_df = peptide_df[peptide_df["rep"] == rep].copy()

            for group1, group2 in COMPARISONS:
                vals1 = rep_df.loc[rep_df["cell"] == group1, feature].dropna()
                vals2 = rep_df.loc[rep_df["cell"] == group2, feature].dropna()

                if len(vals1) == 0 or len(vals2) == 0:
                    continue

                result = ks_2samp(vals1, vals2)

                ks_results.append({
                    "peptide": peptide,
                    "feature": feature,
                    "rep": rep,
                    "comparison": f"{group1} vs {group2}",
                    "group1": group1,
                    "group2": group2,
                    "n_group1": len(vals1),
                    "n_group2": len(vals2),
                    "KS_D": result.statistic,
                    "p_value": result.pvalue,
                })

ks_df = pd.DataFrame(ks_results)

ks_df.to_csv(
    os.path.join(output_folder, "KS_test_per_rep_results.csv"),
    index=False
)

print("\nKS TEST PER REP RESULTS")
print(ks_df.to_string(index=False))

# -------------------------
# N SUMMARY
# -------------------------

total_n_summary = (
    df.groupby(["peptide", "cell"], observed=True)
    .agg(
        total_cells=("cell_number", "count"),
        n_reps=("rep", "nunique")
    )
    .reset_index()
)

total_n_summary.to_csv(
    os.path.join(output_folder, "total_cell_and_rep_N_summary.csv"),
    index=False
)

print("\nTOTAL CELL N AND REP N")
print(total_n_summary.to_string(index=False))


# -------------------------
# ECDF HELPER
# -------------------------

def get_ecdf(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return None, None

    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)

    return x, y


# -------------------------
# PLOT HELPER
# -------------------------

def make_rep_overlay_cumulative_plot(plot_df, feature, peptide):
    plot_df = plot_df[plot_df[feature].notna()].copy()

    if plot_df.empty:
        logger.warning(f"No data for {peptide} {feature}")
        return

    fig, ax = plt.subplots(figsize=(11, 6))

    ks_text_lines = []

    comparisons = [
        ("KO", "WT"),
        ("OE", "WT"),
        ("KO", "OE"),
    ]

    for group1, group2 in comparisons:
        vals1 = plot_df.loc[plot_df["cell"] == group1, feature].dropna()
        vals2 = plot_df.loc[plot_df["cell"] == group2, feature].dropna()

        if len(vals1) > 0 and len(vals2) > 0:
            ks_result = ks_2samp(vals1, vals2)

            ks_text_lines.append(
                f"{group1} vs {group2}: D={ks_result.statistic:.3f}"
            )

    for cell in CELL_ORDER:
        cell_df = plot_df[plot_df["cell"] == cell].copy()

        if cell_df.empty:
            continue

        color = CELL_COLORS[cell]

        # light lines = individual reps
        for rep, rep_df in cell_df.groupby("rep", observed=True):
            x, y = get_ecdf(rep_df[feature].values)

            if x is None:
                continue

            ax.step(
                x,
                y,
                where="post",
                color=color,
                alpha=0.22,
                linewidth=1.2,
            )

        # dark line = pooled cells across all reps
        x_all, y_all = get_ecdf(cell_df[feature].values)

        if x_all is not None:
            ax.step(
                x_all,
                y_all,
                where="post",
                color=color,
                alpha=1,
                linewidth=3,
                label=f"{cell} pooled"
            )

    sns.despine()

    ax.set_title(f"{peptide}: cumulative distribution of {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Cumulative fraction of cells")

    # leave space on right for legend + KS box
    fig.subplots_adjust(right=0.72)

    ax.legend(
        title="Cell type",
        loc="upper left",
        bbox_to_anchor=(1.03, 1.0),
        frameon=False
    )

    if ks_text_lines:
        ks_text = "KS statistic (pooled)\n\n" + "\n".join(ks_text_lines)

        ax.text(
            1.03,
            0.55,
            ks_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            clip_on=False,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="gray",
                alpha=0.9
            )
        )

    save_base = f"{peptide}_{feature}_rep_lines_with_pooled_KS"

    plt.savefig(
        os.path.join(output_folder, f"{save_base}.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.savefig(
        os.path.join(output_folder, f"{save_base}.svg"),
        bbox_inches="tight"
    )

    plt.close()


# -------------------------
# MAKE PLOTS
# -------------------------

for peptide in PEPTIDE_ORDER:
    peptide_df = df[df["peptide"] == peptide].copy()

    if peptide_df.empty:
        logger.warning(f"No data for peptide: {peptide}")
        continue

    for feature in FEATURES_TO_PLOT:
        if feature not in peptide_df.columns:
            logger.warning(f"Skipping missing feature: {feature}")
            continue

        make_rep_overlay_cumulative_plot(
            plot_df=peptide_df,
            feature=feature,
            peptide=peptide
        )


logger.info("done.")