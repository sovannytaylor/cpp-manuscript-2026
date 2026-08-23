import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, mannwhitneyu
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
output_folder = "KDE_KO_vs_OE_per_rep"
os.makedirs(output_folder, exist_ok=True)

PEPTIDE_ORDER = [
    "GR30",
    "CROT",
    "LL37",
    "MOLLUSC",
    "LDL",
    # add your 6th peptide here
]

CELL_ORDER = ["KO", "OE"]

FEATURES_TO_PLOT = [
    "puncta_count",
    "puncta_area_sum_um2",
]

SCALE_PX = 0.69

plt.rcParams.update({
    "font.size": 14,
    "svg.fonttype": "none"
})

CELL_COLORS = {
    "KO": "#1f77b4",
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

if "puncta_area_sum" in df.columns:
    df["puncta_area_sum_um2"] = df["puncta_area_sum"] * (SCALE_PX ** 2)

for feature in FEATURES_TO_PLOT:
    if feature not in df.columns:
        raise ValueError(f"{feature} not found. Columns are:\n{df.columns.tolist()}")


# -------------------------
# KDE + KS HELPER
# -------------------------

def get_ks_results(ko_vals, oe_vals):
    """
    Two-sided KS:
        Tests whether KO and OE are different in either direction.

    One-sided KS for OE > KO:
        Uses ks_2samp(oe_vals, ko_vals, alternative="less")

        This is confusing, but correct:
        If OE values are shifted higher, then the OE CDF is lower than the KO CDF.
    """

    ks_two_sided = ks_2samp(
        ko_vals,
        oe_vals,
        alternative="two-sided"
    )

    ks_oe_positive = ks_2samp(
        oe_vals,
        ko_vals,
        alternative="less"
    )

    return ks_two_sided, ks_oe_positive


def make_kde_plot_per_rep(plot_df, peptide, rep, feature):
    plot_df = plot_df[plot_df[feature].notna()].copy()

    ko_vals = plot_df.loc[plot_df["cell"] == "KO", feature].dropna()
    oe_vals = plot_df.loc[plot_df["cell"] == "OE", feature].dropna()

    if len(ko_vals) < 2 or len(oe_vals) < 2:
        logger.warning(f"Skipping {peptide} rep {rep} {feature}: not enough KO/OE values")
        return None

    ks_result, ks_oe_positive = get_ks_results(ko_vals, oe_vals)

    median_ko = ko_vals.median()
    median_oe = oe_vals.median()
    delta_median = median_oe - median_ko

    mean_ko = ko_vals.mean()
    mean_oe = oe_vals.mean()
    delta_mean = mean_oe - mean_ko

    result_row = {
        "peptide": peptide,
        "rep": rep,
        "feature": feature,
        "comparison": "OE vs KO",
        "n_KO": len(ko_vals),
        "n_OE": len(oe_vals),

        "KS_two_sided_D": ks_result.statistic,
        "KS_two_sided_p_value": ks_result.pvalue,

        "KS_one_sided_D_OE_positive": ks_oe_positive.statistic,
        "KS_one_sided_p_OE_positive": ks_oe_positive.pvalue,

        "median_KO": median_ko,
        "median_OE": median_oe,
        "delta_median_OE_minus_KO": delta_median,

        "mean_KO": mean_ko,
        "mean_OE": mean_oe,
        "delta_mean_OE_minus_KO": delta_mean,
    }

    fig, ax = plt.subplots(figsize=(9, 6))

    for cell in CELL_ORDER:
        cell_vals = plot_df.loc[plot_df["cell"] == cell, feature].dropna()

        sns.kdeplot(
            x=cell_vals,
            cut=0,
            ax=ax,
            color=CELL_COLORS[cell],
            linewidth=2.5,
            fill=True,
            alpha=0.25,
            label=f"{cell} n={len(cell_vals)}",
            common_norm=False,
            bw_adjust=1
        )

    sns.despine()

    ax.set_title(f"{peptide} rep {rep}: KO vs OE KDE")
    ax.set_xlabel(feature)
    ax.set_ylabel("Density")

    stats_text = (
        "KS test\n\n"
        f"Two-sided D = {ks_result.statistic:.3f}\n"
        f"Two-sided p = {ks_result.pvalue:.2e}\n\n"
        "One-sided KS\n"
        "H1: OE > KO\n"
        f"D = {ks_oe_positive.statistic:.3f}\n"
        f"p = {ks_oe_positive.pvalue:.2e}\n\n"
        "Direction\n\n"
        f"Median KO = {median_ko:.3f}\n"
        f"Median OE = {median_oe:.3f}\n"
        f"ΔMedian OE-KO = {delta_median:+.3f}"
    )

    fig.subplots_adjust(right=0.72)

    ax.text(
        1.03,
        0.95,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        clip_on=False,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="gray",
            alpha=0.9
        )
    )

    ax.legend(
        title="Cell type",
        loc="upper left",
        bbox_to_anchor=(1.03, 0.45),
        frameon=False
    )

    safe_rep = str(rep).replace("/", "-").replace("\\", "-").replace(" ", "_")
    save_base = f"{peptide}_rep-{safe_rep}_{feature}_KO_vs_OE_KDE_KS_oneSided"

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

    return result_row


def make_ecdf_plot_per_rep(plot_df, peptide, rep, feature):
    plot_df = plot_df[plot_df[feature].notna()].copy()

    ko_vals = plot_df.loc[plot_df["cell"] == "KO", feature].dropna()
    oe_vals = plot_df.loc[plot_df["cell"] == "OE", feature].dropna()

    if len(ko_vals) < 2 or len(oe_vals) < 2:
        logger.warning(f"Skipping ECDF {peptide} rep {rep} {feature}: not enough KO/OE values")
        return None

    ks_result, ks_oe_positive = get_ks_results(ko_vals, oe_vals)

    median_ko = ko_vals.median()
    median_oe = oe_vals.median()
    delta_median = median_oe - median_ko

    fig, ax = plt.subplots(figsize=(9, 6))

    for cell in CELL_ORDER:
        cell_vals = plot_df.loc[plot_df["cell"] == cell, feature].dropna()

        sns.ecdfplot(
            x=cell_vals,
            ax=ax,
            color=CELL_COLORS[cell],
            linewidth=2.5,
            label=f"{cell} n={len(cell_vals)}"
        )

    sns.despine()

    ax.set_title(f"{peptide} rep {rep}: KO vs OE ECDF")
    ax.set_xlabel(feature)
    ax.set_ylabel("Cumulative probability")

    stats_text = (
        "KS test\n\n"
        f"Two-sided D = {ks_result.statistic:.3f}\n"
        f"Two-sided p = {ks_result.pvalue:.2e}\n\n"
        "One-sided KS\n"
        "H1: OE > KO\n"
        f"D = {ks_oe_positive.statistic:.3f}\n"
        f"p = {ks_oe_positive.pvalue:.2e}\n\n"
        "Direction\n\n"
        f"Median KO = {median_ko:.3f}\n"
        f"Median OE = {median_oe:.3f}\n"
        f"ΔMedian OE-KO = {delta_median:+.3f}"
    )

    fig.subplots_adjust(right=0.72)

    ax.text(
        1.03,
        0.95,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        clip_on=False,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="gray",
            alpha=0.9
        )
    )

    ax.legend(
        title="Cell type",
        loc="upper left",
        bbox_to_anchor=(1.03, 0.45),
        frameon=False
    )

    safe_rep = str(rep).replace("/", "-").replace("\\", "-").replace(" ", "_")
    save_base = f"{peptide}_rep-{safe_rep}_{feature}_KO_vs_OE_ECDF_KS_oneSided"

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
# MAKE KDE + ECDF PLOTS
# -------------------------

all_results = []

for feature in FEATURES_TO_PLOT:
    logger.info(f"Plotting feature: {feature}")

    for peptide in PEPTIDE_ORDER:
        peptide_df = df[df["peptide"] == peptide].copy()

        if peptide_df.empty:
            logger.warning(f"No data for peptide: {peptide}")
            continue

        reps = sorted(peptide_df["rep"].dropna().unique())

        for rep in reps:
            rep_df = peptide_df[peptide_df["rep"] == rep].copy()

            result = make_kde_plot_per_rep(
                plot_df=rep_df,
                peptide=peptide,
                rep=rep,
                feature=feature
            )

            make_ecdf_plot_per_rep(
                plot_df=rep_df,
                peptide=peptide,
                rep=rep,
                feature=feature
            )

            if result is not None:
                all_results.append(result)


# -------------------------
# MANN-WHITNEY ON REP MEDIANS
# -------------------------

mw_results = []

for feature in FEATURES_TO_PLOT:
    for peptide in PEPTIDE_ORDER:
        peptide_df = df[df["peptide"] == peptide].copy()

        if peptide_df.empty:
            continue

        rep_medians = (
            peptide_df
            .groupby(["rep", "cell"])[feature]
            .median()
            .reset_index()
        )

        ko_rep_medians = rep_medians.loc[
            rep_medians["cell"] == "KO", feature
        ].dropna()

        oe_rep_medians = rep_medians.loc[
            rep_medians["cell"] == "OE", feature
        ].dropna()

        if len(ko_rep_medians) < 2 or len(oe_rep_medians) < 2:
            logger.warning(f"Skipping MW {peptide} {feature}: not enough reps")
            continue

        mw_two_sided = mannwhitneyu(
            oe_rep_medians,
            ko_rep_medians,
            alternative="two-sided"
        )

        mw_oe_positive = mannwhitneyu(
            oe_rep_medians,
            ko_rep_medians,
            alternative="greater"
        )

        mw_results.append({
            "peptide": peptide,
            "feature": feature,
            "comparison": "OE vs KO",
            "n_KO_reps": len(ko_rep_medians),
            "n_OE_reps": len(oe_rep_medians),

            "median_of_KO_rep_medians": ko_rep_medians.median(),
            "median_of_OE_rep_medians": oe_rep_medians.median(),
            "delta_median_OE_minus_KO": oe_rep_medians.median() - ko_rep_medians.median(),

            "MW_two_sided_U": mw_two_sided.statistic,
            "MW_two_sided_p_value": mw_two_sided.pvalue,

            "MW_one_sided_U_OE_positive": mw_oe_positive.statistic,
            "MW_one_sided_p_OE_positive": mw_oe_positive.pvalue,
        })

mw_df = pd.DataFrame(mw_results)

mw_df.to_csv(
    os.path.join(output_folder, "MannWhitney_KO_vs_OE_rep_medians_with_one_sided.csv"),
    index=False
)

print("\nMANN-WHITNEY KO vs OE USING REP MEDIANS")
print(mw_df.to_string(index=False))


# -------------------------
# SAVE KS RESULTS TABLE
# -------------------------

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    os.path.join(output_folder, "KS_KO_vs_OE_per_rep_with_one_sided_OE_positive.csv"),
    index=False
)

print("\nKS KO vs OE PER REP RESULTS WITH ONE-SIDED TEST")
print(results_df.to_string(index=False))

logger.info(f"Saved {len(results_df)} KDE plots/results to: {output_folder}")