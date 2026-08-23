import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
output_folder = "raw_KO_WT_OE_plots"
os.makedirs(output_folder, exist_ok=True)

PEPTIDE_ORDER = [
    "GR30",
    "CROT",
    "LL37",
    "MOLLUSC",
    "LDL",
]

CELL_ORDER = ["KO", "WT", "OE"]
PLOT_CELL_ORDER = ["KO", "OE"]

SCALE_PX = 0.693

FEATURES_TO_PLOT = [
    "puncta_count",
    "puncta_area_sum_um2",
    "puncta_area_proportion",
    "puncta_intensity_mean",
    "cell_coi1_intensity_mean",
]

plt.rcParams.update({
    "font.size": 14,
    "svg.fonttype": "none"
})

sns.set_palette("Paired")


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


# -------------------------
# PRINT N
# -------------------------

cell_n_summary = (
    df.groupby(["peptide", "cell"])
    .agg(
        total_cells=("cell_number", "count"),
        n_reps=("rep", "nunique")
    )
    .reset_index()
)

cell_n_summary["peptide"] = pd.Categorical(
    cell_n_summary["peptide"],
    categories=PEPTIDE_ORDER,
    ordered=True
)

cell_n_summary["cell"] = pd.Categorical(
    cell_n_summary["cell"],
    categories=CELL_ORDER,
    ordered=True
)

cell_n_summary = cell_n_summary.sort_values(["peptide", "cell"])

print("\nTOTAL CELL N AND REP N")
print(cell_n_summary.to_string(index=False))

cell_n_summary.to_csv(
    os.path.join(output_folder, "total_cell_and_rep_N_summary.csv"),
    index=False
)


# -------------------------
# REP AVERAGES
# -------------------------

available_features = [
    feature for feature in FEATURES_TO_PLOT
    if feature in df.columns
]

rep_summary = (
    df.groupby(["peptide", "cell", "rep"], as_index=False)
    .agg({feature: "mean" for feature in available_features})
)

rep_summary.to_csv(
    os.path.join(output_folder, "rep_level_raw_feature_means.csv"),
    index=False
)

rep_n_summary = (
    rep_summary.groupby(["peptide", "cell"])
    .agg(N_reps=("rep", "nunique"))
    .reset_index()
)

print("\nREP-LEVEL N")
print(rep_n_summary.to_string(index=False))

rep_n_summary.to_csv(
    os.path.join(output_folder, "rep_level_N_summary.csv"),
    index=False
)


# -------------------------
# PLOT RAW VALUES
# -------------------------

for feature in available_features:
    plot_df = df.dropna(subset=[feature, "peptide", "cell"]).copy()
    rep_plot_df = rep_summary.dropna(subset=[feature, "peptide", "cell"]).copy()

    if plot_df.empty:
        continue

    plt.figure(figsize=(11, 6))

    ax = sns.boxplot(
        data=plot_df,
        x="peptide",
        y=feature,
        hue="cell",
        order=PEPTIDE_ORDER,
        hue_order=PLOT_CELL_ORDER,
        showfliers=False,
        width=0.7,
        zorder=1,
    )

    # individual cell datapoints
    sns.stripplot(
        data=plot_df,
        x="peptide",
        y=feature,
        hue="cell",
        order=PEPTIDE_ORDER,
        hue_order=PLOT_CELL_ORDER,
        dodge=True,
        jitter=0.25,
        size=2,
        alpha=0.25,
        linewidth=0,
        zorder=2,
        ax=ax,
    )

    # rep-average datapoints
    sns.stripplot(
        data=rep_plot_df,
        x="peptide",
        y=feature,
        hue="cell",
        order=PEPTIDE_ORDER,
        hue_order=PLOT_CELL_ORDER,
        dodge=True,
        jitter=False,
        marker="D",
        edgecolor="black",
        linewidth=0.8,
        size=8,
        alpha=1,
        zorder=3,
        ax=ax,
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[:len(PLOT_CELL_ORDER)],
        labels[:len(PLOT_CELL_ORDER)],
        title="Cell"
    )

    sns.despine()
    plt.title(f"{feature} raw values")
    plt.xlabel("Peptide")
    plt.ylabel(feature)
    plt.xticks(rotation=35)
    plt.tight_layout()

    save_base = f"{feature}_raw_box_cells_rep_means"

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


logger.info("done.")