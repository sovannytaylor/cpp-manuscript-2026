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
output_folder = "normalized_KO_OE_plots"
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

FEATURES_TO_PLOT = [
    "puncta_count",
    "puncta_area_sum_um2",
    "puncta_area_proportion",
    "puncta_intensity_mean",
    "cell_coi1_intensity_mean",
]

NORMALIZE_TO = "WT"

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
    df = read_table(path)
    df["source_file"] = file
    dfs.append(df)

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

df["peptide"] = df["peptide"].astype(str).str.strip()

df["peptide"] = df["peptide"].replace({
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


SCALE_PX = 0.69  # use the correct pixel size for this dataset

df["puncta_area_sum_um2"] = (
    df["puncta_area_sum"] * (SCALE_PX ** 2)
)

# -------------------------
# PRINT TOTAL CELL N AND REP N
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
# COLLAPSE TO ONE POINT PER REP
# -------------------------

rep_summary = (
    df.groupby(["peptide", "cell", "rep"], as_index=False)
    .agg({feature: "mean" for feature in FEATURES_TO_PLOT if feature in df.columns})
)

rep_n_summary = (
    rep_summary.groupby(["peptide", "cell"])
    .agg(N_reps=("rep", "nunique"))
    .reset_index()
)

print("\nREP-LEVEL N")
print(rep_n_summary.to_string(index=False))

rep_summary.to_csv(
    os.path.join(output_folder, "rep_level_raw_feature_means.csv"),
    index=False
)


# -------------------------
# NORMALIZE TO WT WITHIN SAME PEPTIDE + REP
# -------------------------

normalized_dfs = []

for feature in FEATURES_TO_PLOT:
    if feature not in rep_summary.columns:
        logger.warning(f"Skipping missing feature: {feature}")
        continue

    temp = rep_summary[["peptide", "cell", "rep", feature]].copy()

    wt = temp[temp["cell"] == NORMALIZE_TO][
        ["peptide", "rep", feature]
    ].rename(columns={feature: f"{feature}_WT"})

    temp = temp.merge(wt, on=["peptide", "rep"], how="left")

    temp[f"{feature}_normalized_to_WT"] = (
        temp[feature] / temp[f"{feature}_WT"]
    )

    temp["feature"] = feature
    temp["raw_value"] = temp[feature]
    temp["WT_value_for_same_rep"] = temp[f"{feature}_WT"]
    temp["normalized_value"] = temp[f"{feature}_normalized_to_WT"]

    normalized_dfs.append(
        temp[
            [
                "peptide",
                "cell",
                "rep",
                "feature",
                "raw_value",
                "WT_value_for_same_rep",
                "normalized_value",
            ]
        ]
    )

norm_df = pd.concat(normalized_dfs, ignore_index=True)

norm_df = norm_df[
    norm_df["cell"].isin(PLOT_CELL_ORDER)
].copy()

norm_df["peptide"] = pd.Categorical(
    norm_df["peptide"],
    categories=PEPTIDE_ORDER,
    ordered=True
)

norm_df["cell"] = pd.Categorical(
    norm_df["cell"],
    categories=PLOT_CELL_ORDER,
    ordered=True
)

norm_df.to_csv(
    os.path.join(output_folder, "rep_level_normalized_to_WT.csv"),
    index=False
)


# -------------------------
# PLOT BOX + REP POINTS
# -------------------------

for feature in FEATURES_TO_PLOT:
    plot_df = norm_df[norm_df["feature"] == feature].copy()

    if plot_df.empty:
        continue

    plt.figure(figsize=(10, 6))

    ax = sns.boxplot(
        data=plot_df,
        x="peptide",
        y="normalized_value",
        hue="cell",
        order=PEPTIDE_ORDER,
        hue_order=PLOT_CELL_ORDER,
        showfliers=False,
        zorder=1,
    )

    sns.stripplot(
        data=plot_df,
        x="peptide",
        y="normalized_value",
        hue="cell",
        order=PEPTIDE_ORDER,
        hue_order=PLOT_CELL_ORDER,
        dodge=True,
        edgecolor="black",
        linewidth=0.4,
        size=7,
        alpha=0.85,
        jitter=True,
        zorder=2,
        ax=ax,
    )

    ax.axhline(1, linestyle="--", linewidth=1)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[:len(PLOT_CELL_ORDER)],
        labels[:len(PLOT_CELL_ORDER)],
        title="Cell"
    )

    sns.despine()
    plt.title(f"{feature} normalized to WT by matched rep")
    plt.xlabel("Peptide")
    plt.ylabel(f"{feature} / WT same rep")
    plt.xticks(rotation=35)
    plt.tight_layout()

    save_base = f"{feature}_KO_OE_normalized_to_WT_boxplot"

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