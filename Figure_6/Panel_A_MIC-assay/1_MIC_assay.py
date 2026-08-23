import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# -------------------------
# Output folder
# -------------------------
output_dir = "results/round2"
os.makedirs(output_dir, exist_ok=True)

plt.rcParams["svg.fonttype"] = "none"

# -------------------------
# Read CSV
# -------------------------
# Update filename as needed
df = pd.read_csv("26016B_mic_assay-r2.csv")
df.columns = df.columns.str.strip()

# Expected columns:
#   ldl_concentration, peptide_concentration, rep1, rep2, rep3

# -------------------------
# Melt replicates to long format
# -------------------------
df_long = df.melt(
    id_vars=["ldl_concentration", "peptide_concentration"],
    value_vars=["rep1", "rep2", "rep3"],
    var_name="rep",
    value_name="od"
).dropna()


buffer = 0.049 

df_long['buffer_subtracted'] = df_long['od'] - 0.049


# -------------------------
# Choose concentrations to keep
# Use None to keep all
# -------------------------

PEPTIDE_CONCENTRATIONS_TO_KEEP = [0, 4, 8, 16, 24]
LDL_CONCENTRATIONS_TO_KEEP = [0, 125, 250, 500]


# Ensure numeric
df_long["ldl_concentration"] = pd.to_numeric(df_long["ldl_concentration"], errors="coerce")
df_long["peptide_concentration"] = pd.to_numeric(df_long["peptide_concentration"], errors="coerce")
df_long["buffer_subtracted"] = pd.to_numeric(df_long["buffer_subtracted"], errors="coerce")
df_long = df_long.dropna(subset=["ldl_concentration", "peptide_concentration", "buffer_subtracted"])


# -------------------------
# Optional concentration filtering
# -------------------------

if PEPTIDE_CONCENTRATIONS_TO_KEEP is not None:
    df_long = df_long[
        df_long["peptide_concentration"].isin(PEPTIDE_CONCENTRATIONS_TO_KEEP)
    ].copy()

if LDL_CONCENTRATIONS_TO_KEEP is not None:
    df_long = df_long[
        df_long["ldl_concentration"].isin(LDL_CONCENTRATIONS_TO_KEEP)
    ].copy()

print("Peptide concentrations kept:", sorted(df_long["peptide_concentration"].unique()))
print("LDL concentrations kept:", sorted(df_long["ldl_concentration"].unique()))


# -------------------------
# Control normalization: (LDL=0, Peptide=0) is "normal growth"
# -------------------------
# Mean of replicate means at the global control condition
control_mask = (df_long["ldl_concentration"] == 0) & (df_long["peptide_concentration"] == 0)
if df_long.loc[control_mask].empty:
    raise ValueError("Could not find control condition where ldl_concentration==0 and peptide_concentration==0.")

control_od = df_long.loc[control_mask, "buffer_subtracted"].mean()

# Percent growth vs global control
df_long["normalized_to_control"] = df_long["buffer_subtracted"] / control_od

# Mean per condition (for heatmap)
means = (
    df_long.groupby(["ldl_concentration", "peptide_concentration"])["normalized_to_control"]
    .mean()
    .reset_index()
)

# Pivot to matrix form for heatmap
heatmap = means.pivot(
    index="ldl_concentration",
    columns="peptide_concentration",
    values="normalized_to_control"
)

# Flip LDL direction (high at top)
heatmap = heatmap.sort_index(ascending=False)

# Sort peptide concentrations ascending along x
heatmap = heatmap.reindex(sorted(heatmap.columns), axis=1)

# -------------------------
# Plot: tight capped scale, centered at 100%
# -------------------------



plt.figure(figsize=(6.2, 5.2))
im = plt.imshow(
    heatmap.values,
    aspect="auto",
    cmap="coolwarm"
)

cbar = plt.colorbar(im)
cbar.set_label("OD-buffer/Control")

plt.xticks(
    np.arange(len(heatmap.columns)),
    [str(c) for c in heatmap.columns]
)
plt.yticks(
    np.arange(len(heatmap.index)),
    [str(i) for i in heatmap.index]
)

plt.xlabel("Peptide concentration")
plt.ylabel("LDL concentration")
plt.title("OD-Buffer/Control")

plt.tight_layout()

outpath = os.path.join(output_dir, "control_normalized_heatmap_tight.png")
plt.savefig(outpath, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved: {outpath}")

# ============================================================
# EXTRA FIGURE: Normalize within each peptide dose to LDL = 0
# (So for each peptide concentration, LDL=0 is 100%)
# ============================================================

# # Mean OD per condition first (averaging reps)
# means_raw = (
#     df_long.groupby(["ldl_concentration", "peptide_concentration"])["od"]
#     .mean()
#     .reset_index()
# )

# # Baseline for each peptide concentration = LDL 0 at that same peptide dose
# baseline_pep = means_raw[means_raw["ldl_concentration"] == 0][
#     ["peptide_concentration", "od"]
# ].rename(columns={"od": "baseline_od"})

# if baseline_pep.empty:
#     raise ValueError("Could not find any baseline rows where ldl_concentration == 0.")

# # Merge baselines onto all LDL conditions by peptide concentration
# means_raw = means_raw.merge(baseline_pep, on="peptide_concentration", how="left")

# # % growth vs LDL=0 at each peptide concentration
# means_raw["pct_vs_no_ldl_at_pep"] = 100.0 * means_raw["od"] / means_raw["baseline_od"]

# # Pivot to heatmap
# heatmap2 = means_raw.pivot(
#     index="ldl_concentration",
#     columns="peptide_concentration",
#     values="pct_vs_no_ldl_at_pep"
# )

# # Flip LDL direction (high at top) and sort peptide axis
# heatmap2 = heatmap2.sort_index(ascending=False)
# heatmap2 = heatmap2.reindex(sorted(heatmap2.columns), axis=1)

# # Plot settings (edit if you want different contrast)
# vmin2 = 60
# vcenter2 = 100
# vmax2 = 160
# norm2 = mcolors.TwoSlopeNorm(vmin=vmin2, vcenter=vcenter2, vmax=vmax2)

# plt.figure(figsize=(6.2, 5.2))
# im2 = plt.imshow(
#     heatmap2.values,
#     aspect="auto",
#     norm=norm2,
#     cmap="coolwarm"
# )

# cbar2 = plt.colorbar(im2)
# cbar2.set_label("% Growth vs LDL=0 (within each peptide dose)")

# plt.xticks(
#     np.arange(len(heatmap2.columns)),
#     [str(c) for c in heatmap2.columns]
# )
# plt.yticks(
#     np.arange(len(heatmap2.index)),
#     [str(i) for i in heatmap2.index]
# )

# plt.xlabel("Peptide concentration")
# plt.ylabel("LDL concentration")
# plt.title("LDL Effect vs No-LDL Baseline (Per Peptide Dose)")

# plt.tight_layout()

# outpath2 = os.path.join(output_dir, "peptide_normalized_vs_no_ldl_heatmap.png")
# plt.savefig(outpath2, dpi=300, bbox_inches="tight")
# plt.show()

# print(f"Saved: {outpath2}")


# ============================================================
# EXTRA FIGURE: Line plot — LDL effect per peptide dose
# (% growth vs LDL=0 baseline at each peptide concentration)
# ============================================================

plt.figure(figsize=(7,5))

color_pallete = [
    "#081F3A",
    "#1B3A5F", 
    "#2F6690",
    "#6CA6CD",
    "#C6DBEF"]

# remove peptide concentration 10 because some type
# of techincal error 
df_pep_norm = df_long.copy()
df_pep_norm["peptide_concentration"] = df_pep_norm["peptide_concentration"].astype(int)


# Mean ± SD
agg_line = (
    df_pep_norm.groupby(["peptide_concentration", "ldl_concentration"])["normalized_to_control"]
    .agg(mean="mean", sd="std")
    .reset_index()
)

# pep order for x-axis
pep_order = sorted(df_pep_norm["peptide_concentration"].unique())
xpos = {c: i for i, c in enumerate(pep_order)}

# Plot a line for each LDL concentration
for i, ldl in enumerate(sorted(agg_line["ldl_concentration"].unique())):
    sub = agg_line[agg_line["ldl_concentration"] == ldl].copy()
    sub["x"] = sub["peptide_concentration"].map(xpos)
    sub = sub.sort_values("x")

    plt.errorbar(
        sub["x"],
        sub["mean"],
        yerr=sub["sd"],
        fmt="-o",
        capsize=3,
        markersize=5,
        color=color_pallete[i],
        label=f"{ldl} ug/ml"
    )
    
plt.axhline(1, linestyle="--", linewidth=1)

plt.xticks(range(len(pep_order)), [str(c) for c in pep_order])
plt.xlabel("LL37 Concentration (uM)")
plt.ylabel("OD-buffer/Control")
plt.title("LDL Effect Across Peptide Concentrations")
plt.legend(title="LDL", frameon=False)

plt.tight_layout()

outpath3 = os.path.join(output_dir, "ldl_effect_normalized_lineplot_simplified.png")
plt.savefig(outpath3, dpi=300, bbox_inches="tight")
# plt.show()

# SVG
outpath_svg = os.path.join(output_dir, "ldl_effect_normalized_lineplot_simplified.svg")
plt.savefig(outpath_svg, bbox_inches="tight")

print(f"Saved: {outpath3}")