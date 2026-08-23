import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# Config
# -----------------------------
output_dir = "results/results_2"
os.makedirs(output_dir, exist_ok=True)

PARAMS = ["zp", "mob", "cond"]

PEPTIDE_COLORS = {
    "GR30": "#1351AF",
    "CROTAMINE": "#D46C26",
    "GP30": "#C820A6",
    "LL37": "#CF1A1A",
}

DEFAULT_COLOR = "gray"

# Keep SVG text editable in Illustrator/Inkscape
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "Arial"


def save_plot(filename_base, dpi=300, bbox_inches=None):
    """Save current matplotlib figure as both PNG and SVG."""
    plt.savefig(
        f"{output_dir}/{filename_base}.png",
        dpi=dpi,
        bbox_inches=bbox_inches
    )
    plt.savefig(
        f"{output_dir}/{filename_base}.svg",
        bbox_inches=bbox_inches
    )


# -----------------------------
# Read in
# -----------------------------
df = pd.read_csv("26031_zeta_potential.csv")
df.columns = df.columns.str.strip()

control_df = df[df["record"].between(844, 853)].copy()
experimental_df = df[~df["record"].between(844, 853)].copy()

peptides = df["peptide"].unique()

# -----------------------------------------
# Individual plots of all parameters
# -----------------------------------------
for peptide in peptides:

    color = PEPTIDE_COLORS.get(peptide, DEFAULT_COLOR)
    sub = df[df["peptide"] == peptide]

    for param in PARAMS:

        plt.figure()

        plt.scatter(
            sub["peptide_concentration"],
            sub[param],
            color=color,
            alpha=0.3
        )

        stats_df = (
            sub.groupby("peptide_concentration")[param]
            .agg(["mean", "std"])
            .reset_index()
        )

        plt.errorbar(
            stats_df["peptide_concentration"],
            stats_df["mean"],
            yerr=stats_df["std"],
            marker="o",
            linewidth=2,
            capsize=4,
            color=color
        )

        plt.title(f"{peptide} — {param}")
        plt.xlabel("Peptide Concentration")
        plt.ylabel(param)

        plt.tight_layout()
        save_plot(f"{peptide}_{param}")
        plt.close()

# -----------------------------------------
# Peptides on the same plot
# -----------------------------------------
plt.figure()

plot_df = experimental_df.copy()
peptides = plot_df["peptide"].unique()

for peptide in peptides:

    color = PEPTIDE_COLORS.get(peptide, DEFAULT_COLOR)
    sub = plot_df[plot_df["peptide"] == peptide]

    plt.scatter(
        sub["peptide_concentration"],
        sub["zp"],
        color=color,
        alpha=0.25
    )

    stats_df = (
        sub.groupby("peptide_concentration")["zp"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    stats_df["sem"] = stats_df["std"] / np.sqrt(stats_df["count"])

    plt.errorbar(
        stats_df["peptide_concentration"],
        stats_df["mean"],
        yerr=stats_df["sem"],
        marker="o",
        linewidth=2,
        capsize=4,
        color=color,
        label=peptide
    )

plt.xlabel("Peptide Concentration (nM)")
plt.ylabel("Zeta Potential (mV)")
plt.title("ZP vs Peptide Concentration")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
save_plot("ALL_PEPTIDES_ZP_overlay", bbox_inches="tight")
plt.close()

# -----------------------------------------
# Exclude the last concentration
# -----------------------------------------
plt.figure()

plot_df = experimental_df.copy()
plot_df = plot_df[plot_df["peptide_concentration"] != 37000]
peptides = plot_df["peptide"].unique()

for peptide in peptides:

    color = PEPTIDE_COLORS.get(peptide, DEFAULT_COLOR)
    sub = plot_df[plot_df["peptide"] == peptide]

    if sub.empty:
        continue

    plt.scatter(
        sub["peptide_concentration"],
        sub["zp"],
        color=color,
        alpha=0.25
    )

    stats_df = (
        sub.groupby("peptide_concentration")["zp"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    stats_df["sem"] = stats_df["std"] / np.sqrt(stats_df["count"])

    plt.errorbar(
        stats_df["peptide_concentration"],
        stats_df["mean"],
        yerr=stats_df["sem"],
        marker="o",
        linewidth=2,
        capsize=4,
        color=color,
        label=peptide
    )

plt.xlabel("Peptide Concentration (nM)")
plt.ylabel("Zeta Potential (mV)")
plt.title("ZP vs Peptide Concentration")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
save_plot("ALL_PEPTIDES_ZP_overlay_no37000", bbox_inches="tight")
plt.close()

# -----------------------------------------
# Plotting controls
# -----------------------------------------
plt.figure()

plot_df = control_df.copy()
plot_df = plot_df[plot_df["peptide"].isin(["LDL-DIL", "LDL"])]

order = ["LDL", "LDL-DIL"]

for peptide in order:

    color = PEPTIDE_COLORS.get(peptide, DEFAULT_COLOR)
    sub = plot_df[plot_df["peptide"] == peptide]

    x = [peptide] * len(sub)

    plt.scatter(
        x,
        sub["zp"],
        color=color,
        alpha=0.4
    )

stats_df = (
    plot_df.groupby("peptide")["zp"]
    .agg(["mean", "std", "count"])
    .reindex(order)
    .reset_index()
)

stats_df["sem"] = stats_df["std"] / np.sqrt(stats_df["count"])

for _, row in stats_df.iterrows():

    color = PEPTIDE_COLORS.get(row["peptide"], DEFAULT_COLOR)

    plt.scatter(
        row["peptide"],
        row["mean"],
        color=color,
        s=100
    )

    plt.errorbar(
        row["peptide"],
        row["mean"],
        yerr=row["sem"],
        color=color,
        capsize=5,
        linestyle="none"
    )

plt.xlabel("Peptide")
plt.ylabel("Zeta Potential (mV)")
plt.title("Control ZP — LDL vs LDL-DIL")

plt.tight_layout()
save_plot("LDL_vs_LDL-DIL_dotplot")
plt.close()

# -----------------------------------------
# Symlog scale
# -----------------------------------------
plt.figure()

plot_df = experimental_df.copy()
peptides = plot_df["peptide"].unique()

for peptide in peptides:

    color = PEPTIDE_COLORS.get(peptide, DEFAULT_COLOR)
    sub = plot_df[plot_df["peptide"] == peptide]

    plt.scatter(
        sub["peptide_concentration"],
        sub["zp"],
        color=color,
        alpha=0.25
    )

    stats_df = (
        sub.groupby("peptide_concentration")["zp"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    stats_df["sem"] = stats_df["std"] / np.sqrt(stats_df["count"])

    plt.errorbar(
        stats_df["peptide_concentration"],
        stats_df["mean"],
        yerr=stats_df["sem"],
        marker="o",
        linewidth=2,
        capsize=4,
        color=color,
        label=peptide
    )

plt.xscale("symlog", linthresh=100)

plt.xlabel("Peptide Concentration (nM) (symlog scale)")
plt.ylabel("Zeta Potential (mV)")
plt.title("ZP vs Peptide Concentration")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
save_plot("ALL_PEPTIDES_ZP_overlay_symlog", bbox_inches="tight")
plt.close()

# -----------------------------------------
# Log-ish symlog scale
# NOTE: matplotlib uses "symlog", not "logsym"
# -----------------------------------------
plt.figure()

plot_df = experimental_df.copy()
peptides = plot_df["peptide"].unique()

for peptide in peptides:

    color = PEPTIDE_COLORS.get(peptide, DEFAULT_COLOR)
    sub = plot_df[plot_df["peptide"] == peptide]

    plt.scatter(
        sub["peptide_concentration"],
        sub["zp"],
        color=color,
        alpha=0.25
    )

    stats_df = (
        sub.groupby("peptide_concentration")["zp"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    stats_df["sem"] = stats_df["std"] / np.sqrt(stats_df["count"])

    plt.errorbar(
        stats_df["peptide_concentration"],
        stats_df["mean"],
        yerr=stats_df["sem"],
        marker="o",
        linewidth=2,
        capsize=4,
        color=color,
        label=peptide
    )

plt.xscale("symlog", linthresh=1)

plt.xlabel("Peptide Concentration (nM) (symlog scale)")
plt.ylabel("Zeta Potential (mV)")
plt.title("ZP vs Peptide Concentration")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
save_plot("ALL_PEPTIDES_ZP_overlay_log", bbox_inches="tight")
plt.close()