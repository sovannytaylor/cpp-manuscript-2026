import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# =========================
# EDIT THIS PATH
# =========================
BASE_DIR = Path(r"C:\Users\u244278\OneDrive - Baylor College of Medicine\Boeynaems, Steven's files - Boeynaems_lab_drive\People Data\Sophie\Papers\Cell-penetrant cationic peptides hijack LDLR\Figure 6 - biophysics\MST-06012026")

OUTPUT_DIR = BASE_DIR / "MST_summary_figures"
OUTPUT_DIR.mkdir(exist_ok=True)


def read_mst_csv(path):
    """Reads MST csv with flexible separators."""
    try:
        return pd.read_csv(path, sep=";", encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="utf-8-sig")


def is_outlier_mask(df):
    if "Is Outlier" not in df.columns:
        return pd.Series(False, index=df.index)

    return df["Is Outlier"].astype(str).str.lower().isin(["true", "1", "yes"])


def find_file(files, pattern):
    """Return first file matching regex pattern."""
    matches = [f for f in files if re.search(pattern, f.name, re.IGNORECASE)]
    return matches[0] if matches else None


def plot_experiment(exp_dir):
    csvs = [f for f in exp_dir.iterdir() if f.suffix.lower() == ".csv"]

    if not csvs:
        print(f"Skipping {exp_dir.name}: no csv files")
        return

    exp_num = re.sub(r"\D", "", exp_dir.name)

    files = {
        "650_initial": find_file(csvs, rf"{exp_num}_?650.*initial"),
        "670_initial": find_file(csvs, rf"{exp_num}_?670.*initial"),
        "ratio": find_file(csvs, rf"{exp_num}_?670-650"),
    }

    times = ["1.5sec", "2.5sec", "5sec", "10sec"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    # -------------------------
    # A. Initial fluorescence QC
    # -------------------------
    ax = axes[0]

    for label, key in [("650 nm", "650_initial"), ("670 nm", "670_initial")]:
        file = files[key]
        if file is None:
            continue

        df = read_mst_csv(file)
        df = df[~is_outlier_mask(df)]

        if {"Ligand Concentration [M]", "Raw Fluorescence [counts]"}.issubset(df.columns):
            ax.scatter(
                df["Ligand Concentration [M]"],
                df["Raw Fluorescence [counts]"],
                label=label
            )

    ax.set_xscale("log")
    ax.set_xlabel("Ligand concentration [M]")
    ax.set_ylabel("Raw fluorescence [counts]")
    ax.set_title("A. Initial fluorescence QC")
    ax.legend(frameon=False)

    # -------------------------
    # B. 670/650 ratio QC
    # -------------------------
    ax = axes[1]
    ratio_file = files["ratio"]

    if ratio_file is not None:
        df = read_mst_csv(ratio_file)
        df = df[~is_outlier_mask(df)]

        ratio_col = None
        for col in df.columns:
            if "ratio" in col.lower() and "670" in col.lower() and "650" in col.lower():
                ratio_col = col
                break

        if ratio_col and "Ligand Concentration [M]" in df.columns:
            ax.scatter(df["Ligand Concentration [M]"], df[ratio_col])

    ax.set_xscale("log")
    ax.set_xlabel("Ligand concentration [M]")
    ax.set_ylabel("Ratio 670 nm / 650 nm")
    ax.set_title("B. Fluorescence ratio QC")

    # -------------------------
    # C-F. MST Fnorm curves
    # -------------------------
    for i, time in enumerate(times):
        ax = axes[i + 2]

        for ch in ["650", "670"]:
            file = find_file(csvs, rf"{exp_num}_?{ch}.*OT.*{time}")

            if file is None:
                continue

            df = read_mst_csv(file)
            df = df[~is_outlier_mask(df)]

            required_raw = {"Ligand Concentration [M]", "Fnorm [‰]"}
            required_fit = {"Ligand Concentration [M] (fitted)", "Fnorm [‰] (fitted)"}

            if required_raw.issubset(df.columns):
                raw = df.dropna(subset=["Ligand Concentration [M]", "Fnorm [‰]"])

                if "Capillary Position" in raw.columns:
                    raw = raw[raw["Capillary Position"].notna()]

                ax.scatter(
                    raw["Ligand Concentration [M]"],
                    raw["Fnorm [‰]"],
                    label=f"{ch} nm data"
                )

            if required_fit.issubset(df.columns):
                fit = df.dropna(subset=["Ligand Concentration [M] (fitted)", "Fnorm [‰] (fitted)"])

                ax.plot(
                    fit["Ligand Concentration [M] (fitted)"],
                    fit["Fnorm [‰] (fitted)"],
                    label=f"{ch} nm fit"
                )

        ax.set_xscale("log")
        ax.set_xlabel("Ligand concentration [M]")
        ax.set_ylabel("Fnorm [‰]")
        ax.set_title(f"MST binding curve, {time.replace('sec', ' s')}")
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle(exp_dir.name, fontsize=16)
    plt.tight_layout()

    out_png = OUTPUT_DIR / f"{exp_dir.name}_MST_summary.png"
    out_pdf = OUTPUT_DIR / f"{exp_dir.name}_MST_summary.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_png}")


# =========================
# RUN ALL EXPERIMENTS
# =========================
experiment_dirs = [
    d for d in BASE_DIR.iterdir()
    if d.is_dir() and d.name.lower().startswith("experiment")
]

for exp_dir in experiment_dirs:
    plot_experiment(exp_dir)

print("Done.")