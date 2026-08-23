import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# =========================
# EDIT THIS PATH
# =========================
BASE_DIR = Path(
    r"C:\Users\u244278\OneDrive - Baylor College of Medicine\Boeynaems, Steven's files - Boeynaems_lab_drive\People Data\Sophie\Papers\Cell-penetrant cationic peptides hijack LDLR\Figure 6 - biophysics\MST-06012026"
)

OUTPUT_DIR = BASE_DIR / "MST_summary_figures"
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================
# SETTINGS
# =========================
TIMES = ["1.5sec", "2.5sec", "5sec", "10sec"]
CHANNELS = ["650", "670"]

CONC_COL = "Ligand Concentration [M]"
FNORM_COL = "Fnorm [‰]"


def read_mst_csv(path):
    try:
        return pd.read_csv(path, sep=";", encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="utf-8-sig")


def is_outlier_mask(df):
    if "Is Outlier" not in df.columns:
        return pd.Series(False, index=df.index)

    return df["Is Outlier"].astype(str).str.lower().isin(["true", "1", "yes"])


def find_file(files, pattern):
    matches = [f for f in files if re.search(pattern, f.name, re.IGNORECASE)]
    return matches[0] if matches else None


def binding_model(x, bottom, top, kd):
    return bottom + (top - bottom) * x / (kd + x)


def fit_kd(summary_df, y_col):
    fit_df = summary_df.dropna(subset=["concentration_M", y_col]).copy()
    fit_df = fit_df[fit_df["concentration_M"] > 0]

    if len(fit_df) < 4:
        return None

    x = fit_df["concentration_M"].values.astype(float)
    y = fit_df[y_col].values.astype(float)

    bottom_guess = np.nanmin(y)
    top_guess = np.nanmax(y)
    kd_guess = np.nanmedian(x)

    bounds = (
        [-np.inf, -np.inf, np.min(x) / 100],
        [np.inf, np.inf, np.max(x) * 100],
    )

    try:
        popt, pcov = curve_fit(
            binding_model,
            x,
            y,
            p0=[bottom_guess, top_guess, kd_guess],
            bounds=bounds,
            maxfev=10000,
        )

        bottom, top, kd = popt

        x_fit = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), 300)
        y_fit = binding_model(x_fit, bottom, top, kd)

        return {
            "bottom": bottom,
            "top": top,
            "kd_M": kd,
            "kd_nM": kd * 1e9,
            "x_fit": x_fit,
            "y_fit": y_fit,
        }

    except Exception as e:
        print(f"Fit failed: {e}")
        return None


def get_peptide_dirs():
    peptide_dirs = [
        d for d in BASE_DIR.iterdir()
        if d.is_dir()
        and d.name != OUTPUT_DIR.name
        and not d.name.lower().startswith("experiment")
    ]

    print("\nPeptide folders found:")
    for d in peptide_dirs:
        print(f"  {d.name}")

    return peptide_dirs


def collect_mst_data(peptide_dir, time, channel):
    all_rows = []

    experiment_dirs = [
        d for d in peptide_dir.iterdir()
        if d.is_dir() and d.name.lower().startswith("experiment")
    ]

    print(f"\n{peptide_dir.name}: experiments found inside folder:")
    for d in experiment_dirs:
        print(f"  {d.name}")

    for exp_dir in experiment_dirs:
        csvs = [f for f in exp_dir.iterdir() if f.suffix.lower() == ".csv"]

        if not csvs:
            continue

        exp_num = re.sub(r"\D", "", exp_dir.name)
        file = find_file(csvs, rf"{exp_num}_?{channel}.*OT.*{time}")

        if file is None:
            continue

        df = read_mst_csv(file)
        df = df[~is_outlier_mask(df)].copy()

        if not {CONC_COL, FNORM_COL}.issubset(df.columns):
            continue

        raw = df.dropna(subset=[CONC_COL, FNORM_COL]).copy()

        if "Capillary Position" in raw.columns:
            raw = raw[raw["Capillary Position"].notna()]

        raw["peptide"] = peptide_dir.name
        raw["experiment"] = exp_dir.name
        raw["time"] = time
        raw["channel"] = channel
        raw["concentration_M"] = raw[CONC_COL].astype(float)
        raw["fnorm"] = raw[FNORM_COL].astype(float)

        all_rows.append(
            raw[
                [
                    "peptide",
                    "experiment",
                    "time",
                    "channel",
                    "concentration_M",
                    "fnorm",
                ]
            ]
        )

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


def collect_spectral_shift_data(peptide_dir):
    all_rows = []

    experiment_dirs = [
        d for d in peptide_dir.iterdir()
        if d.is_dir() and d.name.lower().startswith("experiment")
    ]

    for exp_dir in experiment_dirs:
        csvs = [f for f in exp_dir.iterdir() if f.suffix.lower() == ".csv"]

        if not csvs:
            continue

        exp_num = re.sub(r"\D", "", exp_dir.name)
        ratio_file = find_file(csvs, rf"{exp_num}_?670-650")

        if ratio_file is None:
            continue

        df = read_mst_csv(ratio_file)
        df = df[~is_outlier_mask(df)].copy()

        ratio_col = None
        for col in df.columns:
            if "ratio" in col.lower() and "670" in col.lower() and "650" in col.lower():
                ratio_col = col
                break

        if ratio_col is None or CONC_COL not in df.columns:
            continue

        raw = df.dropna(subset=[CONC_COL, ratio_col]).copy()

        if "Capillary Position" in raw.columns:
            raw = raw[raw["Capillary Position"].notna()]

        raw["peptide"] = peptide_dir.name
        raw["experiment"] = exp_dir.name
        raw["concentration_M"] = raw[CONC_COL].astype(float)
        raw["spectral_shift"] = raw[ratio_col].astype(float)

        all_rows.append(
            raw[
                [
                    "peptide",
                    "experiment",
                    "concentration_M",
                    "spectral_shift",
                ]
            ]
        )

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


def summarize_replicates(df, value_col):
    summary = (
        df.groupby("concentration_M", as_index=False)
        .agg(
            mean_value=(value_col, "mean"),
            sd_value=(value_col, "std"),
            n=(value_col, "count"),
        )
    )

    summary["sem_value"] = summary["sd_value"] / np.sqrt(summary["n"])
    return summary.sort_values("concentration_M")


def plot_spectral_shift(peptide_dir):
    df = collect_spectral_shift_data(peptide_dir)

    if df.empty:
        print(f"No spectral shift data found for {peptide_dir.name}")
        return None

    summary = summarize_replicates(df, "spectral_shift")
    fit = fit_kd(summary, "mean_value")

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(
        df["concentration_M"],
        df["spectral_shift"],
        s=30,
        alpha=0.30,
        color="C1",
        zorder=1,
        label="Individual points",
    )

    ax.errorbar(
        summary["concentration_M"],
        summary["mean_value"],
        yerr=summary["sem_value"],
        fmt="o",
        color="C1",
        ecolor="C1",
        capsize=4,
        markersize=8,
        linewidth=2,
        zorder=3,
        label="Mean ± SEM",
    )

    if fit is not None:
        ax.plot(
            fit["x_fit"],
            fit["y_fit"],
            color="C1",
            linewidth=3,
            zorder=2,
            label=f"Kd = {fit['kd_nM']:.2f} nM",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Ligand concentration [M]")
    ax.set_ylabel("670/650 Spectral Shift")
    ax.set_title(f"{peptide_dir.name} Spectral Shift")
    ax.legend(frameon=False)

    plt.tight_layout()

    out_png = OUTPUT_DIR / f"{peptide_dir.name}_spectral_shift_mean_SEM_Kd.png"
    out_pdf = OUTPUT_DIR / f"{peptide_dir.name}_spectral_shift_mean_SEM_Kd.pdf"
    out_svg = OUTPUT_DIR / f"{peptide_dir.name}_spectral_shift_mean_SEM_Kd.svg"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved spectral shift: {out_png}")

    return {
        "peptide": peptide_dir.name,
        "metric": "Spectral Shift",
        "channel": "670/650",
        "time": "NA",
        "kd_M": fit["kd_M"] if fit is not None else np.nan,
        "kd_nM": fit["kd_nM"] if fit is not None else np.nan,
        "bottom": fit["bottom"] if fit is not None else np.nan,
        "top": fit["top"] if fit is not None else np.nan,
        "n_points": len(df),
        "n_concentrations": summary["concentration_M"].nunique(),
    }


def plot_peptide(peptide_dir):
    print(f"\nProcessing peptide: {peptide_dir.name}")

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=False)
    axes = axes.flatten()

    kd_rows = []
    plot_i = 0

    for channel in CHANNELS:
        for time in TIMES:
            ax = axes[plot_i]
            plot_i += 1

            df = collect_mst_data(peptide_dir, time, channel)

            if df.empty:
                ax.set_title(f"{channel} nm, {time}: no data")
                ax.axis("off")
                continue

            summary = summarize_replicates(df, "fnorm")
            fit = fit_kd(summary, "mean_value")

            ax.scatter(
                df["concentration_M"],
                df["fnorm"],
                s=30,
                alpha=0.30,
                color="C0",
                zorder=1,
                label="Individual points",
            )

            ax.errorbar(
                summary["concentration_M"],
                summary["mean_value"],
                yerr=summary["sem_value"],
                fmt="o",
                color="C0",
                ecolor="C0",
                capsize=4,
                markersize=8,
                linewidth=2,
                zorder=3,
                label="Mean ± SEM",
            )

            if fit is not None:
                ax.plot(
                    fit["x_fit"],
                    fit["y_fit"],
                    color="C0",
                    linewidth=3,
                    zorder=2,
                    label=f"Kd = {fit['kd_nM']:.2f} nM",
                )

            kd_rows.append(
                {
                    "peptide": peptide_dir.name,
                    "metric": "Fnorm",
                    "channel": channel,
                    "time": time,
                    "kd_M": fit["kd_M"] if fit is not None else np.nan,
                    "kd_nM": fit["kd_nM"] if fit is not None else np.nan,
                    "bottom": fit["bottom"] if fit is not None else np.nan,
                    "top": fit["top"] if fit is not None else np.nan,
                    "n_points": len(df),
                    "n_concentrations": summary["concentration_M"].nunique(),
                }
            )

            ax.set_xscale("log")
            ax.set_xlabel("Ligand concentration [M]")
            ax.set_ylabel("Fnorm [‰]")
            ax.set_title(f"{channel} nm, {time.replace('sec', ' s')}")
            ax.legend(frameon=False, fontsize=8)

    fig.suptitle(peptide_dir.name, fontsize=18)
    plt.tight_layout()

    out_png = OUTPUT_DIR / f"{peptide_dir.name}_MST_mean_SEM_Kd.png"
    out_pdf = OUTPUT_DIR / f"{peptide_dir.name}_MST_mean_SEM_Kd.pdf"
    out_svg = OUTPUT_DIR / f"{peptide_dir.name}_MST_mean_SEM_Kd.svg"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved MST summary: {out_png}")

    spectral_result = plot_spectral_shift(peptide_dir)

    if spectral_result is not None:
        kd_rows.append(spectral_result)

    return kd_rows


# =========================
# RUN ALL PEPTIDE FOLDERS
# =========================
all_kd_rows = []

peptide_dirs = get_peptide_dirs()

for peptide_dir in peptide_dirs:
    all_kd_rows.extend(plot_peptide(peptide_dir))

kd_df = pd.DataFrame(all_kd_rows)
kd_out = OUTPUT_DIR / "MST_Kd_summary.csv"
kd_df.to_csv(kd_out, index=False)

print(f"\nSaved Kd summary: {kd_out}")
print("Done.")