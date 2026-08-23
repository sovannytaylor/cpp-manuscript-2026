import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# =========================
# EDIT THESE
# =========================
BASE_DIR = Path(
    r"C:\Users\u244278\OneDrive - Baylor College of Medicine\Boeynaems, Steven's files - Boeynaems_lab_drive\People Data\Sophie\Papers\+CPPs\Fig6\MST-06012026"
)

# Only folders listed here will be analyzed.
# These names must exactly match the peptide folder names inside BASE_DIR.
PEPTIDE_FOLDERS_TO_USE = [
    "CROT",
    "CY5",
    "GP30",
    "GR30",
    "LL37",
]

OUTPUT_DIR = BASE_DIR / "MST_readable_csvs_and_graphs"
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================
# SETTINGS
# =========================
TIMES = ["1.5sec", "2.5sec", "5sec", "10sec"]
CHANNELS = ["650", "670"]

CONC_COL = "Ligand Concentration [M]"
FNORM_COL = "Fnorm [‰]"


def read_mst_csv(path):
    """Read NanoTemper/MST csvs that may be semicolon-separated."""
    try:
        return pd.read_csv(path, sep=";", encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, encoding="utf-8-sig")


def clean_filename(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


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
        popt, _ = curve_fit(
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
    peptide_dirs = []

    for folder_name in PEPTIDE_FOLDERS_TO_USE:
        folder_path = BASE_DIR / folder_name

        if folder_path.exists() and folder_path.is_dir():
            peptide_dirs.append(folder_path)
        else:
            print(f"WARNING: folder not found and will be skipped: {folder_name}")

    print("\nPeptide folders selected:")
    for d in peptide_dirs:
        print(f"  {d.name}")

    return peptide_dirs


def get_experiment_dirs(peptide_dir):
    return [
        d for d in peptide_dir.iterdir()
        if d.is_dir() and d.name.lower().startswith("experiment")
    ]


def collect_mst_data(peptide_dir, time, channel):
    all_rows = []

    experiment_dirs = get_experiment_dirs(peptide_dir)

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
            print(f"Skipping missing required columns: {file.name}")
            continue

        raw = df.dropna(subset=[CONC_COL, FNORM_COL]).copy()

        if "Capillary Position" in raw.columns:
            raw = raw[raw["Capillary Position"].notna()]

        raw["peptide"] = peptide_dir.name
        raw["experiment"] = exp_dir.name
        raw["time"] = time
        raw["channel"] = channel
        raw["source_file"] = file.name
        raw["concentration_M"] = raw[CONC_COL].astype(float)
        raw["concentration_nM"] = raw["concentration_M"] * 1e9
        raw["fnorm"] = raw[FNORM_COL].astype(float)

        keep_cols = [
            "peptide",
            "experiment",
            "time",
            "channel",
            "source_file",
            "concentration_M",
            "concentration_nM",
            "fnorm",
        ]

        all_rows.append(raw[keep_cols])

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


def collect_spectral_shift_data(peptide_dir):
    all_rows = []

    experiment_dirs = get_experiment_dirs(peptide_dir)

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
            print(f"Skipping spectral shift file missing columns: {ratio_file.name}")
            continue

        raw = df.dropna(subset=[CONC_COL, ratio_col]).copy()

        if "Capillary Position" in raw.columns:
            raw = raw[raw["Capillary Position"].notna()]

        raw["peptide"] = peptide_dir.name
        raw["experiment"] = exp_dir.name
        raw["source_file"] = ratio_file.name
        raw["concentration_M"] = raw[CONC_COL].astype(float)
        raw["concentration_nM"] = raw["concentration_M"] * 1e9
        raw["spectral_shift_670_650"] = raw[ratio_col].astype(float)

        all_rows.append(
            raw[
                [
                    "peptide",
                    "experiment",
                    "source_file",
                    "concentration_M",
                    "concentration_nM",
                    "spectral_shift_670_650",
                ]
            ]
        )

    if not all_rows:
        return pd.DataFrame()

    return pd.concat(all_rows, ignore_index=True)


def summarize_replicates(df, value_col):
    summary = (
        df.groupby(["peptide", "concentration_M", "concentration_nM"], as_index=False)
        .agg(
            mean_value=(value_col, "mean"),
            sd_value=(value_col, "std"),
            n_points=(value_col, "count"),
            n_experiments=("experiment", "nunique"),
        )
    )

    summary["sem_value"] = summary["sd_value"] / np.sqrt(summary["n_points"])
    return summary.sort_values("concentration_M")


def save_clean_csv(df, path):
    """Save normal readable comma-separated CSV."""
    df.to_csv(path, index=False, sep=",")


def plot_single_curve(df, summary, fit, peptide, y_col, y_label, title, out_prefix):
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(
        df["concentration_M"],
        df[y_col],
        s=30,
        alpha=0.30,
        zorder=1,
        label="Individual capillaries",
    )

    ax.errorbar(
        summary["concentration_M"],
        summary["mean_value"],
        yerr=summary["sem_value"],
        fmt="o",
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
            linewidth=3,
            zorder=2,
            label=f"Kd = {fit['kd_nM']:.2f} nM",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Ligand concentration [M]")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(frameon=False)

    plt.tight_layout()

    fig.savefig(f"{out_prefix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_prefix}.svg", bbox_inches="tight")

    plt.close(fig)


def plot_spectral_shift(peptide_dir):
    df = collect_spectral_shift_data(peptide_dir)

    if df.empty:
        print(f"No spectral shift data found for {peptide_dir.name}")
        return None

    peptide_clean = clean_filename(peptide_dir.name)

    raw_out = OUTPUT_DIR / f"{peptide_clean}_spectral_shift_raw_readable.csv"
    save_clean_csv(df, raw_out)

    summary = summarize_replicates(df, "spectral_shift_670_650")
    summary_out = OUTPUT_DIR / f"{peptide_clean}_spectral_shift_summary_readable.csv"
    save_clean_csv(summary, summary_out)

    fit = fit_kd(summary, "mean_value")

    out_prefix = OUTPUT_DIR / f"{peptide_clean}_spectral_shift_mean_SEM_Kd"

    plot_single_curve(
        df=df,
        summary=summary,
        fit=fit,
        peptide=peptide_dir.name,
        y_col="spectral_shift_670_650",
        y_label="670/650 spectral shift",
        title=f"{peptide_dir.name} spectral shift",
        out_prefix=out_prefix,
    )

    print(f"Saved spectral shift CSVs and graphs for {peptide_dir.name}")

    return {
        "peptide": peptide_dir.name,
        "metric": "spectral_shift_670_650",
        "channel": "670/650",
        "time": "NA",
        "kd_M": fit["kd_M"] if fit is not None else np.nan,
        "kd_nM": fit["kd_nM"] if fit is not None else np.nan,
        "bottom": fit["bottom"] if fit is not None else np.nan,
        "top": fit["top"] if fit is not None else np.nan,
        "n_points": len(df),
        "n_concentrations": summary["concentration_M"].nunique(),
        "n_experiments": df["experiment"].nunique(),
    }


def plot_peptide_mst_grid(peptide_dir):
    print(f"\nProcessing peptide: {peptide_dir.name}")

    peptide_clean = clean_filename(peptide_dir.name)

    all_raw_rows = []
    all_summary_rows = []
    kd_rows = []

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=False)
    axes = axes.flatten()

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

            all_raw_rows.append(df)

            summary = summarize_replicates(df, "fnorm")
            summary["channel"] = channel
            summary["time"] = time
            all_summary_rows.append(summary)

            fit = fit_kd(summary, "mean_value")

            ax.scatter(
                df["concentration_M"],
                df["fnorm"],
                s=30,
                alpha=0.30,
                zorder=1,
                label="Individual capillaries",
            )

            ax.errorbar(
                summary["concentration_M"],
                summary["mean_value"],
                yerr=summary["sem_value"],
                fmt="o",
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
                    linewidth=3,
                    zorder=2,
                    label=f"Kd = {fit['kd_nM']:.2f} nM",
                )

            kd_rows.append(
                {
                    "peptide": peptide_dir.name,
                    "metric": "fnorm",
                    "channel": channel,
                    "time": time,
                    "kd_M": fit["kd_M"] if fit is not None else np.nan,
                    "kd_nM": fit["kd_nM"] if fit is not None else np.nan,
                    "bottom": fit["bottom"] if fit is not None else np.nan,
                    "top": fit["top"] if fit is not None else np.nan,
                    "n_points": len(df),
                    "n_concentrations": summary["concentration_M"].nunique(),
                    "n_experiments": df["experiment"].nunique(),
                }
            )

            ax.set_xscale("log")
            ax.set_xlabel("Ligand concentration [M]")
            ax.set_ylabel("Fnorm [‰]")
            ax.set_title(f"{channel} nm, {time.replace('sec', ' s')}")
            ax.legend(frameon=False, fontsize=8)

            single_prefix = OUTPUT_DIR / f"{peptide_clean}_fnorm_{channel}_{time}_mean_SEM_Kd"
            plot_single_curve(
                df=df,
                summary=summary,
                fit=fit,
                peptide=peptide_dir.name,
                y_col="fnorm",
                y_label="Fnorm [‰]",
                title=f"{peptide_dir.name}: {channel} nm, {time.replace('sec', ' s')}",
                out_prefix=single_prefix,
            )

    fig.suptitle(peptide_dir.name, fontsize=18)
    plt.tight_layout()

    grid_prefix = OUTPUT_DIR / f"{peptide_clean}_MST_all_channels_times_mean_SEM_Kd"
    fig.savefig(f"{grid_prefix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{grid_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{grid_prefix}.svg", bbox_inches="tight")
    plt.close(fig)

    if all_raw_rows:
        raw_all = pd.concat(all_raw_rows, ignore_index=True)
        raw_out = OUTPUT_DIR / f"{peptide_clean}_fnorm_raw_readable.csv"
        save_clean_csv(raw_all, raw_out)

    if all_summary_rows:
        summary_all = pd.concat(all_summary_rows, ignore_index=True)
        summary_out = OUTPUT_DIR / f"{peptide_clean}_fnorm_summary_readable.csv"
        save_clean_csv(summary_all, summary_out)

    spectral_result = plot_spectral_shift(peptide_dir)

    if spectral_result is not None:
        kd_rows.append(spectral_result)

    print(f"Saved MST CSVs and graphs for {peptide_dir.name}")

    return kd_rows


# =========================
# RUN SELECTED FOLDERS ONLY
# =========================
all_kd_rows = []

peptide_dirs = get_peptide_dirs()

for peptide_dir in peptide_dirs:
    all_kd_rows.extend(plot_peptide_mst_grid(peptide_dir))

kd_df = pd.DataFrame(all_kd_rows)
kd_out = OUTPUT_DIR / "MST_Kd_summary_readable.csv"
save_clean_csv(kd_df, kd_out)

print(f"\nSaved Kd summary: {kd_out}")
print(f"All outputs saved in: {OUTPUT_DIR}")
print("Done.")