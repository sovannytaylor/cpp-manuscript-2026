import pandas as pd
import numpy as np
import glob
import os
import re

# Optional: keeps text editable in Illustrator SVGs
# import matplotlib.pyplot as plt
# plt.rcParams["svg.fonttype"] = "none"

# =========================================================
# SETTINGS
# =========================================================

DATA_FOLDER = r"C:\Users\u244278\OneDrive - Baylor College of Medicine\Documents\Boeynaems Lab 2026\flow_data\26027_endo-inhib\DEF"
FILE_PATTERN = "26027*.csv"

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

RESULTS_DIR = r"C:\Users\u244278\OneDrive - Baylor College of Medicine\Documents\python_projects\ANA_26027_LDLR_Endo-inhib\results"
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

OUTPUT_RAW = "26027DEF_annotated_raw.csv"
OUTPUT_GROUPED = "26027DEF_grouped_by_biorep_peptide_treatment.csv"

# =========================================================
# COLUMN RENAMING
# =========================================================

RENAME_DICT = {
    "Unnamed: 0": "well",
    "Cells/FSC singlets/SSC singlets | Count": "ssc_count",
    "Cells/FSC singlets/SSC singlets/AF594-A, FSC-A subset | Count": "subset_count",
    "Cells/FSC singlets/SSC singlets | Mean (YL1-A :: AF594-A)": "ssc_mean",
    "Cells/FSC singlets/SSC singlets/AF594-A, FSC-A subset | Mean (YL1-A :: AF594-A)": "subset_mean",
    "Cells/FSC singlets/SSC singlets | Median (YL1-A :: AF594-A)": "ssc_median",
    "Cells/FSC singlets/SSC singlets/AF594-A, FSC-A subset | Median (YL1-A :: AF594-A)": "subset_median",
    "Cells/FSC singlets/SSC singlets/AF594-A, FSC-A subset | Freq. of Parent (%)": "freq_subset",
    "Cells/FSC singlets/SSC singlets | Percentile (YL1-A :: AF594-A)": "top10percent",
    "Unnamed: 9": "empty_col",
}

VALUE_COLS = [
    "ssc_count",
    "subset_count",
    "ssc_mean",
    "subset_mean",
    "ssc_median",
    "subset_median",
    "freq_subset",
    "top10percent",
]

# =========================================================
# PLATE KEY
# =========================================================

PLATE_KEY_ROWS = [
    # GR30 and LL37
    ("B2", "GR30", "dynasore"),
    ("B3", "LL37", "dynasore"),
    ("B4", "GR30", "amiloride"),
    ("B5", "LL37", "amiloride"),
    ("B6", "GR30", "pitstop"),
    ("B7", "LL37", "pitstop"),
    ("B8", "GR30", "genistein"),
    ("B9", "LL37", "genistein"),
    ("B10", "GR30", "dmso"),
    ("B11", "LL37", "dmso"),

    ("C2", "GR30", "dynasore"),
    ("C3", "LL37", "dynasore"),
    ("C4", "GR30", "amiloride"),
    ("C5", "LL37", "amiloride"),
    ("C6", "GR30", "pitstop"),
    ("C7", "LL37", "pitstop"),
    ("C8", "GR30", "genistein"),
    ("C9", "LL37", "genistein"),
    ("C10", "GR30", "dmso"),
    ("C11", "LL37", "dmso"),

    ("D2", "GR30", "dynasore"),
    ("D3", "LL37", "dynasore"),
    ("D4", "GR30", "amiloride"),
    ("D5", "LL37", "amiloride"),
    ("D6", "GR30", "pitstop"),
    ("D7", "LL37", "pitstop"),
    ("D8", "GR30", "genistein"),
    ("D9", "LL37", "genistein"),
    ("D10", "GR30", "dmso"),
    ("D11", "LL37", "dmso"),

    # GP30 and CROT
    ("E2", "GP30", "dynasore"),
    ("E3", "CROT", "dynasore"),
    ("E4", "GP30", "amiloride"),
    ("E5", "CROT", "amiloride"),
    ("E6", "GP30", "pitstop"),
    ("E7", "CROT", "pitstop"),
    ("E8", "GP30", "genistein"),
    ("E9", "CROT", "genistein"),
    ("E10", "GP30", "dmso"),
    ("E11", "CROT", "dmso"),

    ("F2", "GP30", "dynasore"),
    ("F3", "CROT", "dynasore"),
    ("F4", "GP30", "amiloride"),
    ("F5", "CROT", "amiloride"),
    ("F6", "GP30", "pitstop"),
    ("F7", "CROT", "pitstop"),
    ("F8", "GP30", "genistein"),
    ("F9", "CROT", "genistein"),
    ("F10", "GP30", "dmso"),
    ("F11", "CROT", "dmso"),

    ("G2", "GP30", "dynasore"),
    ("G3", "CROT", "dynasore"),
    ("G4", "GP30", "amiloride"),
    ("G5", "CROT", "amiloride"),
    ("G6", "GP30", "pitstop"),
    ("G7", "CROT", "pitstop"),
    ("G8", "GP30", "genistein"),
    ("G9", "CROT", "genistein"),
    ("G10", "GP30", "dmso"),
    ("G11", "CROT", "dmso"),

    # Controls
    ("H2", "NEG", "na"),
    ("H3", "UNSTAIN", "na"),
    ("H4", "UNSTAIN", "na"),
]

plate_key = pd.DataFrame(PLATE_KEY_ROWS, columns=["well", "Peptide", "Treatment"])

# normalize treatment spellings just in case
plate_key["Treatment"] = (
    plate_key["Treatment"]
    .astype(str)
    .str.strip()
    .str.lower()
    .replace({"pistop": "pitstop"})
)

# =========================================================
# HELPERS
# =========================================================

def remove_summary_rows(df, well_col="well"):
    return df[
        ~df[well_col]
        .astype(str)
        .str.contains(r"Mean|SD|Std|CV", case=False, na=False)
    ].copy()

def clean_well_string(x: str) -> str:
    return str(x).strip().replace(".fcs", "").replace(".FCS", "").upper()

def extract_bio_rep_from_filename(filename: str) -> str:
    """
    Assumes filenames end in D/E/F before .csv, e.g. 26027D.csv, 26027E.csv, 26027F.csv
    """
    base = os.path.basename(filename)
    m = re.search(r"([DEF])\.csv$", base, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # fallback: trailing D/E/F before anything optional
    m = re.search(r"([DEF])(?:[^A-Za-z0-9]?\.csv)$", base, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return pd.NA

# =========================================================
# READ FILES
# =========================================================

csv_files = sorted(glob.glob(os.path.join(DATA_FOLDER, FILE_PATTERN)))
if len(csv_files) == 0:
    raise FileNotFoundError(f"No files found matching: {os.path.join(DATA_FOLDER, FILE_PATTERN)}")

all_dfs = []

for file in csv_files:
    df = pd.read_csv(file)
    base = os.path.basename(file)

    df = df.rename(columns=RENAME_DICT)

    if "well" not in df.columns:
        raise ValueError(f"'well' column not found after renaming in file: {base}")

    if "empty_col" in df.columns and df["empty_col"].isna().all():
        df = df.drop(columns=["empty_col"])

    df["source_file"] = base
    df["bio_rep"] = extract_bio_rep_from_filename(base)

    # clean well labels like A1.fcs -> A1
    df["well"] = df["well"].map(clean_well_string)

    # remove summary rows like Mean, SD
    df = remove_summary_rows(df, "well")

    # merge in plate key
    df = df.merge(plate_key, on="well", how="left")

    # normalize treatment spelling again after merge
    df["Treatment"] = (
        df["Treatment"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"pistop": "pitstop"})
    )

    # numeric coercion
    for col in VALUE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    all_dfs.append(df)

big_df = pd.concat(all_dfs, ignore_index=True)

# =========================================================
# CLEAN + SAVE RAW ANNOTATED
# =========================================================

# keep rows that mapped to the plate key
big_df = big_df.dropna(subset=["bio_rep", "Peptide", "Treatment"])

big_df.to_csv(os.path.join(TABLES_DIR, OUTPUT_RAW), index=False)

# =========================================================
# GROUP / COLLAPSE
# Group by biological replicate + peptide + treatment
# =========================================================

present_value_cols = [c for c in VALUE_COLS if c in big_df.columns]

grouped_df = (
    big_df
    .groupby(["bio_rep", "Peptide", "Treatment"], as_index=False)[present_value_cols]
    .mean()
)

grouped_df.to_csv(os.path.join(TABLES_DIR, OUTPUT_GROUPED), index=False)

print("DONE")
print(f"Annotated raw table saved to: {os.path.join(TABLES_DIR, OUTPUT_RAW)}")
print(f"Grouped table saved to: {os.path.join(TABLES_DIR, OUTPUT_GROUPED)}")
print("\nGrouped preview:")
print(grouped_df.head())