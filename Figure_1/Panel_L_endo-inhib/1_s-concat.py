import pandas as pd
import os

# =========================================================
# SETTINGS
# =========================================================

ABC_ANNOTATED_CSV = (
    r"C:\Users\u244278\OneDrive - Baylor College of Medicine\Documents\python_projects\ANA_26027_LDLR_Endo-inhib\results\tables\26027_annotated_raw.csv"
)

DEF_ANNOTATED_CSV = (
    r"C:\Users\u244278\OneDrive - Baylor College of Medicine\Documents\python_projects\ANA_26027_LDLR_Endo-inhib\results\tables\26027DEF_annotated_raw.csv"
)

OUTPUT_DIR = (
    r"C:\Users\u244278\OneDrive - Baylor College of Medicine\Documents\python_projects\ANA_26027_LDLR_Endo-inhib\results\tables"
)

OUTPUT_FILE = "26027_annotated_raw_all_6_bioreps.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# CHECK INPUT FILES
# =========================================================

for path in [ABC_ANNOTATED_CSV, DEF_ANNOTATED_CSV]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find input file:\n{path}")

# =========================================================
# READ ANNOTATED PRODUCTS
# =========================================================

abc_df = pd.read_csv(ABC_ANNOTATED_CSV)
def_df = pd.read_csv(DEF_ANNOTATED_CSV)

print(f"ABC rows: {len(abc_df)}")
print(f"DEF rows: {len(def_df)}")

# =========================================================
# CONCATENATE ALL SIX BIOLOGICAL REPLICATES
# =========================================================

all_reps_df = pd.concat(
    [abc_df, def_df],
    ignore_index=True,
    sort=False,
)

# Sort the combined table for readability
sort_cols = [
    col
    for col in ["bio_rep", "Peptide", "Treatment", "well"]
    if col in all_reps_df.columns
]

if sort_cols:
    all_reps_df = all_reps_df.sort_values(sort_cols).reset_index(drop=True)

# =========================================================
# VALIDATION
# =========================================================

if "bio_rep" in all_reps_df.columns:
    detected_reps = sorted(
        all_reps_df["bio_rep"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
    )

    print(f"Biological replicates detected: {detected_reps}")

    expected_reps = {"A", "B", "C", "D", "E", "F"}
    missing_reps = expected_reps.difference(detected_reps)

    if missing_reps:
        print(
            "WARNING: The following biological replicates were not found: "
            f"{sorted(missing_reps)}"
        )

# =========================================================
# SAVE COMBINED ANNOTATED TABLE
# =========================================================

output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

all_reps_df.to_csv(output_path, index=False)

print("\nDONE")
print(f"Combined rows: {len(all_reps_df)}")
print(f"Combined annotated table saved to: {output_path}")