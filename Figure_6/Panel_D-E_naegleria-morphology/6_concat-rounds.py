from pathlib import Path
import pandas as pd
from loguru import logger

logger.info("import ok")

# --------------------------------
# CONFIG
# --------------------------------
BASE_DIR = Path(".")
OUTPUT_DIR = BASE_DIR / "concat_calculations"
OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------
# FIND ALL CSVS
# --------------------------------
csv_groups = {}

for csv_file in BASE_DIR.glob("*_results/summary_calculations/*.csv"):

    csv_name = csv_file.name

    if csv_name not in csv_groups:
        csv_groups[csv_name] = []

    csv_groups[csv_name].append(csv_file)

# --------------------------------
# CONCAT EACH CSV TYPE
# --------------------------------
for csv_name, file_list in csv_groups.items():

    logger.info(f"Concatenating {csv_name}")
    logger.info(f"Found {len(file_list)} files")

    dfs = []

    for file in sorted(file_list):

        logger.info(f"Reading {file}")

        df = pd.read_csv(file)

        # optional: keep track of source plate
        df["source_folder"] = file.parent.parent.name

        dfs.append(df)

    combined_df = pd.concat(
        dfs,
        ignore_index=True
    )

    output_file = OUTPUT_DIR / csv_name

    combined_df.to_csv(
        output_file,
        index=False
    )

    logger.info(
        f"Saved {len(combined_df)} rows -> {output_file}"
    )

logger.info("Done")