"""
Merge the two peptide–EEA1 image folders and the two Napari mask folders
while standardizing filenames to a single hyphen-delimited convention.

Examples
--------
ARMIN_rep1.npy       -> ARMIN-REP1-SET1.npy
ARMIN-rep1.npy       -> ARMIN-REP1-SET2.npy
ARMIN_rep1_01.npy    -> ARMIN-REP1-01.npy
ARMIN_rep1_01_mask.npy -> ARMIN-REP1-01_mask.npy

The script copies files; it never changes or deletes the source data.
When different source files would otherwise receive the same name, it adds
SET1, SET2, etc. based on their source-folder order. The same suffix is
therefore added to an image and its corresponding mask.
It also verifies that every merged image has a matching merged mask.
"""

from pathlib import Path
import hashlib
import logging
import re
import shutil

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# SETTINGS
# ============================================================

RESULTS_FOLDER = Path("results")

IMAGE_SOURCE_FOLDERS = [
    RESULTS_FOLDER / "initial_cleanup",
    RESULTS_FOLDER / "initial_cleanup-02",
]

MASK_SOURCE_FOLDERS = [
    RESULTS_FOLDER / "napari_masking",
    RESULTS_FOLDER / "napari_masking-02",
]

MERGED_IMAGE_FOLDER = RESULTS_FOLDER / "initial_cleanup_merged"
MERGED_MASK_FOLDER = RESULTS_FOLDER / "napari_masking_merged"
MANIFEST_CSV = RESULTS_FOLDER / "merged_filename_manifest.csv"
PAIRING_CSV = RESULTS_FOLDER / "merged_image_mask_pairing_check.csv"

# True: report planned changes without copying anything.
# False: create the merged folders and copy the files.
DRY_RUN = False


# ============================================================
# FILENAME STANDARDIZATION
# ============================================================

def standardize_stem(filename, is_mask=False):
    """
    Convert both underscore- and hyphen-delimited names to:
        PEPTIDE-REP#-FIELD

    The trailing "_mask" text is retained only for mask filenames.
    """
    name = Path(filename).name
    if not name.lower().endswith(".npy"):
        raise ValueError(f"Expected an .npy file: {filename}")

    stem = name[:-4]
    if stem.lower().endswith("_mask"):
        stem = stem[:-5]
        is_mask = True

    parts = [part.strip() for part in re.split(r"[_-]+", stem) if part.strip()]
    if not parts:
        raise ValueError(f"Could not parse filename: {filename}")

    standardized_parts = [parts[0].upper()]
    for part in parts[1:]:
        if re.fullmatch(r"rep\d+", part, flags=re.IGNORECASE):
            standardized_parts.append(part.upper())
        else:
            standardized_parts.append(part)

    standardized = "-".join(standardized_parts)
    if is_mask:
        standardized += "_mask"
    return standardized + ".npy"


def file_md5(path, chunk_size=1024 * 1024):
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def add_set_suffix(filename, set_number):
    """Insert SET# before the optional mask suffix and .npy extension."""
    suffix = "_mask.npy" if filename.lower().endswith("_mask.npy") else ".npy"
    stem = filename[:-len(suffix)]
    return f"{stem}-SET{set_number}{suffix}"


# ============================================================
# MERGING
# ============================================================

def merge_group(source_folders, destination_folder, file_type):
    rows = []
    destinations = {}

    if not DRY_RUN:
        destination_folder.mkdir(parents=True, exist_ok=True)

    # Determine collisions before copying so every member of a collision gets
    # an explicit SET# name, including the first file encountered.
    planned_files = []
    base_name_counts = {}

    for set_number, source_folder in enumerate(source_folders, start=1):
        if not source_folder.exists():
            logger.warning(f"Source folder does not exist: {source_folder}")
            continue

        for source_path in sorted(source_folder.glob("*.npy")):
            is_mask = file_type == "mask" or source_path.name.lower().endswith(
                "_mask.npy"
            )
            base_name = standardize_stem(source_path.name, is_mask=is_mask)
            planned_files.append(
                (set_number, source_folder, source_path, base_name)
            )
            base_name_counts[base_name] = base_name_counts.get(base_name, 0) + 1

    for set_number, source_folder, source_path, base_name in planned_files:
            standardized_name = (
                add_set_suffix(base_name, set_number)
                if base_name_counts[base_name] > 1
                else base_name
            )
            destination_path = destination_folder / standardized_name

            status = "planned" if DRY_RUN else "copied"
            duplicate_of = ""

            if standardized_name in destinations:
                first_source = destinations[standardized_name]
                if file_md5(source_path) == file_md5(first_source):
                    status = "identical_duplicate_skipped"
                    duplicate_of = str(first_source)
                else:
                    raise RuntimeError(
                        "Two different files still resolve to the same "
                        "standardized "
                        f"name:\n  {first_source}\n  {source_path}\n"
                        f"Standardized name: {standardized_name}\n"
                        "No files were overwritten."
                    )
            else:
                destinations[standardized_name] = source_path
                if not DRY_RUN:
                    shutil.copy2(source_path, destination_path)

            rows.append(
                {
                    "file_type": file_type,
                    "source_folder": str(source_folder),
                    "original_filename": source_path.name,
                    "standardized_filename": standardized_name,
                    "destination_folder": str(destination_folder),
                    "status": status,
                    "duplicate_of": duplicate_of,
                }
            )

    return rows


def make_pairing_check():
    image_names = {
        path.stem for path in MERGED_IMAGE_FOLDER.glob("*.npy")
        if not path.name.lower().endswith("_mask.npy")
    }
    mask_names = {
        path.name[:-9] for path in MERGED_MASK_FOLDER.glob("*_mask.npy")
    }

    all_names = sorted(image_names | mask_names)
    rows = [
        {
            "image_name": name,
            "image_present": name in image_names,
            "mask_present": name in mask_names,
            "pair_complete": name in image_names and name in mask_names,
        }
        for name in all_names
    ]
    return pd.DataFrame(rows)


def main():
    manifest_rows = []
    manifest_rows.extend(
        merge_group(
            IMAGE_SOURCE_FOLDERS,
            MERGED_IMAGE_FOLDER,
            file_type="image",
        )
    )
    manifest_rows.extend(
        merge_group(
            MASK_SOURCE_FOLDERS,
            MERGED_MASK_FOLDER,
            file_type="mask",
        )
    )

    manifest = pd.DataFrame(manifest_rows)
    if not DRY_RUN:
        MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(MANIFEST_CSV, index=False)

        pairing = make_pairing_check()
        pairing.to_csv(PAIRING_CSV, index=False)

        n_complete = int(pairing["pair_complete"].sum())
        n_incomplete = int((~pairing["pair_complete"]).sum())
        logger.info(f"Complete image/mask pairs: {n_complete}")
        logger.info(f"Incomplete image/mask pairs: {n_incomplete}")

        if n_incomplete:
            logger.warning(
                f"Review unmatched files in: {PAIRING_CSV}"
            )
    else:
        logger.info(f"Dry run complete. Planned file records: {len(manifest)}")

    logger.info(f"Merged images: {MERGED_IMAGE_FOLDER}")
    logger.info(f"Merged masks: {MERGED_MASK_FOLDER}")


if __name__ == "__main__":
    main()