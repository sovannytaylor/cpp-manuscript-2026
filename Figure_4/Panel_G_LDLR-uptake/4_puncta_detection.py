"""
Detect and analyze puncta per cell using peptide-specific global thresholds.

Workflow:
1. Load images from both batches
2. Load masks from both batches
3. Optional TEST_MODE subset
4. Gaussian blur peptide channel
5. Calculate global threshold per peptide using KO-only pixels
6. Apply same peptide threshold to KO / WT / OE
7. Size/shape filter puncta
8. Save puncta_features.csv + proof images
9. Save threshold diagnostics by peptide and by peptide x rep
"""

import os
import re
import sys
import importlib.util
import functools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib_scalebar.scalebar import ScaleBar
from skimage import measure, morphology, filters
from skimage.morphology import remove_small_objects
from scipy.stats import skewtest
from loguru import logger


# -------------------------
# IMPORT NAPARI UTILS
# -------------------------

napari_utils_path = "punctalyze-SRT/src/3_napari.py"

spec = importlib.util.spec_from_file_location("napari_utils", napari_utils_path)
napari_utils = importlib.util.module_from_spec(spec)
sys.modules["napari_utils"] = napari_utils
spec.loader.exec_module(napari_utils)

remove_saturated_cells = napari_utils.remove_saturated_cells

logger.info("import ok")


# -------------------------
# CONFIG
# -------------------------

plt.rcParams.update({"font.size": 14})
sns.set_palette("Paired")

# Input folders from both batches
image_folders = [
    "results/initial_cleanup/",
    "results/initial_cleanup-02/",
]

mask_folders = [
    "results/napari_masking/",
    "results/napari_masking-02/",
]

# # Optional fallback if you want to try cellpose masks instead
# # For now, napari masks are assumed to be final masks.
# cellpose_mask_folders = [
#     "results/cellpose_masking/",
#     "results/cellpose_masking-02/",
# ]

output_folder = "results/summary_calculations-globalthreshold/"
proofs_folder = "results/proofs-globalthreshold/"

for folder in [output_folder, proofs_folder]:
    os.makedirs(folder, exist_ok=True)


# -------------------------
# TEST MODE
# -------------------------

TEST_MODE = True
MAX_IMAGES_PER_GROUP = 2

TEST_PEPTIDES = ["GP30", "GR30", "CROT", "LL37", "MOLLUSC", "LDL"]
TEST_CELLS = ["KO", "WT", "OE"]


# -------------------------
# THRESHOLD SETTINGS
# -------------------------

GAUSSIAN_SIGMA = 1.0

# calculate thresholds from KO only
THRESHOLD_CELL_TYPES = ["KO"]

# try 99.0, 99.5, 99.7, 99.9
GLOBAL_THRESHOLD_PERCENTILE = 99.5

PEPTIDES = ["GP30", "GR30", "CROT", "LL37", "MOLLUSC", "LDL"]

# If you want to manually override thresholds after testing, add values here.
# Example:
# MANUAL_PEPTIDE_THRESHOLDS = {
#     "LL37": 1450,
#     "LDL": 980,
# }
MANUAL_PEPTIDE_THRESHOLDS = {}


# If False:
#   one threshold per peptide, calculated from all KO reps combined.
# If True:
#   one threshold per peptide x rep, calculated from KO pixels in that same rep.
# Start with False, inspect the CSVs, then switch to True only if reps are wildly different.
USE_REP_SPECIFIC_THRESHOLDS = False

# Flag reps as variable if max/min threshold ratio is above this.
# Example: 1.5 means one rep threshold is 50% higher than another.
REP_THRESHOLD_FOLD_WARNING = 1.5


# -------------------------
# IMAGE / MASK SETTINGS
# -------------------------

SAT_FRAC_CUTOFF = 0.01

COI_1 = 0
COI_2 = 1

COI_1_name = "peptide"
COI_2_name = "eGFP"

SCALE_PX = 0.22
SCALE_UNIT = "um"

QUANT_REGION = "cell"

MIN_PUNCTA_SIZE = 5
MAX_PUNCTA_SIZE = 80
MIN_CIRCULARITY = 0.45
MIN_SOLIDITY = 0.70
MIN_ASPECT_RATIO = 0.35
MAX_ECCENTRICITY = 0.95

SHOW_SURVIVORS_ONLY = True


# -------------------------
# METADATA PARSING
# -------------------------

def parse_metadata_from_name(name):
    """
    Parse peptide, cell type, and biological replicate from names like:
        HEPG2_CROT_KO_REP1-1-01
        HEPG2_LL37_OE-EGFP_REP3-2-02

    Returns:
        peptide, cell, rep
    """
    base = os.path.basename(str(name))
    base = base.removesuffix(".npy").removesuffix("_mask")
    base_upper = base.upper()

    parts = base_upper.split("_")

    peptide = None
    cell = None
    rep = None

    peptide_aliases = {
        "GP30": "GP30",
        "GR30": "GR30",
        "CROT": "CROT",
        "CROTAMINE": "CROT",
        "LL37": "LL37",
        "MOLLUSC": "MOLLUSC",
        "LDL": "LDL",
        "LDL-DIL": "LDL",
        "LDLDIL": "LDL",
    }

    cell_aliases = {
        "KO": "KO",
        "WT": "WT",
        "OE": "OE",
        "OE-EGFP": "OE",
        "OE-EGFP0": "OE",
        "OE-GFP": "OE",
        "OE_EGFP": "OE",
        "OE_EGFP0": "OE",
        "OE_GFP": "OE",
        "OEEGFP": "OE",
        "OEGFP": "OE",
    }

    # Most of your filenames are HEPG2_PEPTIDE_CELL_REP...
    # But this loop is more forgiving if something shifts.
    for part in parts:
        part_clean = part.strip().upper()
        if peptide is None and part_clean in peptide_aliases:
            peptide = peptide_aliases[part_clean]
        if cell is None and part_clean in cell_aliases:
            cell = cell_aliases[part_clean]

    # Also catch aliases that contain hyphens inside the full filename.
    if cell is None:
        for key, val in cell_aliases.items():
            if re.search(rf"(^|[_\-]){re.escape(key)}($|[_\-])", base_upper):
                cell = val
                break

    if peptide is None:
        for key, val in peptide_aliases.items():
            if re.search(rf"(^|[_\-]){re.escape(key)}($|[_\-])", base_upper):
                peptide = val
                break

    # REP1-1-01 -> REP1
    rep_match = re.search(r"(REP\d+)", base_upper)
    if rep_match:
        rep = rep_match.group(1)

    return peptide, cell, rep


# -------------------------
# LOAD DATA
# -------------------------

def load_images_from_folders(folders):
    images = {}

    for folder in folders:
        if not os.path.exists(folder):
            logger.warning(f"Image folder does not exist: {folder}")
            continue

        batch_name = os.path.basename(os.path.normpath(folder))

        for fn in os.listdir(folder):
            if not fn.endswith(".npy"):
                continue

            name = fn.removesuffix(".npy")
            full_path = os.path.join(folder, fn)

            if name in images:
                name = f"{batch_name}__{name}"

            images[name] = np.load(full_path)

    logger.info(f"Loaded {len(images)} images")
    return images


def load_masks_from_folders(folders):
    masks = {}

    for folder in folders:
        if not os.path.exists(folder):
            logger.warning(f"Mask folder does not exist: {folder}")
            continue

        batch_name = os.path.basename(os.path.normpath(folder))

        for fn in os.listdir(folder):
            if not fn.endswith("_mask.npy"):
                continue

            name = fn.removesuffix("_mask.npy")
            full_path = os.path.join(folder, fn)

            if name in masks:
                name = f"{batch_name}__{name}"

            masks[name] = np.load(full_path, allow_pickle=True)

    logger.info(f"Loaded {len(masks)} masks")
    return masks


def align_images_and_masks(images, masks):
    common = sorted(set(images.keys()) & set(masks.keys()))

    missing_masks = sorted(set(images.keys()) - set(masks.keys()))
    missing_images = sorted(set(masks.keys()) - set(images.keys()))

    if missing_masks:
        logger.warning(f"{len(missing_masks)} images missing masks. Example: {missing_masks[:5]}")
    if missing_images:
        logger.warning(f"{len(missing_images)} masks missing images. Example: {missing_images[:5]}")

    images = {k: images[k] for k in common}
    masks = {k: masks[k] for k in common}

    logger.info(f"Aligned {len(common)} image/mask pairs")
    return images, masks


# -------------------------
# TEST SUBSET
# -------------------------

def save_metadata_parsing_table(images):
    rows = []

    for name in sorted(images.keys()):
        peptide, cell, rep = parse_metadata_from_name(name)
        rows.append({
            "image_name": name,
            "peptide": peptide,
            "cell": cell,
            "rep": rep,
            "parse_ok": peptide is not None and cell is not None and rep is not None,
        })

    meta_df = pd.DataFrame(rows)

    meta_df.to_csv(
        os.path.join(output_folder, "metadata_parsing_check.csv"),
        index=False
    )

    bad = meta_df[~meta_df["parse_ok"]]

    if len(bad) > 0:
        logger.warning(f"Metadata parsing failed for {len(bad)} images. See metadata_parsing_check.csv")
        logger.warning(f"Examples:\n{bad.head(10).to_string(index=False)}")
    else:
        logger.info("Metadata parsing successful for all images.")

    return meta_df


def subset_for_testing(images, masks):
    keep_names = []
    counts = {}

    for name in sorted(images.keys()):
        peptide, cell, rep = parse_metadata_from_name(name)

        if peptide not in TEST_PEPTIDES:
            continue
        if cell not in TEST_CELLS:
            continue

        key = (peptide, cell, rep)
        counts[key] = counts.get(key, 0)

        if counts[key] < MAX_IMAGES_PER_GROUP:
            keep_names.append(name)
            counts[key] += 1

    images_sub = {k: images[k] for k in keep_names if k in images}
    masks_sub = {k: masks[k] for k in keep_names if k in masks}

    logger.info(f"TEST MODE: keeping {len(images_sub)} images out of {len(images)}")

    return images_sub, masks_sub


# -------------------------
# MASKS / FILTERING
# -------------------------

def build_quant_masks(masks, region="cell"):
    quant_masks = {}

    for name, m in masks.items():
        cell_mask, nuc_mask = m[0], m[1]

        if region == "cell":
            quant_masks[name] = cell_mask

        elif region == "nucleus":
            quant_masks[name] = morphology.label(nuc_mask > 0)

        else:
            raise ValueError(f"Unknown QUANT_REGION: {region}")

    return quant_masks


def filter_saturated_images(images, quant_masks, masks):
    logger.info("filtering saturated cells...")

    filtered = {}

    for name, img in images.items():
        if name not in quant_masks or name not in masks:
            continue

        stack = np.stack([
            img[COI_2],
            img[COI_1],
            quant_masks[name],
        ])

        cells = remove_saturated_cells(
            image_stack=stack,
            mask_stack=masks[name],
            COI=COI_1
        )

        peptide_blur = filters.gaussian(
            img[COI_1].astype(np.float32),
            sigma=GAUSSIAN_SIGMA,
            preserve_range=True
        )

        filtered[name] = np.stack([
            img[COI_2],
            peptide_blur,
            cells,
        ])

    logger.info("saturated cells filtered and peptide channel blurred.")
    return filtered


# -------------------------
# GLOBAL THRESHOLDS
# -------------------------

def calculate_peptide_global_thresholds(image_dict):
    pixels_by_peptide = {pep: [] for pep in PEPTIDES}

    for name, img in image_dict.items():
        peptide, cell, rep = parse_metadata_from_name(name)

        if peptide is None or cell is None:
            logger.warning(f"Could not parse peptide/cell from name: {name}")
            continue

        if peptide not in PEPTIDES:
            continue

        if cell not in THRESHOLD_CELL_TYPES:
            continue

        coi2, coi1_blur, mask = img

        vals = coi1_blur[mask > 0]

        if vals.size > 0:
            pixels_by_peptide[peptide].append(vals)

    thresholds = {}

    for peptide, pixel_list in pixels_by_peptide.items():
        if peptide in MANUAL_PEPTIDE_THRESHOLDS:
            thresholds[peptide] = MANUAL_PEPTIDE_THRESHOLDS[peptide]
            logger.info(f"Using manual threshold for {peptide}: {thresholds[peptide]}")
            continue

        if len(pixel_list) == 0:
            logger.warning(f"No KO pixels found for peptide {peptide}; threshold will be missing.")
            thresholds[peptide] = np.nan
            continue

        all_pixels = np.concatenate(pixel_list)
        threshold = np.percentile(all_pixels, GLOBAL_THRESHOLD_PERCENTILE)

        thresholds[peptide] = threshold

        logger.info(
            f"{peptide} global threshold from {THRESHOLD_CELL_TYPES} "
            f"at {GLOBAL_THRESHOLD_PERCENTILE} percentile = {threshold:.3f}"
        )

    threshold_df = pd.DataFrame([
        {
            "peptide": pep,
            "threshold": thr,
            "percentile": GLOBAL_THRESHOLD_PERCENTILE,
            "threshold_cell_types": ",".join(THRESHOLD_CELL_TYPES),
            "gaussian_sigma": GAUSSIAN_SIGMA,
        }
        for pep, thr in thresholds.items()
    ])

    threshold_df.to_csv(
        os.path.join(output_folder, "peptide_global_thresholds.csv"),
        index=False
    )

    logger.info("Saved peptide_global_thresholds.csv")

    return thresholds


def calculate_peptide_rep_thresholds(image_dict):
    """
    Calculates KO-only threshold diagnostics per peptide x rep.

    This does NOT change the segmentation threshold unless
    USE_REP_SPECIFIC_THRESHOLDS = True.

    Outputs:
        peptide_rep_thresholds_KO_only.csv
        peptide_rep_threshold_variability_summary.csv
    """
    rows = []

    for name, img in image_dict.items():
        peptide, cell, rep = parse_metadata_from_name(name)

        if peptide is None or cell is None or rep is None:
            logger.warning(f"Could not parse peptide/cell/rep from name for rep threshold: {name}")
            continue

        if peptide not in PEPTIDES:
            continue

        if cell not in THRESHOLD_CELL_TYPES:
            continue

        coi2, coi1_blur, mask = img
        vals = coi1_blur[mask > 0]

        if vals.size == 0:
            continue

        rows.append({
            "image_name": name,
            "peptide": peptide,
            "cell": cell,
            "rep": rep,
            "n_pixels": vals.size,
            "threshold_percentile": GLOBAL_THRESHOLD_PERCENTILE,
            "rep_image_threshold": np.percentile(vals, GLOBAL_THRESHOLD_PERCENTILE),
            "mean_intensity": np.mean(vals),
            "median_intensity": np.median(vals),
            "p95_intensity": np.percentile(vals, 95),
            "p99_intensity": np.percentile(vals, 99),
            "max_intensity": np.max(vals),
        })

    image_level_df = pd.DataFrame(rows)

    if image_level_df.empty:
        logger.warning("No rows found for peptide x rep threshold diagnostics.")
        return {}, image_level_df, pd.DataFrame()

    # One threshold per peptide x rep, pooled across images in that rep.
    pooled_rows = []

    for (peptide, rep), group in image_level_df.groupby(["peptide", "rep"]):
        vals_all = []

        for name in group["image_name"]:
            coi2, coi1_blur, mask = image_dict[name]
            vals = coi1_blur[mask > 0]
            if vals.size > 0:
                vals_all.append(vals)

        if len(vals_all) == 0:
            continue

        vals_all = np.concatenate(vals_all)

        pooled_rows.append({
            "peptide": peptide,
            "rep": rep,
            "n_images": group["image_name"].nunique(),
            "n_pixels": vals_all.size,
            "threshold_percentile": GLOBAL_THRESHOLD_PERCENTILE,
            "rep_threshold": np.percentile(vals_all, GLOBAL_THRESHOLD_PERCENTILE),
            "mean_intensity": np.mean(vals_all),
            "median_intensity": np.median(vals_all),
            "p95_intensity": np.percentile(vals_all, 95),
            "p99_intensity": np.percentile(vals_all, 99),
            "max_intensity": np.max(vals_all),
        })

    rep_threshold_df = pd.DataFrame(pooled_rows)

    rep_threshold_df.to_csv(
        os.path.join(output_folder, "peptide_rep_thresholds_KO_only.csv"),
        index=False
    )

    # Summarize how different reps are for each peptide.
    summary_rows = []

    for peptide, group in rep_threshold_df.groupby("peptide"):
        thresholds = group["rep_threshold"].dropna()

        if len(thresholds) == 0:
            continue

        min_thr = thresholds.min()
        max_thr = thresholds.max()
        mean_thr = thresholds.mean()
        std_thr = thresholds.std(ddof=1) if len(thresholds) > 1 else 0

        fold_max_min = max_thr / min_thr if min_thr != 0 else np.nan
        cv = std_thr / mean_thr if mean_thr != 0 else np.nan

        summary_rows.append({
            "peptide": peptide,
            "n_reps": len(thresholds),
            "mean_rep_threshold": mean_thr,
            "median_rep_threshold": thresholds.median(),
            "std_rep_threshold": std_thr,
            "cv_rep_threshold": cv,
            "min_rep_threshold": min_thr,
            "max_rep_threshold": max_thr,
            "fold_max_over_min": fold_max_min,
            "flag_rep_variability": fold_max_min >= REP_THRESHOLD_FOLD_WARNING if not np.isnan(fold_max_min) else False,
        })

    variability_df = pd.DataFrame(summary_rows)

    variability_df.to_csv(
        os.path.join(output_folder, "peptide_rep_threshold_variability_summary.csv"),
        index=False
    )

    logger.info("Saved peptide_rep_thresholds_KO_only.csv")
    logger.info("Saved peptide_rep_threshold_variability_summary.csv")

    rep_thresholds = {
        (row["peptide"], row["rep"]): row["rep_threshold"]
        for _, row in rep_threshold_df.iterrows()
    }

    return rep_thresholds, rep_threshold_df, variability_df


def get_threshold_for_image(peptide, rep, peptide_thresholds, rep_thresholds=None):
    """
    Decide which threshold to use for a given image.
    """
    if USE_REP_SPECIFIC_THRESHOLDS:
        if rep_thresholds is not None and (peptide, rep) in rep_thresholds:
            return rep_thresholds[(peptide, rep)]

        logger.warning(
            f"USE_REP_SPECIFIC_THRESHOLDS=True but no rep threshold found for {peptide} {rep}. "
            "Falling back to peptide global threshold."
        )

    return peptide_thresholds.get(peptide, np.nan)


# -------------------------
# FEATURE HELPERS
# -------------------------

def feature_extractor(mask, properties=None):
    if properties is None:
        properties = [
            "area",
            "eccentricity",
            "solidity",
            "label",
            "major_axis_length",
            "minor_axis_length",
            "perimeter",
            "coords",
        ]

    props = measure.regionprops_table(mask, properties=properties)
    return pd.DataFrame(props)


def calculate_shape_features(df_p):
    df_p = df_p.copy()

    df_p["puncta_circularity"] = (
        4 * np.pi * df_p["puncta_area"]
    ) / (df_p["puncta_perimeter"] ** 2 + 1e-9)

    df_p["puncta_aspect_ratio"] = (
        df_p["puncta_minor_axis_length"] /
        (df_p["puncta_major_axis_length"] + 1e-9)
    )

    return df_p


def filter_puncta_by_size_shape(puncta_labels):
    df_p = feature_extractor(puncta_labels).add_prefix("puncta_")

    if df_p.empty:
        return puncta_labels, df_p

    size_keep = (
        (df_p["puncta_area"] >= MIN_PUNCTA_SIZE) &
        (df_p["puncta_area"] <= MAX_PUNCTA_SIZE)
    )

    kept_labels = df_p.loc[size_keep, "puncta_label"].astype(int).to_numpy()
    puncta_labels = np.where(np.isin(puncta_labels, kept_labels), puncta_labels, 0)
    puncta_labels = morphology.label(puncta_labels > 0)

    df_p = feature_extractor(puncta_labels).add_prefix("puncta_")

    if df_p.empty:
        return puncta_labels, df_p

    df_p = calculate_shape_features(df_p)

    shape_keep = (
        (df_p["puncta_circularity"] >= MIN_CIRCULARITY) &
        (df_p["puncta_eccentricity"] <= MAX_ECCENTRICITY) &
        (df_p["puncta_solidity"] >= MIN_SOLIDITY) &
        (df_p["puncta_aspect_ratio"] >= MIN_ASPECT_RATIO)
    )

    kept_labels = df_p.loc[shape_keep, "puncta_label"].astype(int).to_numpy()
    puncta_labels = np.where(np.isin(puncta_labels, kept_labels), puncta_labels, 0)
    puncta_labels = morphology.label(puncta_labels > 0)

    df_p = feature_extractor(puncta_labels).add_prefix("puncta_")

    if not df_p.empty:
        df_p = calculate_shape_features(df_p)

    return puncta_labels, df_p


def make_empty_cell_row(
    name,
    lbl,
    cell_mask,
    coi1_vals,
    coi2_vals,
    mean_coi1,
    std_coi1,
    peptide,
    cell,
    rep,
    threshold
):
    return pd.DataFrame([{
        "image_name": name,
        "peptide": peptide,
        "cell": cell,
        "rep": rep,
        "global_threshold": threshold,
        "threshold_mode": "rep_specific" if USE_REP_SPECIFIC_THRESHOLDS else "peptide_global",
        "cell_number": lbl,
        "cell_size": cell_mask.sum(),
        "cell_std": std_coi1,
        "cell_cv": std_coi1 / mean_coi1 if mean_coi1 != 0 else np.nan,
        "cell_skew": skewtest(coi1_vals).statistic if len(coi1_vals) >= 8 else np.nan,
        "cell_coi1_intensity_mean": mean_coi1,
        "cell_coi2_intensity_mean": coi2_vals.mean(),

        "puncta_area": 0,
        "puncta_eccentricity": np.nan,
        "puncta_solidity": np.nan,
        "puncta_label": np.nan,
        "puncta_major_axis_length": np.nan,
        "puncta_minor_axis_length": np.nan,
        "puncta_perimeter": np.nan,
        "puncta_aspect_ratio": np.nan,
        "puncta_circularity": np.nan,
        "puncta_intensity_mean": np.nan,
        "puncta_intensity_median": np.nan,
        "puncta_intensity_max": np.nan,
        "puncta_intensity_mean_in_coi2": np.nan,
    }])


def add_puncta_intensity_features(df_p, puncta_labels, coi1, coi2):
    stats = []

    for _, row in df_p.iterrows():
        label = int(row["puncta_label"])
        p_mask = puncta_labels == label

        puncta_vals = coi1[p_mask]
        coi2_vals = coi2[p_mask]

        stats.append({
            "puncta_intensity_mean": np.nanmean(puncta_vals),
            "puncta_intensity_median": np.nanmedian(puncta_vals),
            "puncta_intensity_max": np.nanmax(puncta_vals),
            "puncta_intensity_mean_in_coi2": np.nanmean(coi2_vals),
        })

    return pd.concat(
        [df_p.reset_index(drop=True), pd.DataFrame(stats).reset_index(drop=True)],
        axis=1
    )


# -------------------------
# FEATURE COLLECTION
# -------------------------

def collect_features(image_dict, peptide_thresholds, rep_thresholds=None):
    logger.info("collecting cell & puncta features with global thresholds...")

    results = []

    for name, img in image_dict.items():
        peptide, cell, rep = parse_metadata_from_name(name)

        if peptide is None:
            logger.warning(f"Skipping {name}: could not parse peptide")
            continue

        threshold = get_threshold_for_image(peptide, rep, peptide_thresholds, rep_thresholds)

        if np.isnan(threshold):
            logger.warning(f"Skipping {name}: no threshold for peptide {peptide}")
            continue

        coi2, coi1_blur, mask = img

        unique_cells = np.unique(mask)
        unique_cells = unique_cells[unique_cells != 0]

        contours = measure.find_contours((mask > 0).astype(int), 0.8)
        contour = [c for c in contours if len(c) >= 100]

        for lbl in unique_cells:
            cell_mask = mask == lbl

            coi1_vals = coi1_blur[cell_mask]
            coi2_vals = coi2[cell_mask]

            if coi1_vals.size == 0:
                continue

            mean_coi1 = coi1_vals.mean()
            std_coi1 = coi1_vals.std()

            binary = (coi1_blur > threshold) & cell_mask

            puncta_labels = morphology.label(binary)
            puncta_labels = remove_small_objects(
                puncta_labels,
                min_size=MIN_PUNCTA_SIZE
            )

            puncta_labels, df_p = filter_puncta_by_size_shape(puncta_labels)

            if df_p.empty:
                results.append(
                    make_empty_cell_row(
                        name=name,
                        lbl=lbl,
                        cell_mask=cell_mask,
                        coi1_vals=coi1_vals,
                        coi2_vals=coi2_vals,
                        mean_coi1=mean_coi1,
                        std_coi1=std_coi1,
                        peptide=peptide,
                        cell=cell,
                        rep=rep,
                        threshold=threshold
                    )
                )
                continue

            df = add_puncta_intensity_features(
                df_p=df_p,
                puncta_labels=puncta_labels,
                coi1=coi1_blur,
                coi2=coi2
            )

            df["image_name"] = name
            df["peptide"] = peptide
            df["cell"] = cell
            df["rep"] = rep
            df["global_threshold"] = threshold
            df["threshold_mode"] = "rep_specific" if USE_REP_SPECIFIC_THRESHOLDS else "peptide_global"
            df["cell_number"] = lbl
            df["cell_size"] = cell_mask.sum()
            df["cell_std"] = std_coi1
            df["cell_cv"] = std_coi1 / mean_coi1 if mean_coi1 != 0 else np.nan
            df["cell_skew"] = skewtest(coi1_vals).statistic if len(coi1_vals) >= 8 else np.nan
            df["cell_coi1_intensity_mean"] = mean_coi1
            df["cell_coi2_intensity_mean"] = coi2_vals.mean()
            df["cell_coords"] = [contour] * len(df)

            results.append(df)

    logger.info("feature extraction done.")
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# -------------------------
# PROOF IMAGES
# -------------------------

def get_surviving_puncta_labels_for_image(puncta_img, mask_labels, threshold):
    binary = (puncta_img > threshold) & (mask_labels > 0)

    plab = morphology.label(binary)
    plab = remove_small_objects(plab, min_size=MIN_PUNCTA_SIZE)

    plab, df_p = filter_puncta_by_size_shape(plab)

    return plab


def generate_proofs(df, image_dict, peptide_thresholds, rep_thresholds=None):
    logger.info("Generating proof plots...")

    for name, img in image_dict.items():
        peptide, cell, rep = parse_metadata_from_name(name)

        if peptide is None:
            continue

        threshold = get_threshold_for_image(peptide, rep, peptide_thresholds, rep_thresholds)

        if np.isnan(threshold):
            continue

        coi2, coi1_blur, mask = img

        region_img = coi1_blur * (mask > 0)

        contours = measure.find_contours((mask > 0).astype(int), 0.8)
        contours = [c for c in contours if len(c) >= 100]

        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 6))

        ax0.imshow(coi2, cmap="gray_r")
        ax0.set_title(f"{COI_2_name}")
        ax0.axis("off")

        ax1.imshow(coi1_blur, cmap="gray_r")
        ax1.set_title(f"{COI_1_name} gaussian sigma={GAUSSIAN_SIGMA}")
        ax1.axis("off")

        ax2.imshow(region_img, cmap="gray_r")

        for line in contours:
            ax2.plot(line[:, 1], line[:, 0], c="k", lw=0.6)

        survivors = get_surviving_puncta_labels_for_image(
            puncta_img=coi1_blur,
            mask_labels=mask,
            threshold=threshold
        )

        for c in measure.find_contours((survivors > 0).astype(float), 0.5):
            ax2.plot(c[:, 1], c[:, 0], color="red", lw=1.0)

        n_survive = np.unique(survivors).size - 1

        ax2.set_title(f"Puncta survivors | threshold={threshold:.1f}")
        ax2.axis("off")

        scalebar = ScaleBar(
            SCALE_PX,
            SCALE_UNIT,
            location="lower right",
            pad=0.3,
            sep=2,
            box_alpha=0,
            color="gray",
            length_fraction=0.3
        )

        ax1.add_artist(scalebar)

        fig.suptitle(
            f"{name} | {peptide} {cell} {rep} | survivors: {n_survive}",
            y=0.98
        )

        fig.tight_layout()

        safe_name = name.replace("/", "-").replace("\\", "-")
        fig.savefig(
            os.path.join(proofs_folder, f"{safe_name}_proof.png"),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

    logger.info("proofs saved.")


# -------------------------
# EXTRA / SAVE
# -------------------------

def extra_puncta_features(df):
    df = df.copy()

    if "puncta_minor_axis_length" in df.columns and "puncta_major_axis_length" in df.columns:
        df["puncta_aspect_ratio"] = (
            df["puncta_minor_axis_length"] /
            (df["puncta_major_axis_length"] + 1e-9)
        )

    if "puncta_area" in df.columns and "puncta_perimeter" in df.columns:
        df["puncta_circularity"] = (
            4 * np.pi * df["puncta_area"]
        ) / (df["puncta_perimeter"] ** 2 + 1e-9)

    return df


# -------------------------
# MAIN
# -------------------------

if __name__ == "__main__":
    logger.info("loading images and masks...")

    images = load_images_from_folders(image_folders)
    masks = load_masks_from_folders(mask_folders)

    images, masks = align_images_and_masks(images, masks)

    if TEST_MODE:
        images, masks = subset_for_testing(images, masks)

    if len(images) == 0:
        logger.warning("No image/mask pairs found. Check folder names and filenames.")
        sys.exit(0)

    save_metadata_parsing_table(images)

    quant_masks = build_quant_masks(masks, QUANT_REGION)

    filtered = filter_saturated_images(
        images=images,
        quant_masks=quant_masks,
        masks=masks
    )

    peptide_thresholds = calculate_peptide_global_thresholds(filtered)

    rep_thresholds, rep_threshold_df, variability_df = calculate_peptide_rep_thresholds(filtered)

    if not variability_df.empty:
        print("\nPEPTIDE THRESHOLD VARIABILITY ACROSS KO REPS")
        print(variability_df.to_string(index=False))

    features = collect_features(
        image_dict=filtered,
        peptide_thresholds=peptide_thresholds,
        rep_thresholds=rep_thresholds
    )

    if features.empty:
        logger.warning("No cells/features detected; nothing to save/plot.")
        sys.exit(0)

    features = extra_puncta_features(features)

    generate_proofs(
        df=features,
        image_dict=filtered,
        peptide_thresholds=peptide_thresholds,
        rep_thresholds=rep_thresholds
    )

    logger.info("proofs complete.")
    logger.info("starting data wrangling and saving...")

    cols_to_drop = [col for col in features.columns if "_coords" in col]
    features = features.drop(columns=cols_to_drop)

    features.to_csv(
        os.path.join(output_folder, "puncta_features.csv"),
        index=False
    )

    logger.info("saved puncta_features.csv")
    logger.info("pipeline complete.")