"""
Detect peptide and lipoprotein puncta per cell, then quantify colocalization.

Channel 0 = peptide puncta
Channel 1 = lipoprotein puncta

GP30 = negative control for peptide puncta.
GP30 is used to set the peptide threshold floor, but GP30 itself is forced to
have zero peptide puncta.

Colocalization definition:
    A peptide punctum is colocalized if it overlaps a lipoprotein punctum
    OR is within 1 pixel in any direction.
"""

import os
import re
import sys
import importlib.util

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib_scalebar.scalebar import ScaleBar
from skimage import measure, morphology, filters
from skimage.morphology import remove_small_objects
from loguru import logger


# -------------------------
# IMPORT NAPARI UTILS
# -------------------------

napari_utils_path = "punctalyze-SRT/src/3_napari.py"

spec = importlib.util.spec_from_file_location("napari_utils", napari_utils_path)
napari_utils = importlib.util.module_from_spec(spec)
sys.modules["napari_utils"] = napari_utils
spec.loader.exec_module(napari_utils)

logger.info("import ok")


# -------------------------
# CONFIG
# -------------------------

plt.rcParams.update({"font.size": 14})
sns.set_palette("Paired")

image_folders = [
    "results/initial_cleanup/",
]

mask_folders = [
    "results/napari_masking/",
]

output_folder = "results/summary_calculations-peptide-lipo-coloc/"
proofs_folder = "results/proofs-peptide-lipo-coloc/"

for folder in [output_folder, proofs_folder]:
    os.makedirs(folder, exist_ok=True)


# -------------------------
# TEST MODE
# -------------------------

TEST_MODE = False
MAX_IMAGES_PER_GROUP = 1

TEST_PEPTIDES = ["CROT", "LL37", "GR30", "GP30"]
TEST_LIPOPROTEINS = ["HDL", "LDL", "VLDL"]


# -------------------------
# IMAGE SETTINGS
# -------------------------

PEPTIDE_CH = 0
LIPO_CH = 1

PEPTIDE_CH_NAME = "peptide"
LIPO_CH_NAME = "lipoprotein"

SCALE_PX = 0.22
SCALE_UNIT = "um"

QUANT_REGION = "cell"

GAUSSIAN_SIGMA = 0.75

GLOBAL_THRESHOLD_PERCENTILE = 99.0

NEGATIVE_CONTROL_PEPTIDE = "GP30"
NEGATIVE_CONTROL_PERCENTILE = 99.9
FORCE_GP30_AS_THRESHOLD_FLOOR = True
FORCE_GP30_ZERO_PUNCTA = True

COLOCALIZATION_DISTANCE_PX = 1


# -------------------------
# PUNCTA FILTER SETTINGS
# -------------------------

MIN_PUNCTA_SIZE = 3
MAX_PUNCTA_SIZE = 80
MIN_CIRCULARITY = 0.55
MIN_SOLIDITY = 0.75
MIN_ASPECT_RATIO = 0.35
MAX_ECCENTRICITY = 0.95


# -------------------------
# METADATA PARSING
# -------------------------
def parse_metadata(name):
    """
    Example filename:
        26018_F3-02.npy

    Returns:
        plate = "26018"
        well = "F3"
        field = 2
    """

    base = clean_name(name)

    match = re.match(
        r"^(?P<plate>[^_]+)_(?P<well>[A-H]\d{1,2})-(?P<field>\d+)$",
        base,
        flags=re.IGNORECASE,
    )

    if not match:
        logger.warning(
            f"Could not parse plate, well, and field from: {name}"
        )
        return None, None, None

    plate = match.group("plate")
    well = match.group("well").upper()
    field = int(match.group("field"))

    return plate, well, field

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
# METADATA CHECK / TEST SUBSET
# -------------------------
def save_metadata_parsing_table(images):
    rows = []

    for name in sorted(images.keys()):
        plate, well, field = parse_metadata(name)

        parse_ok = (
            plate is not None
            and well is not None
            and field is not None
        )

        rows.append({
            "image_name": name,
            "plate": plate,
            "well": well,
            "field": field,
            "parse_ok": parse_ok,
        })

    meta_df = pd.DataFrame(rows)

    output_path = os.path.join(
        OUTPUT_FOLDER,
        "metadata_parsing_check.csv"
    )

    meta_df.to_csv(
        output_path,
        index=False
    )

    bad = meta_df.loc[~meta_df["parse_ok"]]

    if not bad.empty:
        logger.warning(
            f"Metadata parsing failed for {len(bad)} images."
        )
        logger.warning(
            f"Examples:\n{bad.head(10).to_string(index=False)}"
        )
    else:
        logger.info(
            "Metadata parsing successful for all images."
        )

    logger.info(
        f"Saved metadata parsing table: {output_path}"
    )

    return meta_df



def subset_for_testing(images, masks):
    keep_names = []
    counts = {}

    for name in sorted(images.keys()):
        lipoprotein, peptide, pl = parse_metadata_from_name(name)

        if peptide not in TEST_PEPTIDES:
            continue
        if lipoprotein not in TEST_LIPOPROTEINS:
            continue

        key = (lipoprotein, peptide)
        counts[key] = counts.get(key, 0)

        if counts[key] < MAX_IMAGES_PER_GROUP:
            keep_names.append(name)
            counts[key] += 1

    images_sub = {k: images[k] for k in keep_names if k in images}
    masks_sub = {k: masks[k] for k in keep_names if k in masks}

    logger.info(f"TEST MODE: keeping {len(images_sub)} images out of {len(images)}")
    return images_sub, masks_sub


# -------------------------
# MASKS
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


def preprocess_images(images, quant_masks):
    filtered = {}

    for name, img in images.items():
        if name not in quant_masks:
            continue

        peptide_blur = filters.gaussian(
            img[PEPTIDE_CH].astype(np.float32),
            sigma=GAUSSIAN_SIGMA,
            preserve_range=True
        )

        lipo_blur = filters.gaussian(
            img[LIPO_CH].astype(np.float32),
            sigma=GAUSSIAN_SIGMA,
            preserve_range=True
        )

        filtered[name] = np.stack([
            peptide_blur,
            lipo_blur,
            quant_masks[name],
        ])

    logger.info("Preprocessed peptide and lipoprotein channels.")
    return filtered


# -------------------------
# THRESHOLDS
# -------------------------

def calculate_marker_thresholds(image_dict, marker="peptide"):
    if marker == "peptide":
        channel_idx = 0
        out_name = "peptide_thresholds.csv"
    elif marker == "lipoprotein":
        channel_idx = 1
        out_name = "lipoprotein_thresholds.csv"
    else:
        raise ValueError("marker must be peptide or lipoprotein")

    pixels_by_group = {}

    for name, img in image_dict.items():
        lipoprotein, peptide, concentration_nM = parse_metadata_from_name(name)
        group = peptide if marker == "peptide" else lipoprotein

        if group is None:
            logger.warning(f"Could not parse {marker} from name: {name}")
            continue

        channel_img = img[channel_idx]
        mask = img[2]

        vals = channel_img[mask > 0]

        if vals.size > 0:
            pixels_by_group.setdefault(group, []).append(vals)

    thresholds = {}
    rows = []

    for group, pixel_list in pixels_by_group.items():
        all_pixels = np.concatenate(pixel_list)

        if marker == "peptide" and group == NEGATIVE_CONTROL_PEPTIDE:
            percentile_used = NEGATIVE_CONTROL_PERCENTILE
        else:
            percentile_used = GLOBAL_THRESHOLD_PERCENTILE

        threshold = np.percentile(all_pixels, percentile_used)

        thresholds[group] = threshold

        rows.append({
            marker: group,
            "raw_threshold": threshold,
            "final_threshold": threshold,
            "percentile_used": percentile_used,
            "gaussian_sigma": GAUSSIAN_SIGMA,
            "n_pixels": all_pixels.size,
            "mean_intensity": np.mean(all_pixels),
            "median_intensity": np.median(all_pixels),
            "p99_intensity": np.percentile(all_pixels, 99),
            "p99_5_intensity": np.percentile(all_pixels, 99.5),
            "p99_9_intensity": np.percentile(all_pixels, 99.9),
        })

        logger.info(
            f"{marker} {group} raw threshold "
            f"{percentile_used} percentile = {threshold:.3f}"
        )

    threshold_df = pd.DataFrame(rows)

    if marker == "peptide" and FORCE_GP30_AS_THRESHOLD_FLOOR:
        gp30_threshold = thresholds.get(NEGATIVE_CONTROL_PEPTIDE, np.nan)

        if np.isnan(gp30_threshold):
            logger.warning(
                f"No {NEGATIVE_CONTROL_PEPTIDE} threshold found. "
                "Cannot apply negative-control threshold floor."
            )
        else:
            logger.info(
                f"Using {NEGATIVE_CONTROL_PEPTIDE} threshold floor = {gp30_threshold:.3f}"
            )

            for group in list(thresholds.keys()):
                if group != NEGATIVE_CONTROL_PEPTIDE:
                    old_threshold = thresholds[group]
                    new_threshold = max(old_threshold, gp30_threshold)
                    thresholds[group] = new_threshold

                    logger.info(
                        f"{group}: raw threshold={old_threshold:.3f}, "
                        f"final threshold={new_threshold:.3f}"
                    )

            threshold_df["negative_control_floor"] = gp30_threshold
            threshold_df["used_negative_control_floor"] = False

            for idx, row in threshold_df.iterrows():
                group = row[marker]
                final_threshold = thresholds[group]
                threshold_df.loc[idx, "final_threshold"] = final_threshold
                threshold_df.loc[idx, "used_negative_control_floor"] = (
                    group != NEGATIVE_CONTROL_PEPTIDE and
                    final_threshold > row["raw_threshold"]
                )

    threshold_df.to_csv(
        os.path.join(output_folder, out_name),
        index=False
    )

    return thresholds


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


def add_puncta_intensity_features(df_p, puncta_labels, peptide_img, lipo_img):
    stats = []

    for _, row in df_p.iterrows():
        label = int(row["puncta_label"])
        p_mask = puncta_labels == label

        peptide_vals = peptide_img[p_mask]
        lipo_vals = lipo_img[p_mask]

        stats.append({
            "peptide_punctum_intensity_mean": np.nanmean(peptide_vals),
            "peptide_punctum_intensity_median": np.nanmedian(peptide_vals),
            "peptide_punctum_intensity_max": np.nanmax(peptide_vals),
            "lipo_intensity_inside_peptide_punctum_mean": np.nanmean(lipo_vals),
            "lipo_intensity_inside_peptide_punctum_max": np.nanmax(lipo_vals),
        })

    return pd.concat(
        [df_p.reset_index(drop=True), pd.DataFrame(stats).reset_index(drop=True)],
        axis=1
    )


def add_peptide_lipo_colocalization(
    df_p,
    peptide_labels,
    lipo_labels,
    peptide_img,
    lipo_img
):
    lipo_binary = lipo_labels > 0

    footprint = np.ones(
        (
            COLOCALIZATION_DISTANCE_PX * 2 + 1,
            COLOCALIZATION_DISTANCE_PX * 2 + 1
        ),
        dtype=bool
    )

    lipo_dilated = morphology.binary_dilation(lipo_binary, footprint)

    stats = []

    for _, row in df_p.iterrows():
        label = int(row["puncta_label"])
        peptide_mask = peptide_labels == label

        exact_overlap_mask = peptide_mask & lipo_binary
        within_1px_mask = peptide_mask & lipo_dilated

        touching_lipo_labels = np.unique(lipo_labels[within_1px_mask])
        touching_lipo_labels = touching_lipo_labels[touching_lipo_labels != 0]

        stats.append({
            "peptide_punctum_coloc_exact": bool(exact_overlap_mask.any()),
            "peptide_punctum_coloc_within_1px": bool(within_1px_mask.any()),
            "peptide_punctum_exact_overlap_pixels": int(exact_overlap_mask.sum()),
            "peptide_punctum_within_1px_pixels": int(within_1px_mask.sum()),
            "n_lipoprotein_puncta_touching": int(len(touching_lipo_labels)),
            "lipoprotein_labels_touching": ";".join(map(str, touching_lipo_labels)),
            "peptide_punctum_mean_lipo_intensity": np.nanmean(lipo_img[peptide_mask]),
            "peptide_punctum_max_lipo_intensity": np.nanmax(lipo_img[peptide_mask]),
        })

    return pd.concat(
        [df_p.reset_index(drop=True), pd.DataFrame(stats).reset_index(drop=True)],
        axis=1
    )


# -------------------------
# FEATURE COLLECTION
# -------------------------

def collect_features(image_dict, peptide_thresholds, lipoprotein_thresholds):
    logger.info("collecting peptide/lipoprotein puncta and colocalization features...")

    results = []

    for name, img in image_dict.items():
        lipoprotein, peptide, concentration_nM = parse_metadata_from_name(name)

        if peptide is None or lipoprotein is None:
            logger.warning(f"Skipping {name}: could not parse peptide/lipoprotein")
            continue

        peptide_threshold = peptide_thresholds.get(peptide, np.nan)
        lipo_threshold = lipoprotein_thresholds.get(lipoprotein, np.nan)

        if FORCE_GP30_ZERO_PUNCTA and peptide == NEGATIVE_CONTROL_PEPTIDE:
            peptide_threshold = np.inf

        if np.isnan(peptide_threshold) or np.isnan(lipo_threshold):
            logger.warning(f"Skipping {name}: missing threshold")
            continue

        peptide_img = img[0]
        lipo_img = img[1]
        mask = img[2]

        unique_cells = np.unique(mask)
        unique_cells = unique_cells[unique_cells != 0]

        for lbl in unique_cells:
            cell_mask = mask == lbl

            peptide_vals = peptide_img[cell_mask]
            lipo_vals = lipo_img[cell_mask]

            if peptide_vals.size == 0:
                continue

            peptide_binary = (peptide_img > peptide_threshold) & cell_mask
            lipo_binary = (lipo_img > lipo_threshold) & cell_mask

            peptide_labels = morphology.label(peptide_binary)
            peptide_labels = remove_small_objects(
                peptide_labels,
                min_size=MIN_PUNCTA_SIZE
            )
            peptide_labels, df_peptide = filter_puncta_by_size_shape(peptide_labels)

            lipo_labels = morphology.label(lipo_binary)
            lipo_labels = remove_small_objects(
                lipo_labels,
                min_size=MIN_PUNCTA_SIZE
            )
            lipo_labels, df_lipo = filter_puncta_by_size_shape(lipo_labels)

            n_lipo_puncta = len(df_lipo)

            if df_peptide.empty:
                results.append(pd.DataFrame([{
                    "image_name": name,
                    "lipoprotein": lipoprotein,
                    "peptide": peptide,
                    "concentration_nM": concentration_nM,
                    "cell_number": lbl,
                    "cell_size": int(cell_mask.sum()),
                    "peptide_threshold": peptide_threshold,
                    "lipoprotein_threshold": lipo_threshold,
                    "cell_peptide_intensity_mean": peptide_vals.mean(),
                    "cell_lipoprotein_intensity_mean": lipo_vals.mean(),
                    "n_peptide_puncta": 0,
                    "n_lipoprotein_puncta": n_lipo_puncta,
                    "puncta_area": 0,
                    "peptide_punctum_coloc_exact": False,
                    "peptide_punctum_coloc_within_1px": False,
                }]))
                continue

            df = add_puncta_intensity_features(
                df_p=df_peptide,
                puncta_labels=peptide_labels,
                peptide_img=peptide_img,
                lipo_img=lipo_img
            )

            df = add_peptide_lipo_colocalization(
                df_p=df,
                peptide_labels=peptide_labels,
                lipo_labels=lipo_labels,
                peptide_img=peptide_img,
                lipo_img=lipo_img
            )

            df["image_name"] = name
            df["lipoprotein"] = lipoprotein
            df["peptide"] = peptide
            df["concentration_nM"] = concentration_nM
            df["cell_number"] = lbl
            df["cell_size"] = int(cell_mask.sum())
            df["peptide_threshold"] = peptide_threshold
            df["lipoprotein_threshold"] = lipo_threshold
            df["cell_peptide_intensity_mean"] = peptide_vals.mean()
            df["cell_lipoprotein_intensity_mean"] = lipo_vals.mean()
            df["n_peptide_puncta"] = len(df)
            df["n_lipoprotein_puncta"] = n_lipo_puncta

            results.append(df)

    logger.info("feature extraction done.")
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# -------------------------
# SUMMARY
# -------------------------

def make_cell_summary(features):
    rows = []

    for group_cols, g in features.groupby(
        ["image_name", "lipoprotein", "peptide", "concentration_nM", "cell_number"],
        dropna=False
    ):
        image_name, lipoprotein, peptide, concentration_nM, cell_number = group_cols

        peptide_rows = g[g["puncta_area"] > 0]

        n_peptide = len(peptide_rows)
        n_coloc_exact = peptide_rows["peptide_punctum_coloc_exact"].sum() if n_peptide > 0 else 0
        n_coloc_1px = peptide_rows["peptide_punctum_coloc_within_1px"].sum() if n_peptide > 0 else 0

        rows.append({
            "image_name": image_name,
            "lipoprotein": lipoprotein,
            "peptide": peptide,
            "concentration_nM": concentration_nM,
            "cell_number": cell_number,
            "cell_size": g["cell_size"].iloc[0],
            "n_peptide_puncta": n_peptide,
            "n_lipoprotein_puncta": g["n_lipoprotein_puncta"].iloc[0],
            "n_peptide_puncta_coloc_exact": int(n_coloc_exact),
            "n_peptide_puncta_coloc_within_1px": int(n_coloc_1px),
            "fraction_peptide_puncta_coloc_exact": n_coloc_exact / n_peptide if n_peptide > 0 else np.nan,
            "fraction_peptide_puncta_coloc_within_1px": n_coloc_1px / n_peptide if n_peptide > 0 else np.nan,
            "cell_peptide_intensity_mean": g["cell_peptide_intensity_mean"].iloc[0],
            "cell_lipoprotein_intensity_mean": g["cell_lipoprotein_intensity_mean"].iloc[0],
        })

    return pd.DataFrame(rows)


# -------------------------
# PROOF IMAGES
# -------------------------

def get_surviving_labels(channel_img, mask_labels, threshold):
    binary = (channel_img > threshold) & (mask_labels > 0)

    labels = morphology.label(binary)
    labels = remove_small_objects(labels, min_size=MIN_PUNCTA_SIZE)

    labels, df_p = filter_puncta_by_size_shape(labels)

    return labels


def generate_proofs(image_dict, peptide_thresholds, lipoprotein_thresholds):
    logger.info("Generating proof plots...")

    for name, img in image_dict.items():
        lipoprotein, peptide, concentration_nM = parse_metadata_from_name(name)

        if peptide is None or lipoprotein is None:
            continue

        peptide_threshold = peptide_thresholds.get(peptide, np.nan)
        lipo_threshold = lipoprotein_thresholds.get(lipoprotein, np.nan)

        if FORCE_GP30_ZERO_PUNCTA and peptide == NEGATIVE_CONTROL_PEPTIDE:
            peptide_threshold = np.inf

        if np.isnan(peptide_threshold) or np.isnan(lipo_threshold):
            continue

        peptide_img = img[0]
        lipo_img = img[1]
        mask = img[2]

        peptide_labels = get_surviving_labels(
            peptide_img,
            mask,
            peptide_threshold
        )

        lipo_labels = get_surviving_labels(
            lipo_img,
            mask,
            lipo_threshold
        )

        lipo_dilated = morphology.binary_dilation(
            lipo_labels > 0,
            np.ones((3, 3), dtype=bool)
        )

        peptide_coloc = (peptide_labels > 0) & lipo_dilated

        contours = measure.find_contours((mask > 0).astype(int), 0.8)
        contours = [c for c in contours if len(c) >= 100]

        fig, axes = plt.subplots(1, 4, figsize=(20, 6))

        axes[0].imshow(peptide_img, cmap="gray_r")
        axes[0].set_title(f"{PEPTIDE_CH_NAME}: {peptide}")
        axes[0].axis("off")

        axes[1].imshow(lipo_img, cmap="gray_r")
        axes[1].set_title(f"{LIPO_CH_NAME}: {lipoprotein}")
        axes[1].axis("off")

        axes[2].imshow(peptide_img, cmap="gray_r")
        for c in measure.find_contours((peptide_labels > 0).astype(float), 0.5):
            axes[2].plot(c[:, 1], c[:, 0], color="red", lw=1.0)
        for line in contours:
            axes[2].plot(line[:, 1], line[:, 0], c="k", lw=0.6)
        axes[2].set_title("Peptide puncta")
        axes[2].axis("off")

        axes[3].imshow(lipo_img, cmap="gray_r")
        for c in measure.find_contours((lipo_labels > 0).astype(float), 0.5):
            axes[3].plot(c[:, 1], c[:, 0], color="blue", lw=1.0)
        for c in measure.find_contours(peptide_coloc.astype(float), 0.5):
            axes[3].plot(c[:, 1], c[:, 0], color="red", lw=1.0)
        axes[3].set_title("Lipo puncta + coloc peptide pixels")
        axes[3].axis("off")

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

        axes[0].add_artist(scalebar)

        fig.suptitle(
            f"{name} | {lipoprotein} + {peptide} | "
            f"peptide thr={peptide_threshold}, lipo thr={lipo_threshold:.1f}",
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

    filtered = preprocess_images(
        images=images,
        quant_masks=quant_masks
    )

    peptide_thresholds = calculate_marker_thresholds(
        filtered,
        marker="peptide"
    )

    lipoprotein_thresholds = calculate_marker_thresholds(
        filtered,
        marker="lipoprotein"
    )

    features = collect_features(
        image_dict=filtered,
        peptide_thresholds=peptide_thresholds,
        lipoprotein_thresholds=lipoprotein_thresholds
    )

    if features.empty:
        logger.warning("No cells/features detected; nothing to save.")
        sys.exit(0)

    cols_to_drop = [col for col in features.columns if "_coords" in col]
    features = features.drop(columns=cols_to_drop)

    features.to_csv(
        os.path.join(output_folder, "peptide_lipoprotein_puncta_features.csv"),
        index=False
    )

    cell_summary = make_cell_summary(features)

    cell_summary.to_csv(
        os.path.join(output_folder, "peptide_lipoprotein_cell_summary.csv"),
        index=False
    )

    generate_proofs(
        image_dict=filtered,
        peptide_thresholds=peptide_thresholds,
        lipoprotein_thresholds=lipoprotein_thresholds
    )

    logger.info("saved peptide_lipoprotein_puncta_features.csv")
    logger.info("saved peptide_lipoprotein_cell_summary.csv")
    logger.info("pipeline complete.")