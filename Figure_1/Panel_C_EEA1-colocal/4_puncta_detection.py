"""
Peptide–EEA1 puncta overlap analysis using Yen thresholding only for both
peptide and EEA1 puncta, filename-derived metadata, configurable object
filters, a global GP30 peptide-punctum intensity floor, and proof images.

Expected filenames follow the merged convention, for example:
- ARMIN-REP1.npy
- ARMIN-REP1-SET1.npy
- ARMIN-REP1-SET2.npy

For SET filenames, the peptide remains ARMIN and the biological replicate
remains REP1. SET1/SET2 only distinguish separately acquired source files.
No peptide map or threshold-method CSV is used.
"""

from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy import ndimage as ndi
from skimage import filters, measure, morphology


# === CONFIG


IMAGE_FOLDER = Path("results/initial_cleanup_merged")
MASK_FOLDER = Path("results/napari_masking_merged")
OUTPUT_FOLDER = Path("results/peptide_eea1_yen_only_global_gp30_v5")
PROOF_FOLDER = OUTPUT_FOLDER / "proofs"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
PROOF_FOLDER.mkdir(parents=True, exist_ok=True)

TEST_MODE = False 
TEST_N_IMAGES = 10
TEST_IMAGE_NAMES = []  # exact stems; empty means first TEST_N_IMAGES

PEPTIDE_CH = 1
EEA1_CH = 2
QUANT_REGION = "cell"  # "cell" or "nucleus"

SMALL_SMOOTHING_SIGMA = 0.7
PEPTIDE_BACKGROUND_SIGMA = 4.0
EEA1_BACKGROUND_SIGMA = 4.0
USE_BACKGROUND_SUBTRACTED_PEPTIDE = True
USE_BACKGROUND_SUBTRACTED_EEA1 = True
# Yen is the only thresholding method used for both puncta channels.
YEN_MULTIPLIER = 2.0
EEA1_YEN_MULTIPLIER = 1.0

# The GP30 floor is calculated from all shape-filtered, Yen-detected GP30
# puncta across all GP30 images, even in test mode. The single global floor is
# the mean of those puncta's mean intensities.
USE_GP30_AS_PEPTIDE_FLOOR = True
FORCE_GP30_ZERO_PUNCTA = True

PEPTIDE_FILTERS = {
    "min_size": 4,
    "max_size": 150,
    "min_circularity": 0.20,
    "min_solidity": 0.40,
    "min_aspect_ratio": 0.15,
    "max_eccentricity": 0.99,
}

# Edit these to tune EEA1 detection.
EEA1_FILTERS = {
    "min_size": 5,
    "max_size": 250,
    "min_circularity": 0.10,
    "min_solidity": 0.30,
    "min_aspect_ratio": 0.10,
    "max_eccentricity": 1.00,
}

COLOCALIZATION_DISTANCE_PX = 2

MAKE_PROOFS = True
PROOF_AUTOSCALE = True
PROOF_LOW_PERCENTILE = 1.0
PROOF_HIGH_PERCENTILE = 99.8
PROOF_CELL_BOUNDARY_COLOR = "yellow"
PROOF_CELL_BOUNDARY_WIDTH = 0.7


# ============================================================
# METADATA AND LOOKUPS
# ============================================================

def clean_name(name):
    return Path(str(name)).name.removesuffix(".npy").removesuffix("_mask")


def parse_metadata(name):
    base = clean_name(name).strip()
    parts = [part.strip() for part in base.split("-") if part.strip()]
    if not parts:
        return None, None, None

    peptide = parts[0].upper()
    replicate = next(
        (part.upper() for part in parts[1:] if re.fullmatch(r"REP\d+", part, re.I)),
        "UNKNOWN",
    )
    field = next(
        (
            int(part)
            for part in reversed(parts[1:])
            if re.fullmatch(r"\d+", part)
        ),
        None,
    )
    return peptide, replicate, field


def get_peptide_name(image_name, peptide_lookup=None):
    peptide, _, _ = parse_metadata(image_name)
    return peptide


# ============================================================
# LOADING AND PREPROCESSING
# ============================================================

def load_images():
    images = {}
    for path in sorted(IMAGE_FOLDER.glob("*.npy")):
        if path.name.endswith("_mask.npy"):
            continue
        try:
            image = np.load(path, mmap_mode="r")
        except Exception:
            logger.exception(f"Could not load image: {path}")
            continue
        if image.ndim != 3 or image.shape[0] <= max(PEPTIDE_CH, EEA1_CH):
            logger.warning(f"Skipping {path.name}: invalid shape {image.shape}")
            continue
        images[path.stem] = image
    logger.info(f"Loaded {len(images)} images")
    return images


def load_masks():
    masks = {}
    for path in sorted(MASK_FOLDER.glob("*_mask.npy")):
        name = path.name.removesuffix("_mask.npy")
        try:
            masks[name] = np.load(path, allow_pickle=True)
        except Exception:
            logger.exception(f"Could not load mask: {path}")
    logger.info(f"Loaded {len(masks)} masks")
    return masks


def get_quant_mask(mask_array):
    if mask_array.ndim == 2:
        if QUANT_REGION != "cell":
            raise ValueError("A 2D mask only supports QUANT_REGION='cell'")
        return np.asarray(mask_array, dtype=np.int32)
    if mask_array.ndim != 3:
        raise ValueError(f"Unsupported mask shape: {mask_array.shape}")
    if QUANT_REGION == "cell":
        return np.asarray(mask_array[0], dtype=np.int32)
    if QUANT_REGION == "nucleus":
        return measure.label(np.asarray(mask_array[1]) > 0)
    raise ValueError("QUANT_REGION must be 'cell' or 'nucleus'")


def preprocess_pair(name, image, mask_array):
    mask = get_quant_mask(mask_array)
    peptide_raw = np.asarray(image[PEPTIDE_CH], dtype=np.float32)
    eea1_raw = np.asarray(image[EEA1_CH], dtype=np.float32)

    if peptide_raw.shape != mask.shape or eea1_raw.shape != mask.shape:
        logger.warning(
            f"Skipping {name}: shape mismatch peptide={peptide_raw.shape}, "
            f"EEA1={eea1_raw.shape}, mask={mask.shape}"
        )
        return None

    peptide_smoothed = filters.gaussian(
        peptide_raw, sigma=SMALL_SMOOTHING_SIGMA, preserve_range=True
    ).astype(np.float32)
    eea1_smoothed = filters.gaussian(
        eea1_raw, sigma=SMALL_SMOOTHING_SIGMA, preserve_range=True
    ).astype(np.float32)

    peptide_background = filters.gaussian(
        peptide_smoothed, sigma=PEPTIDE_BACKGROUND_SIGMA, preserve_range=True
    )
    peptide_enhanced = peptide_smoothed - peptide_background
    peptide_enhanced[peptide_enhanced < 0] = 0

    eea1_background = filters.gaussian(
        eea1_smoothed, sigma=EEA1_BACKGROUND_SIGMA, preserve_range=True
    )
    eea1_enhanced = eea1_smoothed - eea1_background
    eea1_enhanced[eea1_enhanced < 0] = 0

    return {
        "peptide_raw": peptide_raw,
        "peptide_smoothed": peptide_smoothed,
        "peptide_enhanced": peptide_enhanced.astype(np.float32),
        "eea1_raw": eea1_raw,
        "eea1_smoothed": eea1_smoothed,
        "eea1_enhanced": eea1_enhanced.astype(np.float32),
        "mask": mask,
    }


def preprocess_all(images, masks):
    common = sorted(set(images) & set(masks))
    processed = {}
    for i, name in enumerate(common, start=1):
        logger.info(f"Preprocessing [{i}/{len(common)}] {name}")
        data = preprocess_pair(name, images[name], masks[name])
        if data is not None:
            processed[name] = data
    logger.info(f"Preprocessed {len(processed)} image/mask pairs")
    return processed


def choose_analysis_names(processed_names):
    names = sorted(processed_names)
    if not TEST_MODE:
        return names
    if TEST_IMAGE_NAMES:
        requested = {clean_name(name) for name in TEST_IMAGE_NAMES}
        selected = [name for name in names if name in requested]
    else:
        selected = names[:TEST_N_IMAGES]
    logger.info(f"TEST_MODE: analyzing {len(selected)} images")
    return selected


# ============================================================
# THRESHOLDING
# ============================================================

def get_peptide_analysis_image(data):
    return data["peptide_enhanced"] if USE_BACKGROUND_SUBTRACTED_PEPTIDE else data["peptide_smoothed"]


def get_eea1_analysis_image(data):
    return data["eea1_enhanced"] if USE_BACKGROUND_SUBTRACTED_EEA1 else data["eea1_smoothed"]


def calculate_gp30_floors(processed):
    rows = []
    gp30_punctum_means = []
    for name, data in processed.items():
        if get_peptide_name(name) != "GP30":
            continue
        _, replicate, field = parse_metadata(name)
        image = get_peptide_analysis_image(data)
        for cell_number in np.unique(data["mask"]):
            if cell_number == 0:
                continue
            cell_mask = data["mask"] == cell_number
            binary, yen_threshold = make_yen_binary(
                image, cell_mask, YEN_MULTIPLIER
            )
            labels, puncta_df, _ = filter_binary(
                binary, image, PEPTIDE_FILTERS
            )
            if puncta_df.empty:
                continue
            for row in puncta_df.itertuples():
                mean_intensity = float(row.mean_intensity)
                gp30_punctum_means.append(mean_intensity)
                rows.append({
                    "scope": "punctum",
                    "image_name": name,
                    "replicate": replicate,
                    "field": field,
                    "cell_number": int(cell_number),
                    "punctum_label": int(row.label),
                    "yen_threshold": yen_threshold,
                    "punctum_mean_intensity": mean_intensity,
                })

    global_floor = (
        float(np.mean(gp30_punctum_means))
        if gp30_punctum_means
        else np.nan
    )
    if np.isfinite(global_floor):
        rows.append({
            "scope": "global", "image_name": "", "replicate": "ALL",
            "field": "", "cell_number": "", "punctum_label": "",
            "yen_threshold": "", "punctum_mean_intensity": global_floor,
        })
    else:
        raise RuntimeError(
            "No shape-filtered GP30 puncta were detected with Yen, so the "
            "global GP30 mean-punctum-intensity floor cannot be calculated."
        )

    pd.DataFrame(rows).to_csv(OUTPUT_FOLDER / "gp30_peptide_floors.csv", index=False)
    logger.info(
        f"Global GP30 floor: {global_floor} "
        f"(mean of {len(gp30_punctum_means)} GP30 punctum means)"
    )
    return {}, global_floor


def gp30_floor_for_image(name, replicate_floors, global_floor):
    if not USE_GP30_AS_PEPTIDE_FLOOR:
        return 0.0
    return float(global_floor) if np.isfinite(global_floor) else 0.0


def make_yen_binary(image, cell_mask, multiplier):
    pixels = image[cell_mask]
    pixels = pixels[np.isfinite(pixels)]

    if pixels.size == 0:
        return np.zeros_like(cell_mask, dtype=bool), np.nan

    threshold = float(filters.threshold_yen(pixels)) * float(multiplier)
    return (image > threshold) & cell_mask, threshold


# ============================================================
# OBJECT FILTERING
# ============================================================

def shape_table(labels, intensity_image=None):
    properties = [
        "label", "area", "perimeter", "eccentricity", "solidity",
        "major_axis_length", "minor_axis_length", "centroid",
    ]
    if intensity_image is not None:
        properties += ["mean_intensity", "max_intensity"]
    df = pd.DataFrame(measure.regionprops_table(
        labels, intensity_image=intensity_image, properties=properties
    ))
    if df.empty:
        return df
    df["circularity"] = 4 * np.pi * df["area"] / (df["perimeter"] ** 2 + 1e-9)
    df["aspect_ratio"] = df["minor_axis_length"] / (df["major_axis_length"] + 1e-9)
    return df


def filter_binary(binary, intensity_image, settings):
    binary = morphology.remove_small_objects(
        binary.astype(bool), min_size=settings["min_size"], connectivity=2
    )
    initial_labels = measure.label(binary, connectivity=2)
    df = shape_table(initial_labels, intensity_image)
    if df.empty:
        return np.zeros_like(initial_labels, dtype=np.int32), df, 0

    keep = (
        (df["area"] >= settings["min_size"])
        & (df["area"] <= settings["max_size"])
        & (df["circularity"] >= settings["min_circularity"])
        & (df["solidity"] >= settings["min_solidity"])
        & (df["aspect_ratio"] >= settings["min_aspect_ratio"])
        & (df["eccentricity"] <= settings["max_eccentricity"])
    )
    keep_labels = df.loc[keep, "label"].astype(int).to_numpy()
    final_labels = measure.label(np.isin(initial_labels, keep_labels), connectivity=2)
    return final_labels, shape_table(final_labels, intensity_image), int(initial_labels.max())


def filter_labels_by_mean_intensity(labels, intensity_image, intensity_floor):
    """Remove whole puncta whose mean intensity is at or below the floor."""
    if not USE_GP30_AS_PEPTIDE_FLOOR or intensity_floor <= 0 or labels.max() == 0:
        return labels, shape_table(labels, intensity_image)
    df = shape_table(labels, intensity_image)
    keep_labels = df.loc[
        df["mean_intensity"] > intensity_floor, "label"
    ].astype(int).to_numpy()
    filtered = measure.label(np.isin(labels, keep_labels), connectivity=2)
    return filtered, shape_table(filtered, intensity_image)


# ============================================================
# DETECTION AND OVERLAP
# ============================================================

def detect_full_image(
    name, data, peptide, replicate_floors, global_gp30,
):
    cell_labels = data["mask"]
    peptide_image = get_peptide_analysis_image(data)
    eea1_image = get_eea1_analysis_image(data)
    gp30_floor = gp30_floor_for_image(name, replicate_floors, global_gp30)

    peptide_full = np.zeros_like(cell_labels, dtype=np.int32)
    eea1_full = np.zeros_like(cell_labels, dtype=np.int32)
    peptide_next = 1
    eea1_next = 1
    peptide_thresholds = []
    eea1_thresholds = []

    for cell_number in np.unique(cell_labels):
        if cell_number == 0:
            continue
        cell_mask = cell_labels == cell_number

        if peptide == "GP30" and FORCE_GP30_ZERO_PUNCTA:
            peptide_binary = np.zeros_like(cell_mask, dtype=bool)
            peptide_threshold = np.inf
        else:
            peptide_binary, peptide_threshold = make_yen_binary(
                peptide_image, cell_mask, YEN_MULTIPLIER
            )

        peptide_labels, _, _ = filter_binary(
            peptide_binary, peptide_image, PEPTIDE_FILTERS
        )
        peptide_labels, _ = filter_labels_by_mean_intensity(
            peptide_labels, peptide_image, gp30_floor
        )
        eea1_binary, eea1_threshold = make_yen_binary(
            eea1_image, cell_mask, EEA1_YEN_MULTIPLIER
        )
        eea1_labels, _, _ = filter_binary(eea1_binary, eea1_image, EEA1_FILTERS)

        if np.isfinite(peptide_threshold):
            peptide_thresholds.append(peptide_threshold)
        if np.isfinite(eea1_threshold):
            eea1_thresholds.append(eea1_threshold)

        for local_label in np.unique(peptide_labels):
            if local_label == 0:
                continue
            peptide_full[peptide_labels == local_label] = peptide_next
            peptide_next += 1

        for local_label in np.unique(eea1_labels):
            if local_label == 0:
                continue
            eea1_full[eea1_labels == local_label] = eea1_next
            eea1_next += 1

    representative_threshold = (
        float(np.median(peptide_thresholds)) if peptide_thresholds else np.inf
    )
    representative_eea1_threshold = (
        float(np.median(eea1_thresholds)) if eea1_thresholds else np.inf
    )
    return {
        "peptide_labels": peptide_full,
        "eea1_labels": eea1_full,
        "peptide_threshold": representative_threshold,
        "gp30_floor": gp30_floor,
        "eea1_threshold": representative_eea1_threshold,
    }


def cell_overlap_metrics(peptide_labels, eea1_labels):
    peptide_binary = peptide_labels > 0
    eea1_binary = eea1_labels > 0
    intersection = peptide_binary & eea1_binary
    union = peptide_binary | eea1_binary
    p = int(peptide_binary.sum())
    l = int(eea1_binary.sum())
    i = int(intersection.sum())
    u = int(union.sum())
    return {
        "peptide_mask_pixels": p,
        "eea1_mask_pixels": l,
        "intersection_pixels": i,
        "fraction_peptide_pixels_overlapping_eea1": i / p if p else np.nan,
        "fraction_eea1_pixels_overlapping_peptide": i / l if l else np.nan,
        "jaccard": i / u if u else np.nan,
        "dice": 2 * i / (p + l) if p + l else np.nan,
    }


def puncta_rows_for_cell(
    name, peptide, replicate, field, cell_number,
    peptide_labels, eea1_labels, peptide_image, eea1_image,
    peptide_threshold, gp30_floor, eea1_threshold,
):
    rows = []
    eea1_binary = eea1_labels > 0
    if COLOCALIZATION_DISTANCE_PX > 0:
        size = 2 * COLOCALIZATION_DISTANCE_PX + 1
        eea1_near = morphology.binary_dilation(
            eea1_binary, footprint=np.ones((size, size), dtype=bool)
        )
    else:
        eea1_near = eea1_binary

    distance_to_eea1 = (
        ndi.distance_transform_edt(~eea1_binary)
        if eea1_binary.any()
        else np.full(eea1_binary.shape, np.nan)
    )

    for region in measure.regionprops(peptide_labels, intensity_image=peptide_image):
        p_mask = peptide_labels == region.label
        exact = p_mask & eea1_binary
        near = p_mask & eea1_near
        exact_labels = np.unique(eea1_labels[exact]); exact_labels = exact_labels[exact_labels > 0]
        near_labels = np.unique(eea1_labels[near]); near_labels = near_labels[near_labels > 0]
        area = int(p_mask.sum())
        exact_pixels = int(exact.sum())
        near_pixels = int(near.sum())

        rows.append({
            "image_name": name,
            "peptide": peptide,
            "replicate": replicate,
            "field": field,
            "cell_number": int(cell_number),
            "peptide_punctum_label": int(region.label),
            "peptide_punctum_area_px": area,
            "peptide_punctum_mean_intensity": region.mean_intensity,
            "peptide_punctum_max_intensity": region.max_intensity,
            "eea1_mean_inside_peptide_punctum": float(np.mean(eea1_image[p_mask])),
            "eea1_max_inside_peptide_punctum": float(np.max(eea1_image[p_mask])),
            "exact_overlap": exact_pixels > 0,
            "within_distance_overlap": near_pixels > 0,
            "exact_overlap_pixels": exact_pixels,
            "within_distance_pixels": near_pixels,
            "fraction_peptide_punctum_exact_overlap": exact_pixels / area,
            "fraction_peptide_punctum_within_distance": near_pixels / area,
            "nearest_eea1_distance_px": (
                float(np.nanmin(distance_to_eea1[p_mask])) if eea1_binary.any() else np.nan
            ),
            "n_eea1_puncta_exactly_overlapping": len(exact_labels),
            "n_eea1_puncta_within_distance": len(near_labels),
            "peptide_threshold_method": "yen",
            "peptide_yen_multiplier": YEN_MULTIPLIER,
            "peptide_numeric_threshold": peptide_threshold,
            "gp30_floor": gp30_floor,
            "eea1_threshold_method": "yen",
            "eea1_yen_multiplier": EEA1_YEN_MULTIPLIER,
            "eea1_threshold": eea1_threshold,
        })
    return rows


def analyze(
    processed, analysis_names, replicate_floors, global_gp30,
):
    puncta_rows = []
    cell_rows = []
    detections = {}
    eea1_threshold_rows = []

    for i, name in enumerate(analysis_names, start=1):
        data = processed[name]
        peptide = get_peptide_name(name)
        if peptide is None:
            logger.warning(f"Skipping {name}: peptide could not be parsed")
            continue
        logger.info(
            f"[{i}/{len(analysis_names)}] {name} | {peptide} | "
            f"peptide Yen x {YEN_MULTIPLIER} | "
            f"EEA1 Yen x {EEA1_YEN_MULTIPLIER}"
        )
        detection = detect_full_image(
            name, data, peptide, replicate_floors, global_gp30,
        )
        detections[name] = detection

        _, replicate, field = parse_metadata(name)
        eea1_threshold_rows.append({
            "image_name": name,
            "replicate": replicate,
            "field": field,
            "method": "yen",
            "multiplier": EEA1_YEN_MULTIPLIER,
            "threshold": detection["eea1_threshold"],
            "analysis_image": (
                "background_subtracted" if USE_BACKGROUND_SUBTRACTED_EEA1 else "smoothed_raw"
            ),
        })

        p_full = detection["peptide_labels"]
        l_full = detection["eea1_labels"]
        p_image = get_peptide_analysis_image(data)
        l_image = get_eea1_analysis_image(data)
        cell_labels = data["mask"]

        for cell_number in np.unique(cell_labels):
            if cell_number == 0:
                continue
            cell_mask = cell_labels == cell_number
            p_labels = measure.label((p_full > 0) & cell_mask, connectivity=2)
            l_labels = measure.label((l_full > 0) & cell_mask, connectivity=2)
            p_df = shape_table(p_labels, p_image)
            l_df = shape_table(l_labels, l_image)

            current = puncta_rows_for_cell(
                name, peptide, replicate, field, cell_number,
                p_labels, l_labels, p_image, l_image,
                detection["peptide_threshold"], detection["gp30_floor"],
                detection["eea1_threshold"],
            )
            puncta_rows.extend(current)

            n_peptide = len(p_df)
            n_eea1 = len(l_df)
            n_exact = sum(row["exact_overlap"] for row in current)
            n_near = sum(row["within_distance_overlap"] for row in current)

            row = {
                "image_name": name,
                "peptide": peptide,
                "replicate": replicate,
                "field": field,
                "cell_number": int(cell_number),
                "cell_area_px": int(cell_mask.sum()),
                "n_peptide_puncta": n_peptide,
                "n_eea1_puncta": n_eea1,
                "n_peptide_puncta_exact_overlap": n_exact,
                "n_peptide_puncta_within_distance": n_near,
                "fraction_peptide_puncta_exact_overlap": n_exact / n_peptide if n_peptide else np.nan,
                "fraction_peptide_puncta_within_distance": n_near / n_peptide if n_peptide else np.nan,
                "cell_mean_peptide_raw_intensity": float(np.mean(data["peptide_raw"][cell_mask])),
                "cell_mean_eea1_raw_intensity": float(np.mean(data["eea1_raw"][cell_mask])),
                "peptide_threshold_method": "yen",
                "peptide_yen_multiplier": YEN_MULTIPLIER,
                "peptide_numeric_threshold": detection["peptide_threshold"],
                "gp30_floor": detection["gp30_floor"],
                "eea1_threshold_method": "yen",
                "eea1_yen_multiplier": EEA1_YEN_MULTIPLIER,
                "eea1_threshold": detection["eea1_threshold"],
            }
            row.update(cell_overlap_metrics(p_labels, l_labels))
            cell_rows.append(row)

    pd.DataFrame(eea1_threshold_rows).to_csv(
        OUTPUT_FOLDER / "eea1_thresholds.csv", index=False
    )
    return pd.DataFrame(puncta_rows), pd.DataFrame(cell_rows), detections


# ============================================================
# PROOFS
# ============================================================

def display_limits(image, cell_labels):
    if not PROOF_AUTOSCALE:
        return None, None
    pixels = image[cell_labels > 0]
    pixels = pixels[np.isfinite(pixels)]
    if pixels.size == 0:
        return None, None
    vmin, vmax = np.percentile(pixels, [PROOF_LOW_PERCENTILE, PROOF_HIGH_PERCENTILE])
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def draw_cell_boundaries(axis, cell_labels):
    for cell_number in np.unique(cell_labels):
        if cell_number == 0:
            continue
        for contour in measure.find_contours((cell_labels == cell_number).astype(float), 0.5):
            axis.plot(
                contour[:, 1], contour[:, 0],
                color=PROOF_CELL_BOUNDARY_COLOR,
                linewidth=PROOF_CELL_BOUNDARY_WIDTH,
            )


def show_raw(axis, image, cell_labels):
    vmin, vmax = display_limits(image, cell_labels)
    axis.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
    draw_cell_boundaries(axis, cell_labels)


def draw_contours(axis, binary, color, linewidth):
    for contour in measure.find_contours(binary.astype(float), 0.5):
        axis.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth)


def generate_proofs(processed, analysis_names, detections):
    if not MAKE_PROOFS:
        return

    for i, name in enumerate(analysis_names, start=1):
        if name not in detections:
            continue
        data = processed[name]
        detection = detections[name]
        peptide = get_peptide_name(name)
        cell_labels = data["mask"]
        p_binary = detection["peptide_labels"] > 0
        l_binary = detection["eea1_labels"] > 0
        exact = p_binary & l_binary

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        show_raw(axes[0], data["peptide_raw"], cell_labels)
        axes[0].set_title(f"Peptide raw: {peptide}")

        show_raw(axes[1], data["eea1_raw"], cell_labels)
        axes[1].set_title("EEA1 raw")

        show_raw(axes[2], data["peptide_raw"], cell_labels)
        draw_contours(axes[2], p_binary, "red", 0.8)
        axes[2].set_title(f"Peptide puncta: {detection['peptide_labels'].max()}")

        show_raw(axes[3], data["eea1_raw"], cell_labels)
        draw_contours(axes[3], l_binary, "cyan", 0.8)
        draw_contours(axes[3], exact, "red", 1.2)
        axes[3].set_title(f"EEA1 puncta: {detection['eea1_labels'].max()} + exact overlap")

        for axis in axes:
            axis.axis("off")

        fig.suptitle(
            f"{name} | {peptide}: Yen x {YEN_MULTIPLIER} | "
            f"peptide threshold={detection['peptide_threshold']:.2f} | "
            f"GP30 floor={detection['gp30_floor']:.2f} | "
            f"EEA1 Yen x {EEA1_YEN_MULTIPLIER} "
            f"threshold={detection['eea1_threshold']:.2f}"
        )
        fig.tight_layout()
        fig.savefig(PROOF_FOLDER / f"{name}_proof.png", dpi=250, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved proof [{i}/{len(analysis_names)}]: {name}")


# ============================================================
# VALIDATION AND MAIN
# ============================================================

def save_metadata_table(common_names):
    rows = []
    for name in common_names:
        peptide, replicate, field = parse_metadata(name)
        rows.append({
            "image_name": name,
            "replicate": replicate,
            "field": field,
            "peptide": peptide,
            "parse_ok": peptide is not None and replicate is not None,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FOLDER / "metadata_parsing_check.csv", index=False)
    return df


def main():
    images = load_images()
    masks = load_masks()

    common_names = sorted(set(images) & set(masks))
    if not common_names:
        raise RuntimeError("No matched image/mask pairs found")

    save_metadata_table(common_names)

    # All images are preprocessed so GP30 controls are available in test mode.
    processed = preprocess_all(images, masks)
    if not processed:
        raise RuntimeError("No images survived preprocessing")

    analysis_names = choose_analysis_names(processed.keys())
    replicate_floors, global_gp30 = calculate_gp30_floors(processed)
    puncta_df, cell_df, detections = analyze(
        processed, analysis_names, replicate_floors, global_gp30,
    )

    puncta_df.to_csv(OUTPUT_FOLDER / "peptide_eea1_puncta_features.csv", index=False)
    cell_df.to_csv(OUTPUT_FOLDER / "peptide_eea1_cell_summary.csv", index=False)
    generate_proofs(processed, analysis_names, detections)

    logger.info(f"Saved outputs to: {OUTPUT_FOLDER}")
    logger.info("Pipeline complete")


if __name__ == "__main__":
    main()