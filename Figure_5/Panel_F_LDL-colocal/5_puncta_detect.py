"""
Peptide–LDL puncta overlap analysis using a plate map, a peptide-specific
threshold-method CSV, percentile thresholding for LDL, configurable object
filters, a GP30 negative-control floor, and four-panel proof images.

MAP_CSV columns: plate, well, peptide
METHOD_CSV columns: peptide, method, parameter

Supported peptide methods:
- percentile: parameter is percentile, e.g. 98.5
- yen: parameter is multiplier, e.g. 1.0
- local_mean_std: parameter is local SD multiplier, e.g. 2.5
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy import ndimage as ndi
from skimage import filters, measure, morphology


# ============================================================
# CONFIG
# ============================================================

IMAGE_FOLDER = Path("results/initial_cleanup")
MASK_FOLDER = Path("results/napari_masking")
MAP_CSV = Path("raw_data/26018_26042_MAP.csv")
METHOD_CSV = Path("raw_data/26018_threshold_methods.csv")

OUTPUT_FOLDER = Path("results/peptide_ldl_overlap_selected_methods")
PROOF_FOLDER = OUTPUT_FOLDER / "proofs"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
PROOF_FOLDER.mkdir(parents=True, exist_ok=True)

TEST_MODE = False
TEST_N_IMAGES = 10
TEST_IMAGE_NAMES = []  # exact stems; empty means first TEST_N_IMAGES

PEPTIDE_CH = 2
LDL_CH = 0
QUANT_REGION = "cell"  # "cell" or "nucleus"

SMALL_SMOOTHING_SIGMA = 0.7
PEPTIDE_BACKGROUND_SIGMA = 4.0
LDL_BACKGROUND_SIGMA = 4.0
USE_BACKGROUND_SUBTRACTED_PEPTIDE = True
USE_BACKGROUND_SUBTRACTED_LDL = True
LOCAL_BLOCK_SIZE = 31

# GP30 floor is calculated from all mapped GP30 images, even in test mode.
USE_GP30_AS_PEPTIDE_FLOOR = True
GP30_FLOOR_PERCENTILE = 99.7
FORCE_GP30_ZERO_PUNCTA = True

# LDL percentile thresholding.
LDL_PERCENTILE = 99
LDL_THRESHOLD_SCOPE = "image"  # "image" or "global"

PEPTIDE_FILTERS = {
    "min_size": 4,
    "max_size": 150,
    "min_circularity": 0.20,
    "min_solidity": 0.40,
    "min_aspect_ratio": 0.15,
    "max_eccentricity": 0.99,
}

# Edit these to tune LDL detection.
LDL_FILTERS = {
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
    match = re.match(
        r"^(?P<plate>[A-Za-z0-9]+)[_-](?P<well>[A-Ha-h]\d{1,2})-(?P<field>\d+)$",
        base,
    )
    if not match:
        logger.warning(f"Could not parse metadata from: {base!r}")
        return None, None, None
    return (
        match.group("plate"),
        match.group("well").upper(),
        int(match.group("field")),
    )


def normalize_well(well):
    match = re.match(r"^([A-Ha-h])0*(\d+)$", str(well).strip())
    if not match:
        return str(well).strip().upper()
    return f"{match.group(1).upper()}{int(match.group(2))}"


def load_plate_map():
    df = pd.read_csv(MAP_CSV, dtype={"plate": str, "well": str, "peptide": str})
    required = {"plate", "well", "peptide"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Map CSV is missing columns: {sorted(missing)}")

    df["plate"] = df["plate"].astype(str).str.strip()
    df["well"] = df["well"].map(normalize_well)
    df["peptide"] = df["peptide"].astype(str).str.strip().str.upper()

    duplicated = df.duplicated(["plate", "well"], keep=False)
    if duplicated.any():
        raise ValueError(
            "Duplicate plate/well assignments in map:\n"
            + df.loc[duplicated].to_string(index=False)
        )

    lookup = {
        (str(row.plate), normalize_well(row.well)): row.peptide
        for row in df.itertuples()
    }
    logger.info(f"Loaded {len(lookup)} peptide-map assignments")
    return df, lookup


def load_method_lookup():
    df = pd.read_csv(METHOD_CSV)
    required = {"peptide", "method", "parameter"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Method CSV is missing columns: {sorted(missing)}")

    df["peptide"] = df["peptide"].astype(str).str.strip().str.upper()
    df["method"] = df["method"].astype(str).str.strip().str.lower()
    df["parameter"] = pd.to_numeric(df["parameter"], errors="raise")

    allowed = {"percentile", "yen", "local_mean_std"}
    invalid = df.loc[~df["method"].isin(allowed)]
    if not invalid.empty:
        raise ValueError("Unsupported methods:\n" + invalid.to_string(index=False))

    duplicated = df["peptide"].duplicated(keep=False)
    if duplicated.any():
        raise ValueError(
            "Each peptide must have one selected method:\n"
            + df.loc[duplicated].to_string(index=False)
        )

    lookup = {
        row.peptide: {"method": row.method, "parameter": float(row.parameter)}
        for row in df.itertuples()
    }
    logger.info(f"Loaded selected methods for {len(lookup)} peptides")
    return df, lookup


def get_peptide_name(image_name, peptide_lookup):
    plate, well, _ = parse_metadata(image_name)
    if plate is None or well is None:
        return None
    return peptide_lookup.get((str(plate), normalize_well(well)))


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
        if image.ndim != 3 or image.shape[0] <= max(PEPTIDE_CH, LDL_CH):
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
    if QUANT_REGION == "cell":
        return np.asarray(mask_array[0], dtype=np.int32)
    if QUANT_REGION == "nucleus":
        return measure.label(np.asarray(mask_array[1]) > 0)
    raise ValueError("QUANT_REGION must be 'cell' or 'nucleus'")


def preprocess_pair(name, image, mask_array):
    mask = get_quant_mask(mask_array)
    peptide_raw = np.asarray(image[PEPTIDE_CH], dtype=np.float32)
    ldl_raw = np.asarray(image[LDL_CH], dtype=np.float32)

    if peptide_raw.shape != mask.shape or ldl_raw.shape != mask.shape:
        logger.warning(
            f"Skipping {name}: shape mismatch peptide={peptide_raw.shape}, "
            f"LDL={ldl_raw.shape}, mask={mask.shape}"
        )
        return None

    peptide_smoothed = filters.gaussian(
        peptide_raw, sigma=SMALL_SMOOTHING_SIGMA, preserve_range=True
    ).astype(np.float32)
    ldl_smoothed = filters.gaussian(
        ldl_raw, sigma=SMALL_SMOOTHING_SIGMA, preserve_range=True
    ).astype(np.float32)

    peptide_background = filters.gaussian(
        peptide_smoothed, sigma=PEPTIDE_BACKGROUND_SIGMA, preserve_range=True
    )
    peptide_enhanced = peptide_smoothed - peptide_background
    peptide_enhanced[peptide_enhanced < 0] = 0

    ldl_background = filters.gaussian(
        ldl_smoothed, sigma=LDL_BACKGROUND_SIGMA, preserve_range=True
    )
    ldl_enhanced = ldl_smoothed - ldl_background
    ldl_enhanced[ldl_enhanced < 0] = 0

    return {
        "peptide_raw": peptide_raw,
        "peptide_smoothed": peptide_smoothed,
        "peptide_enhanced": peptide_enhanced.astype(np.float32),
        "ldl_raw": ldl_raw,
        "ldl_smoothed": ldl_smoothed,
        "ldl_enhanced": ldl_enhanced.astype(np.float32),
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


def get_ldl_analysis_image(data):
    return data["ldl_enhanced"] if USE_BACKGROUND_SUBTRACTED_LDL else data["ldl_smoothed"]


def local_mean_std_threshold(image, block_size, std_multiplier):
    local_mean = ndi.uniform_filter(image, size=block_size, mode="reflect")
    local_sq_mean = ndi.uniform_filter(image ** 2, size=block_size, mode="reflect")
    variance = local_sq_mean - local_mean ** 2
    variance[variance < 0] = 0
    return local_mean + std_multiplier * np.sqrt(variance)


def calculate_gp30_floors(processed, peptide_lookup):
    rows = []
    per_plate = {}
    all_floors = []

    for name, data in processed.items():
        if get_peptide_name(name, peptide_lookup) != "GP30":
            continue
        plate, well, field = parse_metadata(name)
        cell_mask = data["mask"] > 0
        image = get_peptide_analysis_image(data)
        pixels = image[cell_mask]
        pixels = pixels[np.isfinite(pixels)]
        if pixels.size == 0:
            continue
        floor = float(np.percentile(pixels, GP30_FLOOR_PERCENTILE))
        per_plate.setdefault(str(plate), []).append(floor)
        all_floors.append(floor)
        rows.append({
            "scope": "image", "image_name": name, "plate": plate,
            "well": normalize_well(well), "field": field,
            "percentile": GP30_FLOOR_PERCENTILE, "gp30_floor": floor,
            "n_pixels": int(pixels.size),
        })

    plate_floors = {}
    for plate, floors in per_plate.items():
        plate_floor = float(np.median(floors))
        plate_floors[plate] = plate_floor
        rows.append({
            "scope": "plate", "image_name": "", "plate": plate,
            "well": "", "field": "", "percentile": GP30_FLOOR_PERCENTILE,
            "gp30_floor": plate_floor, "n_pixels": "",
        })

    global_floor = float(np.median(all_floors)) if all_floors else np.nan
    if np.isfinite(global_floor):
        rows.append({
            "scope": "global", "image_name": "", "plate": "ALL",
            "well": "", "field": "", "percentile": GP30_FLOOR_PERCENTILE,
            "gp30_floor": global_floor, "n_pixels": "",
        })
    else:
        logger.warning("No GP30 controls were found")

    pd.DataFrame(rows).to_csv(OUTPUT_FOLDER / "gp30_peptide_floors.csv", index=False)
    logger.info(f"GP30 plate floors: {plate_floors}")
    logger.info(f"GP30 global floor: {global_floor}")
    return plate_floors, global_floor


def gp30_floor_for_image(name, plate_floors, global_floor):
    if not USE_GP30_AS_PEPTIDE_FLOOR:
        return 0.0
    plate, _, _ = parse_metadata(name)
    if str(plate) in plate_floors:
        return float(plate_floors[str(plate)])
    return float(global_floor) if np.isfinite(global_floor) else 0.0


def make_peptide_binary(image, cell_mask, config, gp30_floor):
    method = config["method"]
    parameter = float(config["parameter"])
    pixels = image[cell_mask]
    pixels = pixels[np.isfinite(pixels)]

    if pixels.size == 0:
        return np.zeros_like(cell_mask, dtype=bool), np.nan

    if method == "percentile":
        threshold = float(np.percentile(pixels, parameter))
        binary = (image > threshold) & cell_mask
        representative = threshold
    elif method == "yen":
        threshold = float(filters.threshold_yen(pixels)) * parameter
        binary = (image > threshold) & cell_mask
        representative = threshold
    elif method == "local_mean_std":
        threshold_image = local_mean_std_threshold(image, LOCAL_BLOCK_SIZE, parameter)
        binary = (image > threshold_image) & cell_mask
        representative = float(np.median(threshold_image[cell_mask]))
    else:
        raise ValueError(f"Unsupported peptide method: {method}")

    if USE_GP30_AS_PEPTIDE_FLOOR and gp30_floor > 0:
        binary &= image > gp30_floor

    return binary, representative


def calculate_global_ldl_threshold(processed):
    arrays = []
    for data in processed.values():
        mask = data["mask"] > 0
        pixels = get_ldl_analysis_image(data)[mask]
        pixels = pixels[np.isfinite(pixels)]
        if pixels.size:
            arrays.append(pixels)
    if not arrays:
        raise RuntimeError("No LDL pixels available for global thresholding")
    return float(np.percentile(np.concatenate(arrays), LDL_PERCENTILE))


def get_ldl_threshold(data, global_threshold):
    if LDL_THRESHOLD_SCOPE == "global":
        return float(global_threshold)
    if LDL_THRESHOLD_SCOPE != "image":
        raise ValueError("LDL_THRESHOLD_SCOPE must be 'image' or 'global'")
    mask = data["mask"] > 0
    pixels = get_ldl_analysis_image(data)[mask]
    pixels = pixels[np.isfinite(pixels)]
    return float(np.percentile(pixels, LDL_PERCENTILE)) if pixels.size else np.inf


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


# ============================================================
# DETECTION AND OVERLAP
# ============================================================

def detect_full_image(name, data, peptide, method_config, plate_floors, global_gp30, global_ldl):
    cell_labels = data["mask"]
    peptide_image = get_peptide_analysis_image(data)
    ldl_image = get_ldl_analysis_image(data)
    gp30_floor = gp30_floor_for_image(name, plate_floors, global_gp30)
    ldl_threshold = get_ldl_threshold(data, global_ldl)

    peptide_full = np.zeros_like(cell_labels, dtype=np.int32)
    ldl_full = np.zeros_like(cell_labels, dtype=np.int32)
    peptide_next = 1
    ldl_next = 1
    peptide_thresholds = []

    for cell_number in np.unique(cell_labels):
        if cell_number == 0:
            continue
        cell_mask = cell_labels == cell_number

        if peptide == "GP30" and FORCE_GP30_ZERO_PUNCTA:
            peptide_binary = np.zeros_like(cell_mask, dtype=bool)
            peptide_threshold = np.inf
        else:
            peptide_binary, peptide_threshold = make_peptide_binary(
                peptide_image, cell_mask, method_config, gp30_floor
            )

        peptide_labels, _, _ = filter_binary(
            peptide_binary, peptide_image, PEPTIDE_FILTERS
        )
        ldl_binary = (ldl_image > ldl_threshold) & cell_mask
        ldl_labels, _, _ = filter_binary(ldl_binary, ldl_image, LDL_FILTERS)

        if np.isfinite(peptide_threshold):
            peptide_thresholds.append(peptide_threshold)

        for local_label in np.unique(peptide_labels):
            if local_label == 0:
                continue
            peptide_full[peptide_labels == local_label] = peptide_next
            peptide_next += 1

        for local_label in np.unique(ldl_labels):
            if local_label == 0:
                continue
            ldl_full[ldl_labels == local_label] = ldl_next
            ldl_next += 1

    representative_threshold = (
        float(np.median(peptide_thresholds)) if peptide_thresholds else np.inf
    )
    return {
        "peptide_labels": peptide_full,
        "ldl_labels": ldl_full,
        "peptide_threshold": representative_threshold,
        "gp30_floor": gp30_floor,
        "ldl_threshold": ldl_threshold,
    }


def cell_overlap_metrics(peptide_labels, ldl_labels):
    peptide_binary = peptide_labels > 0
    ldl_binary = ldl_labels > 0
    intersection = peptide_binary & ldl_binary
    union = peptide_binary | ldl_binary
    p = int(peptide_binary.sum())
    l = int(ldl_binary.sum())
    i = int(intersection.sum())
    u = int(union.sum())
    return {
        "peptide_mask_pixels": p,
        "ldl_mask_pixels": l,
        "intersection_pixels": i,
        "fraction_peptide_pixels_overlapping_ldl": i / p if p else np.nan,
        "fraction_ldl_pixels_overlapping_peptide": i / l if l else np.nan,
        "jaccard": i / u if u else np.nan,
        "dice": 2 * i / (p + l) if p + l else np.nan,
    }


def puncta_rows_for_cell(
    name, peptide, plate, well, field, cell_number,
    peptide_labels, ldl_labels, peptide_image, ldl_image,
    method_config, peptide_threshold, gp30_floor, ldl_threshold,
):
    rows = []
    ldl_binary = ldl_labels > 0
    if COLOCALIZATION_DISTANCE_PX > 0:
        size = 2 * COLOCALIZATION_DISTANCE_PX + 1
        ldl_near = morphology.binary_dilation(
            ldl_binary, footprint=np.ones((size, size), dtype=bool)
        )
    else:
        ldl_near = ldl_binary

    distance_to_ldl = (
        ndi.distance_transform_edt(~ldl_binary)
        if ldl_binary.any()
        else np.full(ldl_binary.shape, np.nan)
    )

    for region in measure.regionprops(peptide_labels, intensity_image=peptide_image):
        p_mask = peptide_labels == region.label
        exact = p_mask & ldl_binary
        near = p_mask & ldl_near
        exact_labels = np.unique(ldl_labels[exact]); exact_labels = exact_labels[exact_labels > 0]
        near_labels = np.unique(ldl_labels[near]); near_labels = near_labels[near_labels > 0]
        area = int(p_mask.sum())
        exact_pixels = int(exact.sum())
        near_pixels = int(near.sum())

        rows.append({
            "image_name": name,
            "peptide": peptide,
            "plate": plate,
            "well": normalize_well(well),
            "field": field,
            "cell_number": int(cell_number),
            "peptide_punctum_label": int(region.label),
            "peptide_punctum_area_px": area,
            "peptide_punctum_mean_intensity": region.mean_intensity,
            "peptide_punctum_max_intensity": region.max_intensity,
            "ldl_mean_inside_peptide_punctum": float(np.mean(ldl_image[p_mask])),
            "ldl_max_inside_peptide_punctum": float(np.max(ldl_image[p_mask])),
            "exact_overlap": exact_pixels > 0,
            "within_distance_overlap": near_pixels > 0,
            "exact_overlap_pixels": exact_pixels,
            "within_distance_pixels": near_pixels,
            "fraction_peptide_punctum_exact_overlap": exact_pixels / area,
            "fraction_peptide_punctum_within_distance": near_pixels / area,
            "nearest_ldl_distance_px": (
                float(np.nanmin(distance_to_ldl[p_mask])) if ldl_binary.any() else np.nan
            ),
            "n_ldl_puncta_exactly_overlapping": len(exact_labels),
            "n_ldl_puncta_within_distance": len(near_labels),
            "peptide_threshold_method": method_config["method"],
            "peptide_threshold_parameter": method_config["parameter"],
            "peptide_numeric_threshold": peptide_threshold,
            "gp30_floor": gp30_floor,
            "ldl_percentile": LDL_PERCENTILE,
            "ldl_threshold_scope": LDL_THRESHOLD_SCOPE,
            "ldl_threshold": ldl_threshold,
        })
    return rows


def analyze(
    processed, analysis_names, peptide_lookup, method_lookup,
    plate_floors, global_gp30, global_ldl,
):
    puncta_rows = []
    cell_rows = []
    detections = {}
    ldl_threshold_rows = []

    for i, name in enumerate(analysis_names, start=1):
        data = processed[name]
        peptide = get_peptide_name(name, peptide_lookup)
        if peptide is None:
            logger.warning(f"Skipping {name}: no peptide map entry")
            continue
        if peptide not in method_lookup:
            logger.warning(f"Skipping {name}: no method for {peptide}")
            continue

        config = method_lookup[peptide]
        logger.info(
            f"[{i}/{len(analysis_names)}] {name} | {peptide} | "
            f"{config['method']} {config['parameter']}"
        )
        detection = detect_full_image(
            name, data, peptide, config, plate_floors, global_gp30, global_ldl
        )
        detections[name] = detection

        plate, well, field = parse_metadata(name)
        ldl_threshold_rows.append({
            "image_name": name,
            "plate": plate,
            "well": normalize_well(well),
            "field": field,
            "percentile": LDL_PERCENTILE,
            "threshold_scope": LDL_THRESHOLD_SCOPE,
            "threshold": detection["ldl_threshold"],
            "analysis_image": (
                "background_subtracted" if USE_BACKGROUND_SUBTRACTED_LDL else "smoothed_raw"
            ),
        })

        p_full = detection["peptide_labels"]
        l_full = detection["ldl_labels"]
        p_image = get_peptide_analysis_image(data)
        l_image = get_ldl_analysis_image(data)
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
                name, peptide, plate, well, field, cell_number,
                p_labels, l_labels, p_image, l_image, config,
                detection["peptide_threshold"], detection["gp30_floor"],
                detection["ldl_threshold"],
            )
            puncta_rows.extend(current)

            n_peptide = len(p_df)
            n_ldl = len(l_df)
            n_exact = sum(row["exact_overlap"] for row in current)
            n_near = sum(row["within_distance_overlap"] for row in current)

            row = {
                "image_name": name,
                "peptide": peptide,
                "plate": plate,
                "well": normalize_well(well),
                "field": field,
                "cell_number": int(cell_number),
                "cell_area_px": int(cell_mask.sum()),
                "n_peptide_puncta": n_peptide,
                "n_ldl_puncta": n_ldl,
                "n_peptide_puncta_exact_overlap": n_exact,
                "n_peptide_puncta_within_distance": n_near,
                "fraction_peptide_puncta_exact_overlap": n_exact / n_peptide if n_peptide else np.nan,
                "fraction_peptide_puncta_within_distance": n_near / n_peptide if n_peptide else np.nan,
                "cell_mean_peptide_raw_intensity": float(np.mean(data["peptide_raw"][cell_mask])),
                "cell_mean_ldl_raw_intensity": float(np.mean(data["ldl_raw"][cell_mask])),
                "peptide_threshold_method": config["method"],
                "peptide_threshold_parameter": config["parameter"],
                "peptide_numeric_threshold": detection["peptide_threshold"],
                "gp30_floor": detection["gp30_floor"],
                "ldl_percentile": LDL_PERCENTILE,
                "ldl_threshold_scope": LDL_THRESHOLD_SCOPE,
                "ldl_threshold": detection["ldl_threshold"],
            }
            row.update(cell_overlap_metrics(p_labels, l_labels))
            cell_rows.append(row)

    pd.DataFrame(ldl_threshold_rows).to_csv(
        OUTPUT_FOLDER / "ldl_thresholds.csv", index=False
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


def generate_proofs(processed, analysis_names, peptide_lookup, method_lookup, detections):
    if not MAKE_PROOFS:
        return

    for i, name in enumerate(analysis_names, start=1):
        if name not in detections:
            continue
        data = processed[name]
        detection = detections[name]
        peptide = get_peptide_name(name, peptide_lookup)
        config = method_lookup[peptide]
        cell_labels = data["mask"]
        p_binary = detection["peptide_labels"] > 0
        l_binary = detection["ldl_labels"] > 0
        exact = p_binary & l_binary

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        show_raw(axes[0], data["peptide_raw"], cell_labels)
        axes[0].set_title(f"Peptide raw: {peptide}")

        show_raw(axes[1], data["ldl_raw"], cell_labels)
        axes[1].set_title("LDL raw")

        show_raw(axes[2], data["peptide_raw"], cell_labels)
        draw_contours(axes[2], p_binary, "red", 0.8)
        axes[2].set_title(f"Peptide puncta: {detection['peptide_labels'].max()}")

        show_raw(axes[3], data["ldl_raw"], cell_labels)
        draw_contours(axes[3], l_binary, "cyan", 0.8)
        draw_contours(axes[3], exact, "red", 1.2)
        axes[3].set_title(f"LDL puncta: {detection['ldl_labels'].max()} + exact overlap")

        for axis in axes:
            axis.axis("off")

        fig.suptitle(
            f"{name} | {peptide}: {config['method']} {config['parameter']} | "
            f"peptide threshold={detection['peptide_threshold']:.2f} | "
            f"GP30 floor={detection['gp30_floor']:.2f} | "
            f"LDL p{LDL_PERCENTILE} threshold={detection['ldl_threshold']:.2f}"
        )
        fig.tight_layout()
        fig.savefig(PROOF_FOLDER / f"{name}_proof.png", dpi=250, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved proof [{i}/{len(analysis_names)}]: {name}")


# ============================================================
# VALIDATION AND MAIN
# ============================================================

def save_metadata_table(common_names, peptide_lookup):
    rows = []
    for name in common_names:
        plate, well, field = parse_metadata(name)
        peptide = get_peptide_name(name, peptide_lookup)
        rows.append({
            "image_name": name,
            "plate": plate,
            "well": normalize_well(well) if well else None,
            "field": field,
            "peptide": peptide,
            "parse_ok": plate is not None and well is not None and field is not None and peptide is not None,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FOLDER / "metadata_parsing_check.csv", index=False)
    return df


def validate_method_coverage(metadata_df, method_lookup):
    peptides = set(metadata_df.loc[metadata_df["parse_ok"], "peptide"].dropna())
    missing = sorted(peptides - set(method_lookup))
    if missing:
        raise ValueError(
            "METHOD_CSV is missing selected methods for:\n" + "\n".join(missing)
        )


def main():
    _, peptide_lookup = load_plate_map()
    _, method_lookup = load_method_lookup()
    images = load_images()
    masks = load_masks()

    common_names = sorted(set(images) & set(masks))
    if not common_names:
        raise RuntimeError("No matched image/mask pairs found")

    metadata_df = save_metadata_table(common_names, peptide_lookup)
    validate_method_coverage(metadata_df, method_lookup)

    # All images are preprocessed so GP30 controls are available in test mode.
    processed = preprocess_all(images, masks)
    if not processed:
        raise RuntimeError("No images survived preprocessing")

    analysis_names = choose_analysis_names(processed.keys())
    plate_floors, global_gp30 = calculate_gp30_floors(processed, peptide_lookup)
    global_ldl = (
        calculate_global_ldl_threshold(processed)
        if LDL_THRESHOLD_SCOPE == "global"
        else np.nan
    )

    puncta_df, cell_df, detections = analyze(
        processed, analysis_names, peptide_lookup, method_lookup,
        plate_floors, global_gp30, global_ldl,
    )

    puncta_df.to_csv(OUTPUT_FOLDER / "peptide_ldl_puncta_features.csv", index=False)
    cell_df.to_csv(OUTPUT_FOLDER / "peptide_ldl_cell_summary.csv", index=False)
    generate_proofs(processed, analysis_names, peptide_lookup, method_lookup, detections)

    logger.info(f"Saved outputs to: {OUTPUT_FOLDER}")
    logger.info("Pipeline complete")


if __name__ == "__main__":
    main()