"""Grid-search peptide puncta thresholding for peptide + EEA1 images.

The peptide channel alone is used to select the best peptide puncta-detection
method for each peptide. EEA1 is loaded and validated for the later overlap
analysis, but it does not influence peptide thresholding.
"""

from pathlib import Path
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy import ndimage as ndi
from skimage import feature, filters, measure, morphology

# =========================
# CONFIG
# =========================
IMAGE_FOLDER = Path("results/initial_cleanup-02")
MASK_FOLDER = Path("results/napari_masking-02")
OUTPUT_FOLDER = Path("results/peptide_eea1_threshold_grid_search")
MONTAGE_FOLDER = OUTPUT_FOLDER / "montages"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
MONTAGE_FOLDER.mkdir(parents=True, exist_ok=True)

# Zero-based channel indices:
#   0 = first stored channel
#   1 = second stored channel (peptide)
#   2 = third stored channel (EEA1)
PEPTIDE_CH = 1
EEA1_CH = 2
PEPTIDES_TO_TEST = None  # None = all peptides found from the filenames
MAX_IMAGES_PER_PEPTIDE = 2

BACKGROUND_SIGMA = 4.0
SMALL_SMOOTHING_SIGMA = 0.7
LOCAL_BLOCK_SIZE = 31

DISPLAY_LOW_PERCENTILE = 1.0
DISPLAY_HIGH_PERCENTILE = 99.8

MIN_PUNCTUM_SIZE = 4
MAX_PUNCTUM_SIZE = 150
MIN_CIRCULARITY = 0.20
MIN_SOLIDITY = 0.40
MIN_ASPECT_RATIO = 0.15
MAX_ECCENTRICITY = 0.99
MAX_CELL_FRACTION_WARNING = 0.10
# ============================================================
# SMALL FIRST-PASS GRID: 7 CONFIGURATIONS
# ============================================================

GRID_CONFIGS = [
    {
        "method": "percentile",
        "percentile": 98.0,
    },
    {
        "method": "percentile",
        "percentile": 98.5,
    },
    {
        "method": "percentile",
        "percentile": 99.0,
    },
    {
        "method": "percentile",
        "percentile": 99.5,
    },
    {
        "method": "yen",
        "multiplier": 1.0,
    },
    {
        "method": "local_mean_std",
        "std_multiplier": 2.5,
        "block_size": LOCAL_BLOCK_SIZE,
    },
    {
        "method": "local_mean_std",
        "std_multiplier": 3.0,
        "block_size": LOCAL_BLOCK_SIZE,
    },
]

# =========================
# FILENAME METADATA
# =========================
def clean_name(name):
    return Path(str(name)).name.removesuffix(".npy").removesuffix("_mask")


def parse_metadata(name):
    base = clean_name(name).strip()
    parts = [part.strip() for part in base.split("-") if part.strip()]
    if not parts:
        return None, None

    peptide = parts[0].upper()
    replicate = next(
        (part.upper() for part in parts[1:] if re.fullmatch(r"REP\d+", part, re.I)),
        "UNKNOWN",
    )
    return peptide, replicate


# =========================
# LOADING
# =========================
def load_images_and_masks():
    records = []

    for image_path in sorted(IMAGE_FOLDER.glob("*.npy")):
        if image_path.name.endswith("_mask.npy"):
            continue

        name = image_path.stem
        peptide, replicate = parse_metadata(name)
        if peptide is None:
            logger.warning(f"Could not parse peptide from filename: {name}")
            continue

        if PEPTIDES_TO_TEST is not None and peptide not in PEPTIDES_TO_TEST:
            continue

        mask_path = MASK_FOLDER / f"{name}_mask.npy"
        if not mask_path.exists():
            logger.warning(f"No mask for {name}")
            continue

        try:
            image = np.load(image_path, mmap_mode="r")
            mask_array = np.load(mask_path, allow_pickle=True)
        except Exception:
            logger.exception(f"Could not load {name}")
            continue

        required_channel = max(PEPTIDE_CH, EEA1_CH)
        if image.ndim != 3 or image.shape[0] <= required_channel:
            logger.warning(f"Skipping {name}: image shape={image.shape}")
            continue

        raw = np.asarray(image[PEPTIDE_CH], dtype=np.float32)
        eea1_raw = np.asarray(image[EEA1_CH], dtype=np.float32)
        if mask_array.ndim == 2:
            cell_labels = np.asarray(mask_array, dtype=np.int32)
        elif mask_array.ndim == 3:
            cell_labels = np.asarray(mask_array[0], dtype=np.int32)
        else:
            logger.warning(
                f"Skipping {name}: unsupported mask shape={mask_array.shape}"
            )
            continue

        if raw.shape != cell_labels.shape or eea1_raw.shape != cell_labels.shape:
            logger.warning(
                f"Shape mismatch for {name}: "
                f"peptide={raw.shape}, EEA1={eea1_raw.shape}, "
                f"mask={cell_labels.shape}"
            )
            continue

        records.append({
            "image_name": name,
            "peptide": peptide,
            "replicate": replicate,
            "raw": raw,
            "eea1_raw": eea1_raw,
            "cell_labels": cell_labels,
        })

    logger.info(f"Loaded {len(records)} image/mask pairs")
    return records


# =========================
# PREPROCESSING
# =========================
def preprocess_image(raw):
    smoothed = filters.gaussian(
        raw,
        sigma=SMALL_SMOOTHING_SIGMA,
        preserve_range=True,
    )
    background = filters.gaussian(
        smoothed,
        sigma=BACKGROUND_SIGMA,
        preserve_range=True,
    )
    enhanced = smoothed - background
    enhanced[enhanced < 0] = 0
    return enhanced.astype(np.float32)


# =========================
# THRESHOLDING
# =========================
def safe_global_threshold(pixels, method):
    if pixels.size == 0:
        return np.inf
    try:
        return float({
            "otsu": filters.threshold_otsu,
            "yen": filters.threshold_yen,
            "triangle": filters.threshold_triangle,
            "li": filters.threshold_li,
        }[method](pixels))
    except Exception:
        logger.exception(f"Threshold method failed: {method}")
        return np.inf


def local_mean_std_threshold(image, block_size, std_multiplier):
    local_mean = ndi.uniform_filter(image, size=block_size, mode="reflect")
    local_squared_mean = ndi.uniform_filter(image ** 2, size=block_size, mode="reflect")
    local_variance = local_squared_mean - local_mean ** 2
    local_variance[local_variance < 0] = 0
    return local_mean + std_multiplier * np.sqrt(local_variance)


def config_name(config):
    parts = [config["method"]]
    parts.extend(f"{key}={value}" for key, value in config.items() if key != "method")
    return "__".join(parts)


def threshold_binary(enhanced, cell_mask, config):
    method = config["method"]
    pixels = enhanced[cell_mask]
    pixels = pixels[np.isfinite(pixels)]

    if pixels.size == 0:
        return np.zeros_like(cell_mask, dtype=bool), np.nan

    if method == "percentile":
        threshold = float(np.percentile(pixels, config["percentile"]))
        return (enhanced > threshold) & cell_mask, threshold

    if method in {"otsu", "yen", "triangle", "li"}:
        threshold = safe_global_threshold(pixels, method) * config["multiplier"]
        return (enhanced > threshold) & cell_mask, threshold

    if method == "local_mean_std":
        threshold_image = local_mean_std_threshold(
            enhanced,
            config["block_size"],
            config["std_multiplier"],
        )
        binary = (enhanced > threshold_image) & cell_mask
        return binary, float(np.median(threshold_image[cell_mask]))

    if method == "peak_local_max":
        threshold = float(np.percentile(pixels, config["percentile"]))
        coordinates = feature.peak_local_max(
            enhanced,
            min_distance=config["min_distance"],
            threshold_abs=threshold,
            labels=cell_mask.astype(np.uint8),
            exclude_border=False,
        )
        binary = np.zeros_like(cell_mask, dtype=bool)
        if coordinates.size > 0:
            binary[coordinates[:, 0], coordinates[:, 1]] = True
            binary = morphology.binary_dilation(binary, footprint=morphology.disk(1))
            binary &= cell_mask
        return binary, threshold

    if method == "log_blob":
        normalized = enhanced.copy()
        image_max = np.nanmax(normalized[cell_mask])
        if image_max > 0:
            normalized /= image_max

        blobs = feature.blob_log(
            normalized,
            min_sigma=config["min_sigma"],
            max_sigma=config["max_sigma"],
            num_sigma=6,
            threshold=config["blob_threshold"],
            overlap=0.5,
        )

        binary = np.zeros_like(cell_mask, dtype=bool)
        rr, cc = np.ogrid[:binary.shape[0], :binary.shape[1]]

        for y, x, sigma in blobs:
            y = int(round(y))
            x = int(round(x))
            if not (0 <= y < binary.shape[0] and 0 <= x < binary.shape[1]):
                continue
            if not cell_mask[y, x]:
                continue
            radius = max(1, int(round(np.sqrt(2) * sigma)))
            disk_mask = (rr - y) ** 2 + (cc - x) ** 2 <= radius ** 2
            binary[disk_mask & cell_mask] = True

        return binary, config["blob_threshold"]

    raise ValueError(f"Unknown threshold method: {method}")


# =========================
# OBJECT FILTERING
# =========================
def filter_objects(binary, enhanced):
    binary = morphology.remove_small_objects(
        binary,
        min_size=MIN_PUNCTUM_SIZE,
        connectivity=2,
    )

    initial_labels = measure.label(binary, connectivity=2)
    props = measure.regionprops_table(
        initial_labels,
        intensity_image=enhanced,
        properties=[
            "label", "area", "perimeter", "eccentricity", "solidity",
            "major_axis_length", "minor_axis_length", "mean_intensity",
            "max_intensity",
        ],
    )
    df = pd.DataFrame(props)

    if df.empty:
        return np.zeros_like(initial_labels, dtype=np.int32), df, 0

    df["circularity"] = 4 * np.pi * df["area"] / (df["perimeter"] ** 2 + 1e-9)
    df["aspect_ratio"] = df["minor_axis_length"] / (df["major_axis_length"] + 1e-9)

    keep = (
        (df["area"] >= MIN_PUNCTUM_SIZE)
        & (df["area"] <= MAX_PUNCTUM_SIZE)
        & (df["circularity"] >= MIN_CIRCULARITY)
        & (df["solidity"] >= MIN_SOLIDITY)
        & (df["aspect_ratio"] >= MIN_ASPECT_RATIO)
        & (df["eccentricity"] <= MAX_ECCENTRICITY)
    )

    filtered_binary = np.isin(initial_labels, df.loc[keep, "label"].astype(int).to_numpy())
    final_labels = measure.label(filtered_binary, connectivity=2)
    final_df = pd.DataFrame(measure.regionprops_table(
        final_labels,
        intensity_image=enhanced,
        properties=[
            "label", "area", "perimeter", "eccentricity", "solidity",
            "major_axis_length", "minor_axis_length", "mean_intensity",
            "max_intensity",
        ],
    ))

    return final_labels, final_df, int(initial_labels.max())


# =========================
# IMAGE SELECTION
# =========================
def select_images_per_peptide(records):
    metadata = pd.DataFrame([
        {key: record[key] for key in ["image_name", "peptide", "replicate"]}
        for record in records
    ])

    selected_names = []

    for _, peptide_df in metadata.groupby("peptide", sort=True):
        peptide_df = peptide_df.sort_values(["replicate", "image_name"])
        chosen_rows = [
            replicate_df.iloc[0]
            for _, replicate_df in peptide_df.groupby("replicate", sort=True)
        ]
        chosen_df = pd.DataFrame(chosen_rows)
        remaining = peptide_df.loc[~peptide_df["image_name"].isin(chosen_df["image_name"])]
        slots = MAX_IMAGES_PER_PEPTIDE - len(chosen_df)
        if slots > 0:
            chosen_df = pd.concat([chosen_df, remaining.head(slots)], ignore_index=True)
        selected_names.extend(chosen_df.head(MAX_IMAGES_PER_PEPTIDE)["image_name"].tolist())

    selected_df = metadata.loc[metadata["image_name"].isin(selected_names)].copy()
    selected_df.to_csv(OUTPUT_FOLDER / "peptide_image_selection.csv", index=False)
    return set(selected_names)


# =========================
# ANALYSIS
# =========================
def summarize_detection(record, config, numeric_threshold, labels, object_df, n_before_filter):
    cell_pixels = int((record["cell_labels"] > 0).sum())
    detected_pixels = int((labels > 0).sum())

    cell_ids = np.unique(record["cell_labels"])
    cell_ids = cell_ids[cell_ids > 0]
    n_cells = int(len(cell_ids))

    n_objects = int(labels.max())

    row = {
        "image_name": record["image_name"],
        "peptide": record["peptide"],
        "replicate": record["replicate"],
        "config_name": config_name(config),
        "method": config["method"],
        "numeric_threshold": numeric_threshold,
        "n_cells": n_cells,
        "n_objects_before_filter": n_before_filter,
        "n_objects_final": n_objects,
        "puncta_per_cell": n_objects / n_cells if n_cells else np.nan,
        "cell_pixels": cell_pixels,
        "detected_pixels": detected_pixels,
        "fraction_cell_detected": detected_pixels / cell_pixels if cell_pixels else np.nan,
        "oversegmentation_warning": (
            detected_pixels / cell_pixels > MAX_CELL_FRACTION_WARNING
            if cell_pixels else False
        ),
    }

    row.update({key: value for key, value in config.items() if key != "method"})

    if object_df.empty:
        row.update({
            "median_object_area": np.nan,
            "mean_object_area": np.nan,
            "median_object_intensity": np.nan,
            "median_object_max_intensity": np.nan,
        })
    else:
        row.update({
            "median_object_area": float(object_df["area"].median()),
            "mean_object_area": float(object_df["area"].mean()),
            "median_object_intensity": float(object_df["mean_intensity"].median()),
            "median_object_max_intensity": float(object_df["max_intensity"].median()),
        })

    return row


def analyze_grid(records):
    selected_names = select_images_per_peptide(records)
    selected_records = [r for r in records if r["image_name"] in selected_names]

    results = []
    proof_records = []
    total_jobs = len(selected_records) * len(GRID_CONFIGS)
    job_number = 0

    for image_i, record in enumerate(selected_records, start=1):
        logger.info(
            f"[IMAGE {image_i}/{len(selected_records)}] "
            f"{record['image_name']} | {record['peptide']}"
        )

        enhanced = preprocess_image(record["raw"])
        cell_mask = record["cell_labels"] > 0

        for config in GRID_CONFIGS:
            job_number += 1
            logger.info(
                f"[GRID {job_number}/{total_jobs}] "
                f"{record['peptide']} | {record['image_name']} | {config_name(config)}"
            )

            try:
                binary, numeric_threshold = threshold_binary(enhanced, cell_mask, config)
                labels, object_df, n_before = filter_objects(binary, enhanced)
            except Exception:
                logger.exception(f"Failed: {record['image_name']} | {config_name(config)}")
                continue

            results.append(summarize_detection(
                record,
                config,
                numeric_threshold,
                labels,
                object_df,
                n_before,
            ))

            proof_records.append({
                "image_name": record["image_name"],
                "peptide": record["peptide"],
                "raw": record["raw"],
                "cell_labels": record["cell_labels"],
                "labels": labels,
                "config": config,
                "n_objects": int(labels.max()),
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FOLDER / "grid_search_results.csv", index=False)
    return results_df, proof_records


# =========================
# MONTAGES
# =========================
def display_limits(image, cell_labels):
    pixels = image[cell_labels > 0]
    pixels = pixels[np.isfinite(pixels)]
    if pixels.size == 0:
        return None, None
    vmin, vmax = np.percentile(
        pixels,
        [DISPLAY_LOW_PERCENTILE, DISPLAY_HIGH_PERCENTILE],
    )
    if vmax <= vmin:
        vmax = vmin + 1
    return float(vmin), float(vmax)


def draw_boundaries(axis, labels, color, linewidth=0.7):
    for contour in measure.find_contours((labels > 0).astype(float), 0.5):
        axis.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth)


def generate_montages(proof_records):
    if not proof_records:
        logger.warning("No proof records were generated")
        return

    proof_df = pd.DataFrame([
        {
            "image_name": item["image_name"],
            "peptide": item["peptide"],
            "index": i,
        }
        for i, item in enumerate(proof_records)
    ])

    for peptide in sorted(proof_df["peptide"].unique()):
        peptide_folder = MONTAGE_FOLDER / peptide
        peptide_folder.mkdir(parents=True, exist_ok=True)
        peptide_df = proof_df.loc[proof_df["peptide"] == peptide]

        for image_name in sorted(peptide_df["image_name"].unique()):
            indices = peptide_df.loc[peptide_df["image_name"] == image_name, "index"].tolist()
            items = [proof_records[i] for i in indices]
            n_cols = 6
            n_rows = math.ceil(len(items) / n_cols)

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(n_cols * 4, n_rows * 4),
                squeeze=False,
            )

            for axis in axes.flat:
                axis.axis("off")

            for axis, item in zip(axes.flat, items):
                vmin, vmax = display_limits(item["raw"], item["cell_labels"])
                axis.imshow(item["raw"], cmap="gray", vmin=vmin, vmax=vmax)
                draw_boundaries(axis, item["cell_labels"], color="yellow", linewidth=0.5)
                draw_boundaries(axis, item["labels"], color="red", linewidth=0.8)
                axis.set_title(
                    f"{config_name(item['config'])}\nN={item['n_objects']}",
                    fontsize=8,
                )

            fig.suptitle(f"{peptide} | {image_name}", fontsize=14)
            fig.tight_layout()
            fig.savefig(
                peptide_folder / f"{image_name}_grid.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(fig)


# =========================
# SUMMARIES
# =========================
def summarize_by_peptide(results_df):
    if results_df.empty:
        return

    summary = (
        results_df.groupby(["peptide", "config_name", "method"], as_index=False)
        .agg(
            n_images=("image_name", "nunique"),
            median_puncta_per_cell=("puncta_per_cell", "median"),
            median_n_objects=("n_objects_final", "median"),
            median_fraction_cell_detected=("fraction_cell_detected", "median"),
            median_object_area=("median_object_area", "median"),
            median_object_intensity=("median_object_intensity", "median"),
            fraction_images_warned=("oversegmentation_warning", "mean"),
        )
    )
    summary.to_csv(OUTPUT_FOLDER / "grid_search_summary_by_peptide.csv", index=False)


def create_selection_template(results_df):
    selection = (
        results_df[["peptide", "config_name", "method"]]
        .drop_duplicates()
        .sort_values(["peptide", "method", "config_name"])
    )
    selection["selected"] = False
    selection["notes"] = ""
    selection.to_csv(OUTPUT_FOLDER / "peptide_method_selection_template.csv", index=False)


# =========================
# MAIN
# =========================
def main():
    records = load_images_and_masks()
    if not records:
        raise RuntimeError("No image/mask pairs were loaded")

    logger.info(f"Testing {len(GRID_CONFIGS)} configurations without Sauvola")

    results_df, proof_records = analyze_grid(records)
    if results_df.empty:
        raise RuntimeError("Grid search produced no results")

    summarize_by_peptide(results_df)
    create_selection_template(results_df)
    generate_montages(proof_records)

    logger.info(f"Grid search complete: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()