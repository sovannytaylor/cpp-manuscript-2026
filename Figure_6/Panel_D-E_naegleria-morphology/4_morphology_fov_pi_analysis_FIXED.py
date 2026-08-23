"""
Analyze cell-mask morphology and image-level PI intensity.

Loops through multiple result folders:
- 26037_results
- 26037-4HR_results
- 26037BC_results

Uses a GLOBAL PI-positive threshold:
    PI_GLOBAL_THRESHOLD = 1167.72464189252
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from skimage import measure, morphology
from loguru import logger


logger.info("import ok")

# plotting setup
plt.rcParams.update({"font.size": 14})
sns.set_palette("Paired")


# -----------------------------
# CONFIG
# -----------------------------

PI_CHANNEL = 0  # channel 0 = PI channel
PI_GLOBAL_THRESHOLD = 1167.72464189252

SCALE_PX = 0.693  # size of one pixel in um
SCALE_UNIT = "um"

MIN_CELL_AREA_UM2 = 10
MIN_CELL_AREA_PX = MIN_CELL_AREA_UM2 / (SCALE_PX ** 2)

MASK_REGION = "cell"

RESULT_FOLDERS = [
    #Path("26037_results"),
    Path("26037-4HR_results"),
    Path("26037BC_results"),
]


# -----------------------------
# LOADERS
# -----------------------------

def load_images(image_folder):
    images = {}
    image_folder = Path(image_folder)

    if not image_folder.exists():
        logger.warning(f"Image folder does not exist: {image_folder}")
        return images

    for fn in image_folder.iterdir():
        if fn.suffix == ".npy":
            name = fn.stem
            images[name] = np.load(fn)

    return images


def load_masks(mask_folder):
    masks = {}
    mask_folder = Path(mask_folder)

    if not mask_folder.exists():
        logger.warning(f"Mask folder does not exist: {mask_folder}")
        return masks

    for fn in mask_folder.iterdir():
        if fn.name.endswith("_mask.npy"):
            name = fn.name.removesuffix("_mask.npy")
            masks[name] = np.load(fn, allow_pickle=True)

    return masks


# -----------------------------
# MASK PROCESSING
# -----------------------------

def build_quant_masks(masks, region="cell"):
    """
    Returns labeled 2D masks defining the objects whose morphology will be measured.
    """

    if region != "cell":
        raise ValueError("This dataset only has cell masks. Set MASK_REGION = 'cell'.")

    quant_masks = {}

    for name, m in masks.items():
        m = np.squeeze(m)

        while m.ndim > 2:
            m = m[0]

        if m.ndim != 2:
            raise ValueError(f"Mask for {name} is not 2D after squeezing: {m.shape}")

        quant_masks[name] = m.astype(int) if np.max(m) > 1 else morphology.label(m > 0)

    return quant_masks


def filter_small_masks(mask, min_area_px):
    """
    Removes labeled objects smaller than min_area_px.
    Relabels remaining objects sequentially.
    """

    filtered = np.zeros_like(mask, dtype=int)

    new_label = 1
    for region in measure.regionprops(mask):
        if region.area >= min_area_px:
            filtered[mask == region.label] = new_label
            new_label += 1

    return filtered


# -----------------------------
# IMAGE HELPERS
# -----------------------------

def get_channel(image, channel):
    """
    Return a 2D channel image from arrays saved by the cleanup script.
    Expected normal shape is (C, Y, X).
    """

    image = np.squeeze(image)

    if image.ndim == 2:
        if channel != 0:
            raise ValueError("Image is 2D, so only channel 0 is available.")
        return image

    if image.ndim == 3:
        if channel >= image.shape[0]:
            raise ValueError(f"Requested channel {channel}, but image shape is {image.shape}")
        return image[channel]

    while image.ndim > 3:
        image = image[0]

    if image.ndim == 3:
        if channel >= image.shape[0]:
            raise ValueError(f"Requested channel {channel}, but image shape is {image.shape}")
        return image[channel]

    raise ValueError(f"Unsupported image shape after squeeze: {image.shape}")


# -----------------------------
# FEATURE FUNCTIONS
# -----------------------------

def box_counting_fractal_dimension(binary_mask):
    """
    Estimate fractal dimension of a 2D binary object using box counting.
    """

    binary_mask = binary_mask.astype(bool)

    if binary_mask.sum() < 4:
        return np.nan

    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    cropped = binary_mask[np.ix_(rows, cols)]

    min_dim = min(cropped.shape)
    if min_dim < 4:
        return np.nan

    sizes = 2 ** np.arange(1, int(np.floor(np.log2(min_dim))) + 1)
    counts = []

    for size in sizes:
        n_rows = int(np.ceil(cropped.shape[0] / size))
        n_cols = int(np.ceil(cropped.shape[1] / size))

        padded = np.pad(
            cropped,
            (
                (0, n_rows * size - cropped.shape[0]),
                (0, n_cols * size - cropped.shape[1]),
            ),
            mode="constant",
            constant_values=False,
        )

        blocks = padded.reshape(n_rows, size, n_cols, size)
        block_has_object = blocks.any(axis=(1, 3))
        counts.append(block_has_object.sum())

    counts = np.asarray(counts)
    valid = counts > 0

    if valid.sum() < 2:
        return np.nan

    coeffs = np.polyfit(np.log(1 / sizes[valid]), np.log(counts[valid]), 1)
    return coeffs[0]


def collect_features(images, quant_masks):
    """
    Collects morphology features from each labeled cell mask.

    Uses GLOBAL PI threshold:
        pi_positive = pi_cell_intensity_mean > PI_GLOBAL_THRESHOLD
    """

    logger.info("collecting cell morphology features...")
    results = []

    properties = [
        "label",
        "area",
        "perimeter",
        "eccentricity",
        "solidity",
        "major_axis_length",
        "minor_axis_length",
        "bbox",
    ]

    for name, img in images.items():
        if name not in quant_masks:
            logger.warning(f"No mask found for image: {name}. Skipping.")
            continue

        pi_img = get_channel(img, PI_CHANNEL)
        mask = quant_masks[name]

        df = pd.DataFrame(measure.regionprops_table(mask, properties=properties))

        if df.empty:
            logger.warning(f"No objects found in mask for image: {name}.")
            continue

        df = df.rename(columns={"label": "cell_number"})

        # morphology
        df["circularity"] = (4 * np.pi * df["area"]) / (df["perimeter"] ** 2 + 1e-9)
        df["aspect_ratio"] = df["minor_axis_length"] / (df["major_axis_length"] + 1e-9)

        convexities = []
        fractal_dimensions = []

        pi_cell_means = []
        pi_cell_medians = []
        pi_cell_sums = []

        for _, row in df.iterrows():
            lbl = int(row["cell_number"])
            single_cell = mask == lbl

            convex_hull = morphology.convex_hull_image(single_cell)
            convex_perimeter = measure.perimeter(convex_hull)
            cell_perimeter = row["perimeter"]
            convexities.append(convex_perimeter / (cell_perimeter + 1e-9))

            fractal_dimensions.append(box_counting_fractal_dimension(single_cell))

            pi_values = pi_img[single_cell]

            pi_cell_means.append(np.nanmean(pi_values))
            pi_cell_medians.append(np.nanmedian(pi_values))
            pi_cell_sums.append(np.nansum(pi_values))

        df["convexity"] = convexities
        df["fractal_dimension"] = fractal_dimensions

        df["pi_cell_intensity_mean"] = pi_cell_means
        df["pi_cell_intensity_median"] = pi_cell_medians
        df["pi_cell_intensity_sum"] = pi_cell_sums

        # GLOBAL PI-positive threshold
        df["pi_positive_threshold"] = PI_GLOBAL_THRESHOLD
        df["pi_positive"] = df["pi_cell_intensity_mean"] > PI_GLOBAL_THRESHOLD

        percent_pi_positive = 100 * df["pi_positive"].mean()

        # FOV-level PI summary
        df["image_name"] = name
        df["pi_fov_intensity_mean"] = np.nanmean(pi_img)
        df["pi_fov_intensity_median"] = np.nanmedian(pi_img)
        df["pi_fov_intensity_sum"] = np.nansum(pi_img)
        df["n_cells_in_fov"] = len(df)
        df["n_pi_positive_in_fov"] = int(df["pi_positive"].sum())
        df["percent_pi_positive_in_fov"] = percent_pi_positive

        # physical units
        df["area_um2"] = df["area"] * (SCALE_PX ** 2)
        df["perimeter_um"] = df["perimeter"] * SCALE_PX

        results.append(df)

    logger.info("feature extraction done.")
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# -----------------------------
# PROOF PLOTS
# -----------------------------

def generate_proofs(images, quant_masks, proofs_folder):
    """
    Simple proof plot per image:
      1. PI channel raw
      2. cell mask labels
      3. PI channel with cell outlines
    """

    logger.info("generating proof plots...")

    proofs_folder = Path(proofs_folder)
    proofs_folder.mkdir(parents=True, exist_ok=True)

    for name, img in images.items():
        if name not in quant_masks:
            continue

        pi_img = get_channel(img, PI_CHANNEL)
        mask = quant_masks[name]
        contours = measure.find_contours((mask > 0).astype(int), 0.8)
        contours = [c for c in contours if len(c) >= 20]

        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 6))

        ax0.imshow(pi_img, cmap="gray_r")
        ax0.set_title("PI channel raw")
        ax0.axis("off")

        ax1.imshow(mask, cmap="nipy_spectral")
        ax1.set_title("Cell mask labels")
        ax1.axis("off")

        ax2.imshow(pi_img, cmap="gray_r")
        for line in contours:
            ax2.plot(line[:, 1], line[:, 0], c="k", lw=0.6)
        ax2.set_title("PI + cell outlines")
        ax2.axis("off")

        scalebar = ScaleBar(
            SCALE_PX,
            SCALE_UNIT,
            location="lower right",
            pad=0.3,
            sep=2,
            box_alpha=0,
            color="gray",
            length_fraction=0.3,
        )
        ax0.add_artist(scalebar)

        fig.suptitle(f"{name} | cells: {np.unique(mask).size - 1}", y=0.98)
        fig.tight_layout()

        fig.savefig(
            proofs_folder / f"{name}_morphology_proof.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

    logger.info("proofs saved.")


# -----------------------------
# MAIN LOOP
# -----------------------------

if __name__ == "__main__":

    for result_folder in RESULT_FOLDERS:
        logger.info("=" * 80)
        logger.info(f"Processing folder: {result_folder}")

        image_folder = result_folder / "initial_cleanup"
        mask_folder = result_folder / "napari_masking"
        output_folder = result_folder / "summary_calculations"
        proofs_folder = result_folder / "proofs"

        output_folder.mkdir(parents=True, exist_ok=True)
        proofs_folder.mkdir(parents=True, exist_ok=True)

        logger.info("loading images and masks...")
        images = load_images(image_folder)
        masks = load_masks(mask_folder)

        if not images:
            logger.warning(f"No images found in {image_folder}. Skipping {result_folder}.")
            continue

        if not masks:
            logger.warning(f"No masks found in {mask_folder}. Skipping {result_folder}.")
            continue

        quant_masks = build_quant_masks(masks, MASK_REGION)

        logger.info(f"Filtering masks smaller than {MIN_CELL_AREA_UM2} um²...")
        logger.info(f"Min area pixels: {MIN_CELL_AREA_PX:.2f}")

        quant_masks = {
            name: filter_small_masks(mask, MIN_CELL_AREA_PX)
            for name, mask in quant_masks.items()
        }

        features = collect_features(images, quant_masks)

        if features.empty:
            logger.warning(f"No cell masks found for {result_folder}; skipping.")
            continue

        generate_proofs(images, quant_masks, proofs_folder)

        logger.info("saving morphology features...")

        features.to_csv(
            output_folder / "cell_morphology_features.csv",
            index=False,
        )

        image_pi_summary = features[
            [
                "image_name",
                "pi_fov_intensity_mean",
                "pi_fov_intensity_median",
                "pi_fov_intensity_sum",
                "n_cells_in_fov",
                "n_pi_positive_in_fov",
                "percent_pi_positive_in_fov",
                "pi_positive_threshold",
            ]
        ].drop_duplicates()

        image_pi_summary.to_csv(
            output_folder / "image_level_pi_intensity.csv",
            index=False,
        )

        logger.info(f"Finished folder: {result_folder}")

    logger.info("=" * 80)
    logger.info("All folders complete.")