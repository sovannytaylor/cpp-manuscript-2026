"""
Quality control: use napari to validate individually saved
Cellpose-generated cell masks.
"""

import os
import numpy as np
from skimage.segmentation import clear_border
from loguru import logger
import napari
from qtpy.QtWidgets import QApplication


logger.info("import ok")


# ============================================================
# configuration
# ============================================================

image_folder = "results/initial_cleanup-02/"
mask_folder = "results/cellpose_masking-02/"
output_folder = "results/napari_masking-02/"

# Individual Cellpose masks are named:
# GR30_rep1_cellmask.npy
MASK_SUFFIX = "_cellmask.npy"

SATURATION_THRESHOLD = 60000
SATURATION_FRAC_CUTOFF = 0.05

BORDER_BUFFER_SIZE = 10

# Channel of interest for saturation and fluorescence checks.
# 0 means the first channel.
COI = 0

FLUORO_INTENSITY_THRESHOLD = 200

FLUORO_FRACTION_CUTOFF = 0.1

# Set to 1 for testing.
# Set to None to process every image.
TEST_N_IMAGES = 1


# ============================================================
# setup
# ============================================================

def ensure_output_folder(path):
    os.makedirs(
        path,
        exist_ok=True,
    )


# ============================================================
# IO
# ============================================================

def load_images(image_folder):
    """
    Load individually saved image arrays.

    Expected image shape:
        (channels, height, width)
    """

    images = {}

    for fname in sorted(os.listdir(image_folder)):
        if not fname.endswith(".npy"):
            continue

        image_name = fname.replace(
            ".npy",
            "",
        )

        image_path = os.path.join(
            image_folder,
            fname,
        )

        image = np.load(
            image_path
        )

        if image.ndim != 3:
            logger.warning(
                f"Skipping {fname}: expected a (C, Y, X) image, "
                f"but found shape {image.shape}"
            )
            continue

        images[image_name] = image

        logger.info(
            f"Loaded image: {image_name} | "
            f"shape={image.shape}"
        )

    return images


def load_masks(mask_folder):
    """
    Load individually saved Cellpose masks.

    Expected filenames:
        GR30_rep1_cellmask.npy

    Returns
    -------
    dict
        Keys match the corresponding image names:
        {
            "GR30_rep1": 2D mask,
            ...
        }
    """

    masks = {}

    for fname in sorted(os.listdir(mask_folder)):
        if not fname.endswith(MASK_SUFFIX):
            continue

        image_name = fname[
            :-len(MASK_SUFFIX)
        ]

        mask_path = os.path.join(
            mask_folder,
            fname,
        )

        mask = np.load(
            mask_path
        )

        # Handle an unnecessary singleton dimension, if present.
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]

        if mask.ndim != 2:
            logger.warning(
                f"Skipping {fname}: expected a 2D mask, "
                f"but found shape {mask.shape}"
            )
            continue

        masks[image_name] = mask.astype(
            np.int32,
            copy=False,
        )

        logger.info(
            f"Loaded mask: {image_name} | "
            f"shape={mask.shape}"
        )

    return masks


def save_mask(
    image_name,
    cell_mask,
):
    """
    Save the edited 2D cell mask.
    """

    out_path = os.path.join(
        output_folder,
        f"{image_name}_mask.npy",
    )

    np.save(
        out_path,
        cell_mask.astype(np.int32),
    )

    logger.info(
        f"Mask saved: {out_path} | "
        f"shape={cell_mask.shape}"
    )


# ============================================================
# mask filtering
# ============================================================

def remove_saturated_cells(
    image_stack,
    cells_mask,
    COI=COI,
):
    """
    Remove masks for saturated cells based on intensity threshold.
    """

    raw = image_stack[
        COI,
        :,
        :,
    ]

    valid_labels = []

    for label in np.unique(cells_mask)[1:]:
        pixel_mask = cells_mask == label

        pixel_count = np.count_nonzero(
            pixel_mask
        )

        if pixel_count == 0:
            continue

        saturated = np.count_nonzero(
            raw[pixel_mask] > SATURATION_THRESHOLD
        )

        saturated_fraction = (
            saturated / pixel_count
        )

        if saturated_fraction < SATURATION_FRAC_CUTOFF:
            valid_labels.append(
                label
            )

    filtered_cells = np.where(
        np.isin(
            cells_mask,
            valid_labels,
        ),
        cells_mask,
        0,
    )

    return filtered_cells.astype(
        np.int32,
        copy=False,
    )


def filter_cells_by_fluoro_expression(
    image_stack,
    cells_mask,
):
    """
    Keep only cells with significant fluorescence signal.
    """

    fluoro = image_stack[
        COI,
        :,
        :,
    ]

    valid_labels = []

    for label in np.unique(cells_mask)[1:]:
        pixel_mask = cells_mask == label

        pixel_count = np.count_nonzero(
            pixel_mask
        )

        if pixel_count == 0:
            continue

        fluoro_pixels = fluoro[
            pixel_mask
        ]

        bright_pixels = np.count_nonzero(
            fluoro_pixels
            > FLUORO_INTENSITY_THRESHOLD
        )

        bright_fraction = (
            bright_pixels / pixel_count
        )

        if bright_fraction > FLUORO_FRACTION_CUTOFF:
            valid_labels.append(
                label
            )

    filtered_cells = np.where(
        np.isin(
            cells_mask,
            valid_labels,
        ),
        cells_mask,
        0,
    )

    return filtered_cells.astype(
        np.int32,
        copy=False,
    )


def remove_border_objects(mask):
    """
    Remove cells touching the edge or border buffer.
    """

    return clear_border(
        mask,
        buffer_size=BORDER_BUFFER_SIZE,
    )


def filter_masks_auto(
    image_stack,
    cells_mask,
    filter_fluoro=False,
):
    """
    Apply automatic filtering to the 2D cell mask.
    """

    cells_filtered = remove_saturated_cells(
        image_stack,
        cells_mask,
    )

    cells_filtered = remove_border_objects(
        cells_filtered
    )

    if filter_fluoro:
        cells_filtered = filter_cells_by_fluoro_expression(
            image_stack,
            cells_filtered,
        )

    return cells_filtered.astype(
        np.int32,
        copy=False,
    )


# ============================================================
# manual QC
# ============================================================

def validate_with_napari(
    image_stack,
    image_name,
    cells_mask,
):
    """
    Launch Napari, allow the cell mask to be edited, and save it
    after the window closes.

    The fluorescence image remains one (C, Y, X) image layer.
    Napari therefore displays the channel axis as a slider, allowing
    you to move through the channels individually.
    """

    app = QApplication.instance()

    if not app:
        app = QApplication([])

    viewer = napari.Viewer()

    # Keep the channels together as one stack.
    # Do not use channel_axis=0 here.
    viewer.add_image(
        image_stack,
        name="image_stack",
    )

    # Only one mask layer: cells.
    viewer.add_labels(
        cells_mask,
        name="cells",
    )

    # Show and block until the window closes.
    viewer.window._qt_window.show()
    app.exec_()

    # Retrieve the edited 2D cell mask.
    cells = np.asarray(
        viewer.layers["cells"].data
    )

    save_mask(
        image_name,
        cells,
    )

    return cells


# ============================================================
# main QC pipeline
# ============================================================

def run_qc_pipeline(
    filter_fluoro=False,
):
    ensure_output_folder(
        output_folder
    )

    images = load_images(
        image_folder
    )

    masks = load_masks(
        mask_folder
    )

    image_names = sorted(
        set(images).intersection(masks)
    )

    missing_masks = sorted(
        set(images) - set(masks)
    )

    masks_without_images = sorted(
        set(masks) - set(images)
    )

    if missing_masks:
        logger.warning(
            "Images without matching masks:\n"
            + "\n".join(missing_masks)
        )

    if masks_without_images:
        logger.warning(
            "Masks without matching images:\n"
            + "\n".join(masks_without_images)
        )

    if not image_names:
        raise RuntimeError(
            "No matching image and Cellpose mask pairs were found."
        )

    logger.info(
        f"Found {len(image_names)} matching image-mask pairs"
    )

    already_filtered = {
        fname.replace(
            "_mask.npy",
            "",
        )
        for fname in os.listdir(output_folder)
        if fname.endswith("_mask.npy")
    }

    image_names = [
        name
        for name in image_names
        if name not in already_filtered
    ]

    if TEST_N_IMAGES is not None:
        image_names = image_names[
            :TEST_N_IMAGES
        ]

        logger.warning(
            f"TEST MODE: processing "
            f"{len(image_names)} image(s)"
        )

    logger.info(
        "Starting automated mask filtering"
    )

    for index, name in enumerate(
        image_names,
        start=1,
    ):
        logger.info(
            f"Processing {index}/{len(image_names)}: {name}"
        )

        image = images[
            name
        ]

        cells_mask = masks[
            name
        ]

        if image.shape[-2:] != cells_mask.shape:
            raise ValueError(
                f"Image-mask shape mismatch for {name}: "
                f"image={image.shape}, "
                f"mask={cells_mask.shape}"
            )

        filtered_mask = filter_masks_auto(
            image_stack=image,
            cells_mask=cells_mask,
            filter_fluoro=filter_fluoro,
        )

        logger.info(
            f"Opening Napari for {name}"
        )

        validate_with_napari(
            image_stack=image,
            image_name=name,
            cells_mask=filtered_mask,
        )

    logger.info(
        "Napari mask QC complete"
    )


# ============================================================
# entry point
# ============================================================

if __name__ == "__main__":
    run_qc_pipeline(
        filter_fluoro=True
    )