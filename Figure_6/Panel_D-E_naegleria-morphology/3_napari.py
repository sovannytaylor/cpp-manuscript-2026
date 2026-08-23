"""
Quality control: use napari to validate Cellpose-generated cell masks only.
"""

import os
import numpy as np
from skimage.segmentation import clear_border
from loguru import logger
import napari
from qtpy.QtWidgets import QApplication

logger.info('import ok')

# configuration
image_folder = '26037BC_results/initial_cleanup/'
mask_folder = '26037BC_results/cellpose_masking/'
output_folder = '26037BC_results/napari_masking/'
mask_filename = 'cellpose_cellmasks.npy'

SATURATION_THRESHOLD = 60000
SATURATION_FRAC_CUTOFF = 0.05
BORDER_BUFFER_SIZE = 10

COI = 1  # channel of interest for saturation / fluorescence check
FLUORO_INTENSITY_THRESHOLD = 200
FLUORO_FRACTION_CUTOFF = 0.1


# ---------------- Setup ----------------
def ensure_output_folder(path):
    os.makedirs(path, exist_ok=True)


# ---------------- IO ----------------
def load_images(image_folder):
    return {
        fname.replace('.npy', ''): np.load(os.path.join(image_folder, fname))
        for fname in os.listdir(image_folder)
        if fname.endswith('.npy')
    }


def load_masks(mask_path, image_keys):
    """
    Load one cell mask per image.

    Expected:
    - all_masks[i] is a 2D cell mask for image i
    OR
    - all_masks[i, 0] is cell mask if masks were saved with an extra axis
    """
    all_masks = np.load(mask_path)

    masks = {}
    for i, image_name in enumerate(image_keys):
        current_mask = all_masks[i]

        # If mask has an extra leading axis, take the first layer as cells
        if current_mask.ndim == 3:
            current_mask = current_mask[0]

        masks[image_name] = current_mask

    return masks


def save_mask(image_name, mask):
    out_path = os.path.join(output_folder, f'{image_name}_mask.npy')
    np.save(out_path, mask)
    logger.info(f'Mask saved: {out_path}')


# ---------------- Mask Filtering ----------------
def remove_saturated_cells(image_stack, cells_mask, COI=COI):
    """Remove masks for saturated cells based on intensity threshold."""
    raw = image_stack[COI, :, :]

    valid_labels = []
    for label in np.unique(cells_mask)[1:]:
        pixel_mask = (cells_mask == label)
        pixel_count = np.count_nonzero(pixel_mask)

        if pixel_count == 0:
            continue

        saturated = np.count_nonzero(raw[pixel_mask] > SATURATION_THRESHOLD)
        if saturated / pixel_count < SATURATION_FRAC_CUTOFF:
            valid_labels.append(label)

    filtered_cells = np.where(np.isin(cells_mask, valid_labels), cells_mask, 0)
    return filtered_cells


def filter_cells_by_fluoro_expression(image_stack, cells_mask):
    """Keep only cells with significant fluorescence signal."""
    fluoro = image_stack[COI, :, :]
    valid_labels = []

    for label in np.unique(cells_mask)[1:]:
        mask = (cells_mask == label)
        pixel_count = np.count_nonzero(mask)

        if pixel_count == 0:
            continue

        fluoro_pixels = fluoro[mask]
        bright_pixels = np.count_nonzero(fluoro_pixels > FLUORO_INTENSITY_THRESHOLD)

        if bright_pixels / pixel_count > FLUORO_FRACTION_CUTOFF:
            valid_labels.append(label)

    filtered_cells = np.where(np.isin(cells_mask, valid_labels), cells_mask, 0)
    return filtered_cells


def remove_border_objects(mask):
    return clear_border(mask, buffer_size=BORDER_BUFFER_SIZE)


def filter_masks_auto(image_stack, cells_mask, filter_fluoro=False):
    """
    Automatic filtering for cell masks only.
    """
    cells_filtered = remove_saturated_cells(image_stack, cells_mask)
    cells_filtered = remove_border_objects(cells_filtered)

    if filter_fluoro:
        cells_filtered = filter_cells_by_fluoro_expression(image_stack, cells_filtered)

    return cells_filtered


# ---------------- Manual QC ----------------
def validate_with_napari(image_stack, image_name, cells_mask):
    """Launch napari, allow user to edit cell masks, then save upon exit."""
    app = QApplication.instance()
    if not app:
        app = QApplication([])

    viewer = napari.Viewer()
    viewer.add_image(image_stack, name='image_stack')
    viewer.add_labels(cells_mask, name='cells')

    # Show and block until window closed
    viewer.window._qt_window.show()
    app.exec_()

    # After closing the window, get edited data
    cells = viewer.layers['cells'].data

    save_mask(image_name, cells)
    return cells


# ---------------- Main QC Pipeline ----------------
def run_qc_pipeline(filter_fluoro=False):
    ensure_output_folder(output_folder)

    images = load_images(image_folder)
    masks = load_masks(os.path.join(mask_folder, mask_filename), images.keys())

    logger.info('starting automated mask filtering')
    filtered_masks = {
        name: filter_masks_auto(image, masks[name], filter_fluoro=filter_fluoro)
        for name, image in images.items()
    }

    logger.info('starting manual validation in napari')
    already_filtered = {
        fname.replace('_mask.npy', '')
        for fname in os.listdir(output_folder)
        if fname.endswith('_mask.npy')
    }

    for name, image in images.items():
        if name not in already_filtered:
            _ = validate_with_napari(image, name, filtered_masks[name])


# ---------------- Entry Point ----------------
if __name__ == '__main__':
    run_qc_pipeline(filter_fluoro=True)