"""
Import stitched microscopy images and save them as NumPy arrays.

This version:
- Extracts peptide_rep# from each filename
- Example:
      ...-Scene-01-GR30_rep1-B02.czi
  becomes:
      GR30_rep1.npy
- Reconstructs true mosaic CZI files using read_mosaic()
- Does NOT pass S to read_mosaic()
- Preserves the full stitched image
- Keeps channels in (C, Y, X) order
- Pads the shorter Y/X dimension with zeros
- Does not crop or resize
- Processes only 1 image while testing
"""

import os
import re
from pathlib import Path

import numpy as np
from loguru import logger
from bioio import BioImage
from bioio.writers import OmeTiffWriter

import bioio_ome_tiff
import bioio_nd2

try:
    from aicspylibczi import CziFile
except ImportError:
    CziFile = None


logger.info("imports complete")


# ============================================================
# configuration
# ============================================================

input_path = (
    r"P:\Sophie\uptake_project\Figure1_cells"
    r"\U2OS-63xwater-EEA1\python_raw_files"
)

output_folder = r"results\initial_cleanup"

image_extensions = [
    ".czi",
    ".tif",
    ".tiff",
    ".lif",
    ".nd2",
]

# Skip output names containing any of these strings.
do_not_quantitate = []

SAVE_NPY = True
SAVE_TIFF = False

# Maximum-project across Z if a Z-stack is present.
MIP = False

# Pad the shorter spatial dimension so Y == X.
PAD_TO_SQUARE = True

# True adds padding evenly to both sides.
# False adds padding only to the bottom or right.
CENTER_PADDING = True

# False prevents existing files from being overwritten.
OVERWRITE_EXISTING = False

# Process only one image during testing.
TEST_N_FILES = None


# ============================================================
# filename parsing
# ============================================================

def get_output_name_from_filename(image_path):
    """
    Extract peptide_rep# from the filename.

    Example
    -------
    Input:
        ...-Scene-01-GR30_rep1-B02.czi

    Output:
        GR30_rep1
    """

    basename = os.path.basename(image_path)
    stem = os.path.splitext(basename)[0]

    # Preferred format:
    # -Scene-01-GR30_rep1-B02
    match = re.search(
        r"-Scene-\d+-(.+?_rep\d+)-[A-H]\d{1,2}$",
        stem,
        flags=re.IGNORECASE,
    )

    if match is not None:
        return match.group(1)

    # Fallback: locate peptide_rep# anywhere in the filename.
    match = re.search(
        r"([A-Za-z0-9_.+-]+_rep\d+)",
        stem,
        flags=re.IGNORECASE,
    )

    if match is not None:
        return match.group(1)

    logger.warning(
        f"Could not extract peptide_rep# from: {basename}"
    )

    return None


# ============================================================
# padding
# ============================================================

def squarify(image, center=True):
    """
    Pad the final Y and X dimensions so the image becomes square.

    This function only adds zero-valued pixels. It does not crop,
    resize, rescale, or interpolate the original image.
    """

    if image.ndim < 2:
        raise ValueError(
            f"Expected at least two dimensions, received {image.shape}"
        )

    original_shape = image.shape

    y_size = image.shape[-2]
    x_size = image.shape[-1]

    square_size = max(y_size, x_size)

    total_y_padding = square_size - y_size
    total_x_padding = square_size - x_size

    if center:
        pad_top = total_y_padding // 2
        pad_bottom = total_y_padding - pad_top

        pad_left = total_x_padding // 2
        pad_right = total_x_padding - pad_left
    else:
        pad_top = 0
        pad_bottom = total_y_padding

        pad_left = 0
        pad_right = total_x_padding

    pad_width = [(0, 0)] * image.ndim

    pad_width[-2] = (
        pad_top,
        pad_bottom,
    )

    pad_width[-1] = (
        pad_left,
        pad_right,
    )

    padded = np.pad(
        image,
        pad_width=pad_width,
        mode="constant",
        constant_values=0,
    )

    logger.info(
        f"Padding: {original_shape} -> {padded.shape}"
    )

    logger.info(
        f"Padding added: "
        f"top={pad_top}, bottom={pad_bottom}, "
        f"left={pad_left}, right={pad_right}"
    )

    return padded


# ============================================================
# CZI dimension helpers
# ============================================================

def get_dimension_start_and_size(
    dimension_dictionary,
    dimension_name,
):
    """
    Return the start coordinate and size for a CZI dimension.

    Example:
        "S": (39, 1)

    means the valid scene coordinate begins at 39.
    """

    if dimension_name not in dimension_dictionary:
        return 0, 1

    start, size = dimension_dictionary[dimension_name]

    return int(start), int(size)


def clean_non_mosaic_czi_array(
    image,
    shape_description,
):
    """
    Clean and reorder an array returned by CziFile.read_image().

    The final array uses:
        CYX
        CZYX
        CTYX
        CTZYX
    """

    dimension_names = [
        dimension
        for dimension, size in shape_description
    ]

    logger.info(
        f"Raw CZI dimension description: {shape_description}"
    )

    logger.info(
        f"Raw CZI array shape: {image.shape}"
    )

    if image.ndim != len(dimension_names):
        raise RuntimeError(
            "The returned CZI array does not match its dimension "
            f"description. Shape={image.shape}, "
            f"description={shape_description}"
        )

    # Remove unnecessary singleton dimensions while retaining C, T, Z,
    # Y and X.
    allowed_dimensions = {
        "C",
        "T",
        "Z",
        "Y",
        "X",
    }

    for axis in reversed(range(len(dimension_names))):
        dimension_name = dimension_names[axis]
        dimension_size = image.shape[axis]

        if dimension_name not in allowed_dimensions:
            if dimension_size != 1:
                raise ValueError(
                    f"Unsupported non-singleton CZI dimension "
                    f"{dimension_name} with size {dimension_size}"
                )

            image = np.take(
                image,
                indices=0,
                axis=axis,
            )

            dimension_names.pop(axis)

    if "Y" not in dimension_names:
        raise RuntimeError(
            f"No Y dimension was found: {dimension_names}"
        )

    if "X" not in dimension_names:
        raise RuntimeError(
            f"No X dimension was found: {dimension_names}"
        )

    desired_dimensions = []

    if "C" in dimension_names:
        desired_dimensions.append("C")

    if "T" in dimension_names:
        desired_dimensions.append("T")

    if "Z" in dimension_names:
        desired_dimensions.append("Z")

    desired_dimensions.extend(
        ["Y", "X"]
    )

    transpose_axes = [
        dimension_names.index(dimension_name)
        for dimension_name in desired_dimensions
    ]

    image = np.transpose(
        image,
        axes=transpose_axes,
    )

    dimension_order = "".join(
        desired_dimensions
    )

    # Ensure that single-channel arrays still have a C dimension.
    if "C" not in dimension_order:
        image = image[np.newaxis, ...]
        dimension_order = "C" + dimension_order

    logger.info(
        f"Cleaned non-mosaic CZI: "
        f"order={dimension_order}, shape={image.shape}"
    )

    return image, dimension_order


# ============================================================
# mosaic CZI loading
# ============================================================

def load_mosaic_czi(
    czi,
    dimension_dictionary,
):
    """
    Reconstruct a true mosaic CZI file.

    Important:
        S is intentionally NOT passed to read_mosaic().
    """

    logger.info(
        "Mosaic CZI detected. Reconstructing the full stitched image."
    )

    c_start, c_size = get_dimension_start_and_size(
        dimension_dictionary,
        "C",
    )

    t_start, t_size = get_dimension_start_and_size(
        dimension_dictionary,
        "T",
    )

    z_start, z_size = get_dimension_start_and_size(
        dimension_dictionary,
        "Z",
    )

    logger.info(
        f"Mosaic dimension ranges: "
        f"C=({c_start}, {c_size}), "
        f"T=({t_start}, {t_size}), "
        f"Z=({z_start}, {z_size})"
    )

    if t_size > 1:
        raise ValueError(
            f"The CZI contains {t_size} timepoints. "
            "This script currently expects one timepoint."
        )

    if z_size > 1 and not MIP:
        raise ValueError(
            f"The CZI contains {z_size} Z slices. "
            "Set MIP=True to maximum-project the Z-stack."
        )

    reconstructed_channels = []

    for channel_offset in range(c_size):
        channel_index = c_start + channel_offset

        logger.info(
            f"Reconstructing channel "
            f"{channel_offset + 1}/{c_size} "
            f"using C={channel_index}"
        )

        mosaic_arguments = {
            "C": channel_index,
            "scale_factor": 1.0,
        }

        # Pass T only when T exists and has one valid position.
        if "T" in dimension_dictionary:
            mosaic_arguments["T"] = t_start

        # For the current single-Z images, specify the valid Z value.
        if "Z" in dimension_dictionary and z_size == 1:
            mosaic_arguments["Z"] = z_start

        # DO NOT add S here.
        logger.info(
            f"read_mosaic arguments: {mosaic_arguments}"
        )

        channel_image = czi.read_mosaic(
            **mosaic_arguments
        )

        channel_image = np.asarray(
            channel_image
        )

        logger.info(
            f"Raw reconstructed channel shape: "
            f"{channel_image.shape}"
        )

        # Remove singleton dimensions such as T, Z or C.
        channel_image = np.squeeze(
            channel_image
        )

        if channel_image.ndim != 2:
            raise ValueError(
                f"Expected a reconstructed 2D image for "
                f"C={channel_index}, but received "
                f"shape {channel_image.shape}"
            )

        reconstructed_channels.append(
            channel_image
        )

    if not reconstructed_channels:
        raise RuntimeError(
            "No mosaic channels were reconstructed."
        )

    # Verify that all channels have identical Y/X dimensions.
    reference_shape = reconstructed_channels[0].shape

    for channel_number, channel_image in enumerate(
        reconstructed_channels,
        start=1,
    ):
        if channel_image.shape != reference_shape:
            raise RuntimeError(
                f"Mosaic channel shapes differ. "
                f"Channel 1={reference_shape}, "
                f"channel {channel_number}={channel_image.shape}"
            )

    image = np.stack(
        reconstructed_channels,
        axis=0,
    )

    dimension_order = "CYX"

    logger.info(
        f"Full reconstructed mosaic: "
        f"order={dimension_order}, "
        f"shape={image.shape}, "
        f"dtype={image.dtype}"
    )

    return image, dimension_order


# ============================================================
# non-mosaic CZI loading
# ============================================================

def load_non_mosaic_czi(
    czi,
    dimension_dictionary,
):
    """
    Read a standard, non-mosaic CZI file.

    Actual dimension start values are used instead of assuming zero.
    """

    logger.info(
        "Non-mosaic CZI detected. Reading the image directly."
    )

    read_arguments = {}

    for dimension_name in [
        "S",
        "B",
        "T",
        "Z",
        "V",
    ]:
        if dimension_name not in dimension_dictionary:
            continue

        start, size = get_dimension_start_and_size(
            dimension_dictionary,
            dimension_name,
        )

        # Select singleton dimensions using their actual coordinate.
        if size == 1:
            read_arguments[dimension_name] = start

    logger.info(
        f"read_image arguments: {read_arguments}"
    )

    image, shape_description = czi.read_image(
        **read_arguments
    )

    image = np.asarray(
        image
    )

    image, dimension_order = clean_non_mosaic_czi_array(
        image,
        shape_description,
    )

    return image, dimension_order


# ============================================================
# CZI dispatcher
# ============================================================

def load_czi_direct(image_path):
    """
    Load a CZI file with aicspylibczi.

    Mosaic files are reconstructed with read_mosaic().
    Non-mosaic files are loaded with read_image().
    """

    if CziFile is None:
        raise ImportError(
            "aicspylibczi is required for CZI files. "
            "Install it with:\n"
            "pip install aicspylibczi"
        )

    czi = CziFile(
        Path(image_path)
    )

    dimension_ranges = czi.get_dims_shape()

    if not dimension_ranges:
        raise RuntimeError(
            f"No CZI dimensions were found for:\n{image_path}"
        )

    dimension_dictionary = dimension_ranges[0]

    logger.info(
        f"CZI dimension order: {czi.dims}"
    )

    logger.info(
        f"CZI dimension ranges: {dimension_ranges}"
    )

    is_mosaic = czi.is_mosaic()

    logger.info(
        f"CZI mosaic status: {is_mosaic}"
    )

    if is_mosaic or "M" in dimension_dictionary:
        return load_mosaic_czi(
            czi=czi,
            dimension_dictionary=dimension_dictionary,
        )

    return load_non_mosaic_czi(
        czi=czi,
        dimension_dictionary=dimension_dictionary,
    )


# ============================================================
# BioIO loading for non-CZI files
# ============================================================

def load_with_bioio(image_path):
    """
    Load TIFF, ND2, LIF and other BioIO-supported formats.
    """

    bio_image = BioImage(
        image_path
    )

    logger.info(
        f"Available BioIO scenes: {bio_image.scenes}"
    )

    if len(bio_image.scenes) > 0:
        first_scene = bio_image.scenes[0]

        logger.info(
            f"Selecting BioIO scene: {first_scene}"
        )

        bio_image.set_scene(
            first_scene
        )

    logger.info(
        f"BioIO dimensions: {bio_image.dims}"
    )

    logger.info(
        f"BioIO dimension order: {bio_image.dims.order}"
    )

    logger.info(
        f"BioIO dimension shape: {bio_image.dims.shape}"
    )

    t_size = bio_image.dims["T"][0]
    z_size = bio_image.dims["Z"][0]

    if t_size > 1 and z_size > 1:
        image = bio_image.get_image_data(
            "CTZYX"
        )
        dimension_order = "CTZYX"

    elif t_size > 1:
        image = bio_image.get_image_data(
            "CTYX",
            Z=0,
        )
        dimension_order = "CTYX"

    elif z_size > 1:
        image = bio_image.get_image_data(
            "CZYX",
            T=0,
        )
        dimension_order = "CZYX"

    else:
        image = bio_image.get_image_data(
            "CYX",
            T=0,
            Z=0,
        )
        dimension_order = "CYX"

    image = np.asarray(
        image
    )

    logger.info(
        f"BioIO loaded: "
        f"order={dimension_order}, "
        f"shape={image.shape}, "
        f"dtype={image.dtype}"
    )

    return image, dimension_order


def load_microscopy_image(image_path):
    """
    Select the appropriate reader based on file extension.
    """

    extension = os.path.splitext(
        image_path
    )[1].lower()

    if extension == ".czi":
        return load_czi_direct(
            image_path
        )

    return load_with_bioio(
        image_path
    )


# ============================================================
# Z projection
# ============================================================

def max_project_z(
    image,
    dimension_order,
):
    """
    Maximum-project only the Z dimension.
    """

    if "Z" not in dimension_order:
        logger.info(
            "MIP requested, but no Z dimension is present."
        )

        return image, dimension_order

    z_axis = dimension_order.index(
        "Z"
    )

    projected = np.max(
        image,
        axis=z_axis,
    )

    projected_order = dimension_order.replace(
        "Z",
        "",
    )

    logger.info(
        f"Z projection: "
        f"{dimension_order} {image.shape} -> "
        f"{projected_order} {projected.shape}"
    )

    return projected, projected_order


# ============================================================
# image conversion
# ============================================================

def image_converter(
    image_path,
    output_folder,
):
    """
    Convert one microscopy image and save it using peptide_rep#.
    """

    output_name = get_output_name_from_filename(
        image_path
    )

    if output_name is None:
        return False

    if any(
        excluded_term.lower() in output_name.lower()
        for excluded_term in do_not_quantitate
    ):
        logger.info(
            f"Skipping excluded sample: {output_name}"
        )

        return False

    logger.info("=" * 80)

    logger.info(
        f"Input file: {image_path}"
    )

    logger.info(
        f"Output name: {output_name}"
    )

    image, dimension_order = load_microscopy_image(
        image_path
    )

    original_shape = image.shape

    logger.info(
        f"Loaded complete image: "
        f"order={dimension_order}, "
        f"shape={image.shape}, "
        f"dtype={image.dtype}"
    )

    logger.info(
        f"Loaded spatial dimensions: "
        f"Y={image.shape[-2]}, X={image.shape[-1]}"
    )

    if image.shape[-2] <= 0:
        raise RuntimeError(
            f"Invalid Y dimension: {image.shape}"
        )

    if image.shape[-1] <= 0:
        raise RuntimeError(
            f"Invalid X dimension: {image.shape}"
        )

    if MIP:
        image, dimension_order = max_project_z(
            image,
            dimension_order,
        )

    shape_before_padding = image.shape

    if PAD_TO_SQUARE:
        image = squarify(
            image,
            center=CENTER_PADDING,
        )

    final_shape = image.shape

    if final_shape[-2] < shape_before_padding[-2]:
        raise RuntimeError(
            f"Y was unexpectedly cropped: "
            f"{shape_before_padding} -> {final_shape}"
        )

    if final_shape[-1] < shape_before_padding[-1]:
        raise RuntimeError(
            f"X was unexpectedly cropped: "
            f"{shape_before_padding} -> {final_shape}"
        )

    save_name = output_name

    if MIP:
        save_name = f"{save_name}_mip"

    npy_path = os.path.join(
        output_folder,
        f"{save_name}.npy",
    )

    tif_path = os.path.join(
        output_folder,
        f"{save_name}.tif",
    )

    existing_outputs = []

    if SAVE_NPY and os.path.exists(npy_path):
        existing_outputs.append(
            npy_path
        )

    if SAVE_TIFF and os.path.exists(tif_path):
        existing_outputs.append(
            tif_path
        )

    if existing_outputs and not OVERWRITE_EXISTING:
        logger.warning(
            "Output already exists; skipping to prevent overwrite:\n"
            + "\n".join(existing_outputs)
        )

        return False

    if SAVE_NPY:
        np.save(
            npy_path,
            image,
        )

        logger.info(
            f"Saved NPY: {npy_path}"
        )

        saved_check = np.load(
            npy_path,
            mmap_mode="r",
        )

        if saved_check.shape != image.shape:
            raise RuntimeError(
                f"Saved shape mismatch for {output_name}: "
                f"expected {image.shape}, "
                f"found {saved_check.shape}"
            )

        if saved_check.dtype != image.dtype:
            raise RuntimeError(
                f"Saved dtype mismatch for {output_name}: "
                f"expected {image.dtype}, "
                f"found {saved_check.dtype}"
            )

        logger.info(
            f"Verified saved NPY: "
            f"shape={saved_check.shape}, "
            f"dtype={saved_check.dtype}"
        )

        del saved_check

    if SAVE_TIFF:
        OmeTiffWriter.save(
            image,
            tif_path,
            dim_order=dimension_order,
        )

        logger.info(
            f"Saved TIFF: {tif_path}"
        )

    logger.info(
        f"Finished {output_name}: "
        f"loaded={original_shape}, "
        f"saved={final_shape}"
    )

    return True


# ============================================================
# main
# ============================================================

if __name__ == "__main__":

    os.makedirs(
        output_folder,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # find microscopy files
    # --------------------------------------------------------

    files_found = []

    for root, directories, files in os.walk(
        input_path
    ):
        for filename in files:
            if any(
                filename.lower().endswith(extension)
                for extension in image_extensions
            ):
                files_found.append(
                    os.path.join(
                        root,
                        filename,
                    )
                )

    files_found = sorted(
        dict.fromkeys(files_found)
    )

    logger.info(
        f"Found {len(files_found)} microscopy files"
    )

    if not files_found:
        raise FileNotFoundError(
            f"No microscopy files were found under:\n"
            f"{input_path}"
        )

    # --------------------------------------------------------
    # parse output names and check duplicates
    # --------------------------------------------------------

    parsed_files = []
    seen_output_names = {}
    duplicate_names = set()

    for filepath in files_found:
        output_name = get_output_name_from_filename(
            filepath
        )

        if output_name is None:
            continue

        if any(
            excluded_term.lower() in output_name.lower()
            for excluded_term in do_not_quantitate
        ):
            logger.info(
                f"Excluded: {output_name} | "
                f"{os.path.basename(filepath)}"
            )

            continue

        normalized_name = output_name.lower()

        if normalized_name in seen_output_names:
            duplicate_names.add(
                normalized_name
            )

            logger.error(
                f"Duplicate output name detected: {output_name}\n"
                f"First file:  {seen_output_names[normalized_name]}\n"
                f"Second file: {filepath}"
            )

            continue

        seen_output_names[normalized_name] = filepath

        parsed_files.append(
            (
                output_name,
                filepath,
            )
        )

    # Remove the first occurrence of any duplicated output name too.
    if duplicate_names:
        parsed_files = [
            (
                output_name,
                filepath,
            )
            for output_name, filepath in parsed_files
            if output_name.lower() not in duplicate_names
        ]

    parsed_files.sort(
        key=lambda item: item[0].lower()
    )

    logger.info(
        f"Found {len(parsed_files)} uniquely named images"
    )

    for output_name, filepath in parsed_files:
        logger.info(
            f"Queued: {output_name} | "
            f"{os.path.basename(filepath)}"
        )

    # --------------------------------------------------------
    # test mode
    # --------------------------------------------------------

    if TEST_N_FILES is not None:
        parsed_files = parsed_files[
            :TEST_N_FILES
        ]

        logger.warning(
            f"TEST MODE: processing only "
            f"{len(parsed_files)} image(s)"
        )

    # --------------------------------------------------------
    # convert images
    # --------------------------------------------------------

    n_saved = 0
    n_skipped = 0
    n_failed = 0

    for image_number, (
        output_name,
        filepath,
    ) in enumerate(
        parsed_files,
        start=1,
    ):
        logger.info(
            f"Processing image "
            f"{image_number}/{len(parsed_files)}: "
            f"{output_name}"
        )

        try:
            success = image_converter(
                image_path=filepath,
                output_folder=output_folder,
            )

            if success:
                n_saved += 1
            else:
                n_skipped += 1

        except Exception:
            n_failed += 1

            logger.exception(
                f"Failed while processing: "
                f"{output_name} | {filepath}"
            )

            # Stop immediately in test mode so the actual traceback
            # remains visible.
            if TEST_N_FILES is not None:
                raise

    logger.info("=" * 80)

    logger.info(
        f"Images successfully saved: {n_saved}"
    )

    logger.info(
        f"Images skipped: {n_skipped}"
    )

    logger.info(
        f"Images failed: {n_failed}"
    )

    logger.info(
        "Initial cleanup complete :-)"
    )