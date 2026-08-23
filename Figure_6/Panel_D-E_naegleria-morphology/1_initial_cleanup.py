"""
Import data as numpy array
"""

import os
import numpy as np
from loguru import logger
from bioio import BioImage
from bioio.writers import OmeTiffWriter
from aicspylibczi import CziFile
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import bioio_ome_tiff
import bioio_nd2
import re

logger.info('import ok')

#used the bioimage environment 
# configuration
input_path = r"P:\Sophie\uptake_project\Figure6_conservation\KILL-CURVES\scenes\26037BC"
output_folder = r"26037BC_results/initial_cleanup/"
image_extensions = ['.czi', '.tif', '.tiff', '.lif', '.nd2']


# --------------------------- plate-specific name dictionaries ---------------------------
def make_short_name(image_path):
    basename = os.path.basename(image_path)
    basename = os.path.splitext(basename)[0]

    # Example:
    # 26037B_NAEG-PR39-4HR-01-Split Scenes (Write files)-01-Scene-01-B2-B02
    # becomes:
    # 26037B_NAEG-PR39-4HR-01-B02

    prefix = basename.split("-Split Scenes")[0]

    match = re.search(r"-([A-H]\d{2})$", basename)
    if not match:
        raise ValueError(f"Could not find final well ID in filename: {basename}")

    well_id = match.group(1)

    return f"{prefix}-{well_id}"

def squarify(image_stack):
    new_channels = []
    for idx, image in enumerate(image_stack):
        # make sure images are square
        rows, cols = image.shape
        max_dim = max(rows, cols)
        # calculate padding needed for each side
        pad_r = max_dim - rows
        pad_c = max_dim - cols
        # pad at the end (bottom/right)
        padded_image = np.pad(image, ((0, pad_r), (0, pad_c)), 
                                mode='constant', constant_values=0)
        new_channels.append(padded_image)
    image_stack = np.array(new_channels)
    return image_stack

# function for multi-scene data
def scene_finder(image_path, names_mapped):
    """Find scenes in multi-scene acquisitions."""

    czi = CziFile(image_path)
    data, dims = czi.read_image(return_dims=True)

    # squeeze unused dims
    data = np.squeeze(data)

    try:
        # mosaic-specific stitching
        bboxes = czi.get_all_mosaic_tile_bounding_boxes()

        tile_positions = []
        for tile_info, rect in bboxes.items():
            if tile_info.m_index < data.shape[1]:
                tile_positions.append((tile_info.m_index, rect))

        tile_positions.sort(key=lambda x: x[0])

        xs = []
        ys = []
        for m_index, rect in tile_positions:
            xs.append(rect.x)
            ys.append(rect.y)

        min_x, min_y = min(xs), min(ys)
        xs = [x - min_x for x in xs]
        ys = [y - min_y for y in ys]

        tile_h, tile_w = data.shape[2], data.shape[3]
        n_channels = data.shape[0]
        canvas_w = max(xs) + tile_w
        canvas_h = max(ys) + tile_h
        stitched = np.zeros((n_channels, canvas_h, canvas_w), dtype=data.dtype)

        for (m_index, rect), x, y in zip(tile_positions, xs, ys):
            stitched[:, y:y+tile_h, x:x+tile_w] = data[:, m_index, :, :]

    except RuntimeError:
        logger.info(f"File is not mosaic, using raw data directly: {image_path}")
        stitched = data

    # make sure non-mosaic single-channel data still has channel axis
    if stitched.ndim == 2:
        stitched = np.expand_dims(stitched, axis=0)

    # make scene x and y dimensions the same
    stitched = squarify(stitched)

    # find matching name
    # make output name directly from filename
    well_id = make_short_name(image_path)

    # if well_id is in do_not_quantitate list, skip saving
    if any(word in well_id for word in do_not_quantitate):
        logger.info(f"Skipping {well_id} due to do_not_quantitate criteria")
        return

    # save image as numpy array
    np.save(f'{output_folder}{well_id}.npy', stitched)


def image_converter(image_path, output_folder, tiff=False, MIP=False, array=True, split_scenes=False, find_scenes=False, name_dict=None):
    """Stack images from nested .czi files and save for subsequent processing

    Args:
        image_path (str): filepath for the image to be converted
        output_folder (str): filepath for saving the converted images
        tiff (bool, optional): Save tiff. Defaults to False.
        MIP (bool, optional): Save np array as maximum projected image along third to last axis. Defaults to False.
        array (bool, optional): Save np array. Defaults to True.
        split_scenes (bool, optional): Split scenes. Defaults to False.
        find_scenes (bool, optional): Find scenes. Defaults to False.
        names_mapped (dict, optional): Dictionary mapping scene names to desired output names. Required if split_scenes or find_scenes is True. Defaults to None.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # check if image exists
    full_path = None
    if os.path.exists(image_path):
        full_path = image_path

    if full_path is None:
        logger.warning(f'File not found for {image_path}')
        return
    
    if split_scenes == True:
        scene_splitter(image_path, name_dict)
        return
    
    if find_scenes == True:
        scene_finder(image_path, name_dict)
        return

    # get a bioimage object
    bio_image = BioImage(full_path)
    image_shape = bio_image.dims

    # import single channel timeseries
    if (image_shape['T'][0] > 1) & (image_shape['C'][0] == 1):
        image = bio_image.get_image_data("TYX", C=0, Z=0)

    # import multichannel timeseries
    if (image_shape['T'][0] > 1) & (image_shape['C'][0] > 1):
        image = bio_image.get_image_data("CTYX", B=0, Z=0, V=0)

    # import multichannel z-stack
    if image_shape['Z'][0] > 1:
        image = bio_image.get_image_data("CZYX", B=0, V=0, T=0)

    # import multichannel single z-slice single timepoint
    if (image_shape['Z'][0] == 1) & (image_shape['T'][0] == 1) & (image_shape['C'][0] > 1):
        image = bio_image.get_image_data("CYX", B=0, Z=0, V=0, T=0)

    # make more human readable name
    short_name = os.path.basename(image_path)
    short_name = short_name.split('.')[0]  # remove file extension

    if tiff == True:
        # save image as tiff file
        OmeTiffWriter.save(image, f'{output_folder}{short_name}.tif')

    if array == True:
        # save image as numpy array
        np.save(f'{output_folder}{short_name}.npy', image)

    if MIP == True:
        # save image as maximum intensity projection (MIP) numpy array 
        mip_image = np.max(image, axis=-3) # assuming axis for projection is third from last
        np.save(f'{output_folder}{short_name}_mip.npy', mip_image)





if __name__ == '__main__':

    # --------------- initalize file_list ---------------
    if input_path == r"P:\Sophie\uptake_project\Figure6_conservation\KILL-CURVES\scenes\26037BC":
        flat_file_list = [
        os.path.join(input_path, filename)
        for filename in os.listdir(input_path)
        if any(sub in filename for sub in image_extensions)]

    else:
        # find subdirectories of interest
        experiments = ['240509-Processed']
        # if you want all images from all subdirectories in file path, set experiments to 'walk_list'
        walk_list = [x[0] for x in os.walk(input_path)]
        walk_list = [item for item in walk_list if any(x in item for x in experiments)]

        # read in all image file names
        file_list = [
        [os.path.join(root, filename) for filename in files]
        for folder_path in walk_list
        for root, dirs, files in os.walk(folder_path)
    ]

        # flatten file_list
        flat_file_list = [item for sublist in file_list for item in sublist if any(sub in item for sub in image_extensions)]

    # remove images that do not require analysis (e.g., qualitative controls)
    do_not_quantitate = [] 
    image_names = [filename for filename in flat_file_list if not any(word in filename for word in do_not_quantitate)]

    # remove duplicates
    image_names = list(dict.fromkeys(image_names))
    image_names = [name for name in image_names if '(' in name] # keep only images with parentheses in name, which indicates they are from multi-scene acquisitions and need to be split

    # --------------- collect image names and convert ---------------
    # collect and convert images to np arrays
    for name in image_names:
        image_converter(
            name,
            output_folder=output_folder,
            find_scenes=True,
            name_dict=None
        )

    logger.info('initial cleanup complete :-)')
