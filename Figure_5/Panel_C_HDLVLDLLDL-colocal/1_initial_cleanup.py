"""
Import data as numpy array
"""

# used bioiamge-fazallab environment

import os
import numpy as np
from loguru import logger
from bioio import BioImage
from bioio.writers import OmeTiffWriter
import bioio_ome_tiff
import bioio_nd2

logger.info('import ok')

# -------------------
# configuration
# -------------------
input_path = r"P:\Sophie\uptake_project\Figure4_ldl-colocalization-interaction\Lipoproteins\26043\26043_python"
output_folder = r"C:\Users\u244278\OneDrive - Baylor College of Medicine\Documents\python_projects\ANA_26043_LDLR_HDLLDLVLDL\results\initial_cleanup"
image_extensions = ['.czi', '.tif', '.tiff', '.lif', '.nd2']


def image_converter(image_path, output_folder, tiff=False, MIP=False, array=True):
    """Stack images from nested files and save for subsequent processing

    Args:
        image_path (str): filepath for the image to be converted
        output_folder (str): folder for saving the converted images
        tiff (bool, optional): Save OME-TIFF. Defaults to False.
        MIP (bool, optional): Save np array as maximum projected image along third to last axis. Defaults to False.
        array (bool, optional): Save np array. Defaults to True.
    """
    os.makedirs(output_folder, exist_ok=True)

    # check if image exists
    if not os.path.exists(image_path):
        logger.warning(f'File not found for {image_path}')
        return

    # get a bioimage object
    bio_image = BioImage(image_path)
    image_shape = bio_image.dims

    # NOTE: using separate if/elif so only one path runs
    # import single channel timeseries
    if (image_shape['T'][0] > 1) and (image_shape['C'][0] == 1):
        image = bio_image.get_image_data("TYX", C=0, Z=0)

    # import multichannel timeseries
    elif (image_shape['T'][0] > 1) and (image_shape['C'][0] > 1):
        image = bio_image.get_image_data("CTYX", B=0, Z=0, V=0)

    # import multichannel z-stack
    elif image_shape['Z'][0] > 1:
        image = bio_image.get_image_data("CZYX", B=0, V=0, T=0)

    # import multichannel single z-slice single timepoint
    elif (image_shape['Z'][0] == 1) and (image_shape['T'][0] == 1) and (image_shape['C'][0] > 1):
        image = bio_image.get_image_data("CYX", B=0, Z=0, V=0, T=0)

    else:
        # fallback: try a common minimal read
        # (adjust if you have other cases like single-channel single-plane)
        image = bio_image.get_image_data("YX", C=0, Z=0, T=0)

    # make more human readable name
    short_name = os.path.splitext(os.path.basename(image_path))[0]
    short_name = short_name.replace(" ", "-")

    if tiff:
        OmeTiffWriter.save(image, os.path.join(output_folder, f"{short_name}.tif"))

    if array:
        np.save(os.path.join(output_folder, f"{short_name}.npy"), image)

    if MIP:
        # save image as maximum intensity projection (MIP) numpy array
        mip_image = np.max(image, axis=-3)  # assuming axis for projection is third from last
        np.save(os.path.join(output_folder, f"{short_name}_mip.npy"), mip_image)


if __name__ == '__main__':

    # --------------- initialize file_list ---------------
    # Build file list correctly (JOIN PATHS + proper extension check)
    flat_file_list = [
        os.path.join(input_path, filename)
        for filename in os.listdir(input_path)
        if filename.lower().endswith(tuple(ext.lower() for ext in image_extensions))
    ]

    # remove images that do not require analysis (e.g., qualitative controls)
    do_not_quantitate = ['2X']
    image_names = [fp for fp in flat_file_list if not any(word in fp for word in do_not_quantitate)]

    # remove duplicates (preserve order)
    image_names = list(dict.fromkeys(image_names))

    # --------------- collect image names and convert ---------------
    for fp in image_names:
        
        short_name = os.path.splitext(os.path.basename(fp))[0]
        short_name = short_name.replace(" ", "-")

        output_file = os.path.join(output_folder, f"{short_name}.npy")

        #skip if already processed
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            logger.info(f"Skipping already processed: {short_name}")
            continue

        image_converter(fp, output_folder=output_folder, tiff=False, MIP=False, array=True)
        logger.info('initial_cleanup_complete')