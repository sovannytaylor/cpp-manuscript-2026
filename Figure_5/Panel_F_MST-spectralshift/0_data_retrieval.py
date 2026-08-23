"""
Download a ZIP file, extract its contents into a folder located beside this
script, flatten the extracted directory structure, and delete the ZIP file.
"""

import logging
import shutil
import zipfile
from pathlib import Path
from urllib import request
from contextlib import closing


# -----------------------------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------------------------

URL = "https..."
ZIP_FILENAME = "raw_data.zip"
EXTRACTED_FOLDER_NAME = "raw_data"


# Setup basic logger
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def download_and_extract_zip(filename, url, output_folder):
    """
    Download a ZIP file, extract all files into a single output folder,
    and delete the downloaded ZIP file.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    zip_path = output_folder / filename
    extract_folder = output_folder / EXTRACTED_FOLDER_NAME
    extract_folder.mkdir(parents=True, exist_ok=True)

    # Download the ZIP file
    try:
        with closing(request.urlopen(url)) as response:
            with zip_path.open("wb") as file:
                shutil.copyfileobj(response, file)

        logger.info("Downloaded: %s", zip_path)

    except Exception as error:
        logger.error("Download failed for %s: %s", filename, error)
        return

    # Extract and flatten all files
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            for member in zip_file.infolist():
                # Skip directory entries
                if member.is_dir():
                    continue

                # Use only the filename, removing all ZIP subfolders
                destination = extract_folder / Path(member.filename).name

                # Prevent files with identical names from being overwritten
                if destination.exists():
                    stem = destination.stem
                    suffix = destination.suffix
                    number = 1

                    while destination.exists():
                        destination = (
                            extract_folder / f"{stem}_{number}{suffix}"
                        )
                        number += 1

                with zip_file.open(member) as source:
                    with destination.open("wb") as target:
                        shutil.copyfileobj(source, target)

        logger.info("Extracted files to: %s", extract_folder)

    except Exception as error:
        logger.error("Extraction failed for %s: %s", filename, error)
        return

    # Delete the downloaded ZIP file
    try:
        zip_path.unlink()
        logger.info("Deleted ZIP file: %s", zip_path)

    except Exception as error:
        logger.warning("Could not delete ZIP file %s: %s", zip_path, error)


if __name__ == "__main__":
    # Folder containing this Python script
    script_folder = Path(__file__).resolve().parent

    download_and_extract_zip(
        filename=ZIP_FILENAME,
        url=URL,
        output_folder=script_folder,
    )