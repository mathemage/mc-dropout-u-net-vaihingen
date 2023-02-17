import logging
import os

import numpy as np
from PIL import Image
from patchify import patchify


def patchify_directory(directory, patch_size=128, channels=3, allowed_extensions=('.tif', '.tiff')):
    output_directory = f"{directory}/patches_{patch_size}x{patch_size}x{channels}"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    logging.info(f"output_directory: {output_directory}")

    filenames = os.listdir(directory)
    logging.debug(f"filenames: {' '.join(filenames)}")
    logging.debug(f"len(filenames): {len(filenames)}")

    filenames = [file for file in filenames if file.endswith(allowed_extensions)]
    logging.info(f"filenames: {' '.join(filenames)}")
    logging.info(f"len(filenames): {len(filenames)}")

    for filename in filenames:
        path = f"{directory}/{filename}"
        logging.info(f"path: {path}")

        image = Image.open(path)
        image = np.asarray(image)
        logging.debug(f"image.shape: {image.shape}")

        patches = patchify(image, (patch_size, patch_size, channels), step=patch_size)
        logging.debug(f"patches.shape: {patches.shape}")

        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                patch = patches[i, j, 0]
                patch = Image.fromarray(patch)

                output_file = f"{output_directory}/patch_{i}_{j}_of_{filename}"
                patch.save(output_file)
                logging.info(f"Patch {output_file} saved.")

    output_filenames = os.listdir(output_directory)
    logging.info(f"output_files: {' '.join(output_filenames)}")
    logging.critical(f"len(output_files): {len(output_filenames)}")


def patchify_dataset(root_directory="./data/vaihingen", img_directory=None, mask_directory=None):
    if img_directory is None:
        img_directory = f"{root_directory}/imgs"
    patchify_directory(img_directory)

    if mask_directory is None:
        mask_directory = f"{root_directory}/masks"
    patchify_directory(mask_directory)


if __name__ == "__main__":
    logging_level = logging.INFO
    # logging_level = logging.DEBUG
    # logging_level = logging.CRITICAL
    logging.basicConfig(level=logging_level, format='[%(levelname)s] %(message)s')

    patchify_dataset()

    # TODO call this script before training
