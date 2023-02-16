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

    input_filenames = os.listdir(directory)
    logging.debug(f"input_files: {input_filenames}")

    input_filenames = [file for file in input_filenames if file.endswith(allowed_extensions)]
    logging.info(f"input_files: {' '.join(input_filenames)}")
    logging.info(f"len(input_files): {len(input_filenames)}")

    for input_filename in input_filenames:
        input_path = f"{directory}/{input_filename}"
        logging.info(f"input_path: {input_path}")

        input_image = Image.open(input_path)
        input_image = np.asarray(input_image)
        logging.debug(f"input_image.shape: {input_image.shape}")

        patches = patchify(input_image, (patch_size, patch_size, channels), step=patch_size)
        logging.debug(f"patches.shape: {patches.shape}")

        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                patch = patches[i, j, 0]
                patch = Image.fromarray(patch)

                basename = os.path.splitext(input_filename)[0]
                output_file = f"{output_directory}/{basename}_patch_{i}_{j}.tif"
                patch.save(output_file)
                logging.info(f"Patch {output_file} saved.")

    output_filenames = os.listdir(output_directory)
    logging.info(f"output_files: {' '.join(output_filenames)}")
    logging.info(f"len(output_files): {len(output_filenames)}")


if __name__ == "__main__":
    logging_level = logging.INFO
    # logging_level = logging.DEBUG
    logging.basicConfig(level=logging_level, format='[%(levelname)s] %(message)s')

    root_directory = "./data/vaihingen"
    img_directory = f"{root_directory}/imgs"

    patchify_directory(img_directory)

    # TODO iterate over targets, too

    # TODO call this script before training
