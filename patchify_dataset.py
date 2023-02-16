import logging
import os

import numpy as np
from PIL import Image
from patchify import patchify

logging_level = logging.INFO
# logging_level = logging.DEBUG
logging.basicConfig(level=logging_level, format='[%(levelname)s] %(message)s')

root_dir = "./data/vaihingen"
img_dir = f"{root_dir}/imgs"

patch_size = 128
channels = 3

output_directory = f"{img_dir}/patches_{patch_size}x{patch_size}x{channels}"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
logging.info(f"output_directory: {output_directory}")

input_filenames = os.listdir(img_dir)
logging.debug(f"input_files: {input_filenames}")

allowed_extensions = ('.tif', '.tiff')
input_filenames = [file for file in input_filenames if file.endswith(allowed_extensions)]
logging.info(f"input_files: {' '.join(input_filenames)}")
logging.info(f"len(input_files): {len(input_filenames)}")

# input_file = f"{img_dir}/top_mosaic_09cm_area1.tif"  # ./data/vaihingen/imgs/top_mosaic_09cm_area1.tif
for input_filename in input_filenames:
    input_path = f"{img_dir}/{input_filename}"
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

# TODO iterate over targets, too
