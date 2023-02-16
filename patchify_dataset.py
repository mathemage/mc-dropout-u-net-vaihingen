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

input_files = os.listdir(img_dir)
logging.debug(f"input_files: {input_files}")

allowed_extensions = ('.tif', '.tiff')
input_files = [file for file in input_files if file.endswith(allowed_extensions)]
logging.info(f"input_files: {' '.join(input_files)}")
logging.info(f"len(input_files): {len(input_files)}")
exit(1)

input_file = f"{img_dir}/top_mosaic_09cm_area1.tif"  # ./data/vaihingen/imgs/top_mosaic_09cm_area1.tif
input_image = Image.open(input_file)
input_image = np.asarray(input_image)
logging.debug(f"image.shape: {input_image.shape}")

patches = patchify(input_image, (patch_size, patch_size, channels), step=patch_size)
logging.debug(f"patches.shape: {patches.shape}")

for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        patch = patches[i, j, 0]
        patch = Image.fromarray(patch)

        output_file = f"{output_directory}/top_mosaic_09cm_area1_patch_{i}_{j}.tif"
        patch.save(output_file)
        # logging.info(f"Patch {output_file} saved.")
