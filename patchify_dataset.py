import logging
import os

import numpy as np
from PIL import Image
from patchify import patchify

logging_level = logging.INFO
# logging_level = logging.DEBUG
logging.basicConfig(level=logging_level, format='[%(levelname)s] %(message)s')

# From https://levelup.gitconnected.com/how-to-split-an-image-into-patches-with-python-e1cf42cf4f77
root_dir = "./data/vaihingen"
img_dir = f"{root_dir}/imgs"
files = [f for f in os.listdir('.') if os.path.isfile(f)]
logging.debug(f"ls: {files}")

input_file = f"{img_dir}/top_mosaic_09cm_area1.tif"  # ./data/vaihingen/imgs/top_mosaic_09cm_area1.tif

input_image = Image.open(input_file)
input_image = np.asarray(input_image)
logging.info(f"image.shape: {input_image.shape}")

patch_size = 128
channels = 3
patches = patchify(input_image, (patch_size, patch_size, channels), step=patch_size)
logging.info(f"patches.shape: {patches.shape}")

output_directory = f"{img_dir}/patches_{patch_size}x{patch_size}x{channels}"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
logging.debug(f"output_directory: {output_directory}")

for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        output_file = f"{output_directory}/top_mosaic_09cm_area1_patch_{i}_{j}.tif"
        patch = patches[i, j, 0]
        patch = Image.fromarray(patch)
        num = i * patches.shape[1] + j
        patch.save(output_file)
        logging.info(f"Patch {output_file} saved.")
