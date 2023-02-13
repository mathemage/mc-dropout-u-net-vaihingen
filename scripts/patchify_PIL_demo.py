import numpy as np
from patchify import patchify, unpatchify
import logging
import os
from PIL import Image

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# From https://levelup.gitconnected.com/how-to-split-an-image-into-patches-with-python-e1cf42cf4f77
root_dir = "../data/vaihingen"
img_dir = f"{root_dir}/imgs"
files = [f for f in os.listdir('.') if os.path.isfile(f)]
logging.debug(f"ls: {files}")

input_file = f"{img_dir}/top_mosaic_09cm_area1.tif"  # ./data/vaihingen/imgs/top_mosaic_09cm_area1.tif

image = Image.open(input_file)  # for example (3456, 5184, 3)
image = np.asarray(image)
logging.info(f"image.shape: {image.shape}")  # (6, 10, 1, 512, 512, 3)

patches = patchify(image, (512, 512, 3), step=512)
logging.info(f"patches.shape: {patches.shape}")  # (6, 10, 1, 512, 512, 3)

for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        output_file = f"{img_dir}/patches/top_mosaic_09cm_area1_patch_{i}_{j}.tif"
        patch = patches[i, j, 0]
        patch = Image.fromarray(patch)
        num = i * patches.shape[1] + j
        patch.save(output_file)
        logging.info(f"Patch {output_file} saved.")  # (6, 10, 1, 512, 512, 3)
