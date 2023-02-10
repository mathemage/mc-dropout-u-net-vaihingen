import numpy as np
from patchify import patchify, unpatchify
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# From https://github.com/dovahcrow/patchify.py#2d-image-patchify-and-merge
image = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
logging.info(f"image: {image}")
patches = patchify(image, (2, 2), step=1)  # split image into 2*3 small 2*2 patches.
logging.info(f"patches: {patches}")

assert patches.shape == (2, 3, 2, 2)

reconstructed_image = unpatchify(patches, image.shape)
assert (reconstructed_image == image).all()

# From https://levelup.gitconnected.com/how-to-split-an-image-into-patches-with-python-e1cf42cf4f77
from PIL import Image

# root_dir = "./data/vaihingen"
root_dir = "../data/vaihingen"
img_dir = f"{root_dir}/imgs"
import os   # TODO remove
files = [f for f in os.listdir('.') if os.path.isfile(f)]
logging.info(f"ls: {files}")

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
