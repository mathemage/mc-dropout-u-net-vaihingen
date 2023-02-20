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
