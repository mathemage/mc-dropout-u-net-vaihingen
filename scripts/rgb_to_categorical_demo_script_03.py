# from https://stackoverflow.com/questions/64300830/pytorch-tensor-get-the-index-of-the-element-with-specific-values

import torch

# Six categories/classes have been defined:
#
# Impervious surfaces (RGB: 255, 255, 255)
# Building (RGB: 0, 0, 255)
# Low vegetation (RGB: 0, 255, 255)
# Tree (RGB: 0, 255, 0)
# Car (RGB: 255, 255, 0)
# Clutter/background (RGB: 255, 0, 0)
vaihingen_lut = [
    [255, 255, 255],
    [0, 0, 255],
    [0, 255, 255],
    [0, 255, 0],
    [255, 255, 0],
    [255, 0, 0]
]


def rgb_to_onehot(rgb_target=None, color_lut=vaihingen_lut):
    if rgb_target is None:
        rgb_target = torch.Tensor([1, 2, 2, 3, 4, 4, 4, 5])
    print(f"rgb_target: {rgb_target}")

    b = torch.Tensor([1, 2, 4])

    lut_indices = torch.zeros_like(rgb_target)
    print(f"lut_indices: {lut_indices}")
    for i, e in enumerate(b):
        lut_indices = lut_indices + (rgb_target == e) * (i + 1)
        print(f"i+1 == {i + 1}, lut_indices: {lut_indices}")


if __name__ == "__main__":
    print(torch.__version__)
    rgb_to_onehot()
