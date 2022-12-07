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
vaihingen_lut_dict = dict(enumerate(vaihingen_lut))


def rgb_to_onehot(rgb_target=None, color_lut=None):
    if rgb_target is None:
        rgb_target = torch.Tensor([1, 2, 2, 3, 4, 4, 4, 5])
    print(f"rgb_target: {rgb_target}")

    if color_lut is None:
        color_lut = torch.Tensor([1, 2, 4])
    print(f"color_lut: {color_lut}")

    lut_indices = torch.zeros_like(rgb_target)
    print(f"lut_indices: {lut_indices}")
    for i, e in enumerate(color_lut):
        lut_indices = lut_indices + (rgb_target == e) * (i + 1)
        print(f"i+1 == {i + 1}, lut_indices: {lut_indices}")


if __name__ == "__main__":
    print(torch.__version__)
    print(f"vaihingen_lut_dict: {vaihingen_lut_dict}")

    sample_target = torch.tensor([
        vaihingen_lut,
        torch.flip(torch.tensor(vaihingen_lut), dims=[0])
    ])
    print(f"sample_target: {sample_target}")

    rgb_to_onehot()
