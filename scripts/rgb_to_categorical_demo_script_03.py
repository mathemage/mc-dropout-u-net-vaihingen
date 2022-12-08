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
vaihingen_lut = torch.tensor([
    [255, 255, 255],
    [0, 0, 255],
    [0, 255, 255],
    [0, 255, 0],
    [255, 255, 0],
    [255, 0, 0]
])
vaihingen_lut_dict = dict(enumerate(vaihingen_lut))


def linesep():
    print("_" * 80)


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
        print(f"i+1 == {i + 1}, e: {e}")
        print(f"lut_indices: {lut_indices}")


if __name__ == "__main__":
    print(torch.__version__)
    print(f"vaihingen_lut_dict: {vaihingen_lut_dict}")

    flipped_vaihingen_lut = torch.flip(vaihingen_lut, dims=[0])
    print(f"flipped_vaihingen_lut: {flipped_vaihingen_lut}")
    linesep()

    rgb_to_onehot()
    linesep()

    sample_target1 = torch.cat((
        vaihingen_lut,
        flipped_vaihingen_lut
    ), dim=0)
    print(f"sample_target1: {sample_target1}")
    print(f"sample_target1.size: {sample_target1.size()}")
    # rgb_to_onehot(rgb_target=sample_target1, color_lut=vaihingen_lut)  # TODO fix masking in the last dim
    linesep()

    sample_target2 = torch.stack([
        vaihingen_lut,
        flipped_vaihingen_lut
    ])
    print(f"sample_target2: {sample_target2}")
    print(f"sample_target2.size: {sample_target2.size()}")
    # rgb_to_onehot(rgb_target=sample_target2, color_lut=vaihingen_lut)  # TODO fix masking in the last dim
    linesep()
