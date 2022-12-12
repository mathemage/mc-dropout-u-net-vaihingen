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

UNDEF_CLS = -1
assert UNDEF_CLS < 0


def linesep():
    print("_" * 80)


def rgb_to_onehot(rgb_target=vaihingen_lut, color_lut=vaihingen_lut):
    print(f"rgb_target: {rgb_target}")          # TODO extract into `print_log(tensor, tensor_name)`
    print(f"rgb_target.shape: {rgb_target.shape}")
    print(f"color_lut: {color_lut}")
    print(f"color_lut.shape: {color_lut.shape}")

    lut_indices = torch.zeros(rgb_target.shape[:-1])
    lut_indices.fill_(UNDEF_CLS)
    print(f"lut_indices: {lut_indices}")
    print(f"lut_indices.shape: {lut_indices.shape}")

    for i, e in enumerate(color_lut):
        offset = -UNDEF_CLS + i
        lut_indices += torch.all(rgb_target == e, dim=-1) * offset  # from https://stackoverflow.com/a/57216689/1460660
        print(f"\ni == {i}, offset == {offset}, e: {e}")
        print(f"lut_indices: {lut_indices}")


if __name__ == "__main__":
    print(torch.__version__)

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
    rgb_to_onehot(rgb_target=sample_target1, color_lut=vaihingen_lut)
    linesep()

    sample_target2 = torch.stack([
        vaihingen_lut,
        flipped_vaihingen_lut
    ])
    print(f"sample_target2: {sample_target2}")
    print(f"sample_target2.size: {sample_target2.size()}")
    rgb_to_onehot(rgb_target=sample_target2, color_lut=vaihingen_lut)
    linesep()
