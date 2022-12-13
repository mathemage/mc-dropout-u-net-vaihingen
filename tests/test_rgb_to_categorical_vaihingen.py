# from https://stackoverflow.com/questions/64300830/pytorch-tensor-get-the-index-of-the-element-with-specific-values

import torch

from rgb_to_categorical_vaihingen import UNDEF_CLS, vaihingen_lut, rgb_to_onehot

flipped_vaihingen_lut = torch.flip(vaihingen_lut, dims=[0])


# print(f"flipped_vaihingen_lut: {flipped_vaihingen_lut}")


def test_undef_cls():
    assert UNDEF_CLS < 0


def test_cat_lut_flipped():
    sample_target1 = torch.cat((
        vaihingen_lut,
        flipped_vaihingen_lut
    ), dim=0)
    print(f"sample_target1: {sample_target1}")
    print(f"sample_target1.size: {sample_target1.size()}")

    onehot = rgb_to_onehot(rgb_target=sample_target1, color_lut=vaihingen_lut)
    print(f"onehot: {onehot}")
    print(f"onehot.size: {onehot.size()}")

    expected_onehot = torch.tensor([0., 1., 2., 3., 4., 5., 5., 4., 3., 2., 1., 0.])
    print(f"expected_onehot): {expected_onehot}")
    print(f"onehot == expected_onehot): {onehot == expected_onehot}")
    assert torch.all(onehot == expected_onehot)


# def test_stack_lut_flipped():
#     sample_target2 = torch.stack([
#         vaihingen_lut,
#         flipped_vaihingen_lut
#     ])
#     print(f"sample_target2: {sample_target2}")
#     print(f"sample_target2.size: {sample_target2.size()}")
#
#     onehot = rgb_to_onehot(rgb_target=sample_target2, color_lut=vaihingen_lut)
#     print(f"onehot: {onehot}")
#     print(f"onehot.size: {onehot.size()}")
#
#     print(
#         f"onehot == torch.tensor([0., 1., 2., 3., 4., 5., 5., 4., 3., 2., 1., 0.]): {onehot == torch.tensor([0., 1., 2., 3., 4., 5., 5., 4., 3., 2., 1., 0.])}")
#     assert torch.all(onehot == torch.tensor([0., 1., 2., 3., 4., 5., 5., 4., 3., 2., 1., 0.]))
#
