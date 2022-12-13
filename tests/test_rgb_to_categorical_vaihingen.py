# from https://stackoverflow.com/questions/64300830/pytorch-tensor-get-the-index-of-the-element-with-specific-values

import torch

from rgb_to_categorical_vaihingen import UNDEF_CLS, vaihingen_lut

flipped_vaihingen_lut = torch.flip(vaihingen_lut, dims=[0])
# print(f"flipped_vaihingen_lut: {flipped_vaihingen_lut}")


def test_undef_cls():
    assert UNDEF_CLS < 0


# def test_cat_lut_flipped():
#     sample_target1 = torch.cat((
#         vaihingen_lut,
#         flipped_vaihingen_lut
#     ), dim=0)
#     print(f"sample_target1: {sample_target1}")
#     print(f"sample_target1.size: {sample_target1.size()}")
#     rgb_to_onehot(rgb_target=sample_target1, color_lut=vaihingen_lut)

# if __name__ == "__main__":
#
#     rgb_to_onehot()
#
#     sample_target2 = torch.stack([
#         vaihingen_lut,
#         flipped_vaihingen_lut
#     ])
#     print(f"sample_target2: {sample_target2}")
#     print(f"sample_target2.size: {sample_target2.size()}")
#     rgb_to_onehot(rgb_target=sample_target2, color_lut=vaihingen_lut)
