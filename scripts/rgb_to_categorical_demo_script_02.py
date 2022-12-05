# from https://stackoverflow.com/questions/43884463/how-to-convert-rgb-image-to-one-hot-encoded-3d-array-based-on-color-using-numpy

# Six categories/classes have been defined:
#
# Impervious surfaces (RGB: 255, 255, 255)
# Building (RGB: 0, 0, 255)
# Low vegetation (RGB: 0, 255, 255)
# Tree (RGB: 0, 255, 0)
# Car (RGB: 255, 255, 0)
# Clutter/background (RGB: 255, 0, 0)
color_dict = {0: (255, 255, 255),
              1: (0, 0, 255),
              2: (0, 255, 255),
              3: (0, 255, 0),
              4: (255, 255, 0),
              5: (255, 0, 0)
              }


def rgb_to_onehot(rgb_arr, color_dict):
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2] + (num_classes,)
    arr = np.zeros(shape, dtype=np.int8)
    for i, cls in enumerate(color_dict):
        arr[:, :, i] = np.all(rgb_arr.reshape((-1, 3)) == color_dict[i], axis=1).reshape(shape[:2])
    return arr


import torch

print(torch.__version__)
# convert rgb mask to integer class labels
#   red    =  class-0
#   green  =  class-1
#   blue   =  class-2
#   black  =  class-3
#   white  =  class-4
#   (bad)  =  class-5
# create example mask tensor of shape [nBatch, nRGB = 3, height]
nBatch = 2
height = 4
rgb_mask = 255 * torch.ones((nBatch, 3, height), dtype=torch.int64)
rgb_mask[0, :, 0] = 0  # black
rgb_mask[0, [1, 2], 1] = 0  # red
rgb_mask[0, [0, 2], 2] = 0  # green
rgb_mask[0, [0, 1], 3] = 0  # blue
rgb_mask[1, :, 0] = 255  # white
rgb_mask[1, :, 1] = 0  # black
rgb_mask[1, [0, 1], 2] = 0  # blue
rgb_mask[1, [1, 2], 3] = 0  # red
print(rgb_mask.shape)
print(rgb_mask)
# build lookup table
lut_dim = torch.tensor(rgb_mask.shape)
lut_dim[1] = -1
lut = torch.tensor([3, 0, 1, 5, 2, 5, 5, 4])  # color-class encoding
lut = lut.unsqueeze(-1).expand(lut_dim.tolist())
# convert rgb colors to 0-7 indices
label_mask = torch.sign(rgb_mask)
powers2 = torch.tensor([2]) ** torch.arange(3)
label_mask = (powers2 * label_mask.transpose(1, -1)).transpose(1, -1).sum(dim=1, keepdim=True)
# index into lut to get integer class labels
label_mask = torch.gather(lut, 1, label_mask).squeeze()
print(label_mask.shape)
print(label_mask)
