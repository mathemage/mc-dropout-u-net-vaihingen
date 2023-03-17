import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from rgb_to_categorical_vaihingen import rgb_to_onehot
from utils.dice_score import multiclass_dice_coeff, dice_coeff, dice_loss


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    criterion = nn.CrossEntropyLoss()

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        mask_true = rgb_to_onehot(rgb_target=mask_true, quiet=True)
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true_one_hot = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true_one_hot, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true_one_hot[:, 1:, ...],
                                                    reduce_batch_first=False)

            # loss = criterion(mask_pred, mask_true) \
            #        + dice_loss(F.softmax(mask_pred, dim=1).float(),
            #                    F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float(),
            #                    multiclass=True)
            loss = criterion(mask_pred, mask_true)

    net.train()

    # Fixes a potential division by zero error
    score = dice_score if num_val_batches == 0 else dice_score / num_val_batches
    return score, loss
