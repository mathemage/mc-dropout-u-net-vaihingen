import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms

import wandb
from patchify_dataset import patchify_dataset
from rgb_to_categorical_vaihingen import rgb_to_onehot
from test_on_vaihingen import test_net
from utils.EarlyStopper import EarlyStopper

wandb.init(project="MCD-U-Net-Vaihingen", entity="mathemage")

from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from evaluate_on_vaihingen import evaluate
from unet import UNet

# dir_img = Path('./data/vaihingen/imgs/')
# dir_mask = Path('./data/vaihingen/masks/')
dir_img = Path('./data/vaihingen/trainset/imgs/patches_128x128x3/')
dir_mask = Path('./data/vaihingen/trainset/masks/patches_128x128x3/')

dir_checkpoint = Path('./checkpoints/vaihingen/')


def train_net(
        net, device, epochs: int = 5, batch_size: int = 1, learning_rate: float = 1e-3, val_percent: float = 0.1,
        save_checkpoint: bool = True, img_scale: float = 0.5, amp: bool = False, use_histograms=False,
        flip_horizontally=False, flip_vertically=False
):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    data_augmentation = []
    if flip_horizontally:
        data_augmentation.append(transforms.RandomHorizontalFlip())
    if flip_vertically:
        data_augmentation.append(transforms.RandomVerticalFlip())
    preprocessors = transforms.Compose(data_augmentation)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
    # learning rate is reduced on the plateau (learning rate divided by 10  if  no  decay  in  the  validation  loss  is
    #  observed  in the 10 last epochs)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    early_stopper = EarlyStopper(patience=20)  # also perform early stopping (stop the training if no decay in the
    early_stop = 0
    # validation loss is observed in the 20 last epochs)
    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']
                true_masks = rgb_to_onehot(rgb_target=true_masks, quiet=True)

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                # https://discuss.pytorch.org/t/for-segmentation-how-to-perform-data-augmentation-in-pytorch/89484/5
                state = torch.get_rng_state()
                images = preprocessors(images)
                torch.set_rng_state(state)
                true_masks = preprocessors(true_masks)

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, true_masks) \
                           + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # log only once a while, not after every training step
                log_wandb_every_steps = 10
                division_step = (n_train // (log_wandb_every_steps * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                        })

                # optimize memory by deallocating on CUDA
                del true_masks
                torch.cuda.empty_cache()

            # Evaluation round
            histograms = {}
            if use_histograms:
                for tag, value in net.named_parameters():
                    tag = tag.replace('/', '.')
                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

            val_score, val_loss = evaluate(net, val_loader, device)
            scheduler.step(val_loss)
            if val_loss is not None and early_stopper.early_stop(val_loss):
                logging.critical(f"Early stop at epoch {epoch}. val_loss == {val_loss}, val_score == {val_score}")
                logging.info(f"val_loss == {val_loss}\n"
                             f"val_score == {val_score}")
                early_stop = 1

            logging.info('Validation Dice score: {}'.format(val_score))
            experiment.log({
                'validation Dice': val_score,
                'validation loss': val_loss,
                'epoch': epoch,
                'early_stop': early_stop,
                **histograms
            })

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

        if early_stop:
            break


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=64, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')

    # This data set contains 33 images with associated DSM. 16 ground-truth images are provided for
    # training.
    # one_sixteenth = 1.0 / 16.0  # We use one of them as validation set and the remaining images as training models.
    one_tenth = 0.1
    parser.add_argument('--validation', '-v', dest='val', type=float, default=one_tenth * 100,
                        help='Percent of the data that is used as validation (0-100)')

    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--histograms', action='store_true', default=False, help='Use histograms to track weights and '
                                                                                 'gradients')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    dir_img_is_valid = os.path.isdir(dir_img) and (os.listdir(dir_img) is not None)
    dir_mask_is_valid = os.path.isdir(dir_mask) and (os.listdir(dir_mask) is not None)
    logging.info(f"dir_img_is_valid == {dir_img_is_valid}")
    logging.info(f"dir_mask_is_valid == {dir_mask_is_valid}")
    if not (dir_img_is_valid and dir_mask_is_valid):
        logging.critical(f"Patchifying dataset:")
        patchify_dataset()
    logging.info(f"dir_img: {dir_img}")
    logging.info(f"dir_mask: {dir_mask}")

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    logging.info('Training phase:')
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp,
                  use_histograms=args.histograms)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt (training)')
        raise

    logging.info('Testing phase:')
    try:
        test_net(net=net,
                 device=device,
                 img_scale=args.scale,
                 amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt (testing)')
        raise
