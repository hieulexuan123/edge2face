from torchvision.utils import save_image
import os
import torch
import config

def denormalize(imgs, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    for i in range(3):
        imgs[:, i, :, :] = imgs[:, i, :, :] * std[i] + mean[i]
    return imgs

def save_val_predictions(gen, val_loader, epoch, folder_path):
    x, y = next(iter(val_loader))
    x, y_real = x.to(config.device), y.to(config.device)

    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_real = denormalize(y_real)
        y_fake = denormalize(y_fake)
        concat_imgs = torch.cat([y_real, y_fake], dim=2)
        for i in range(len(concat_imgs)):
            concat_img = concat_imgs[i]
            save_image(concat_img, os.path.join(folder_path, f"image_{i}_epoch_{epoch}.png"))
    gen.train()

def save_loss(loss, epoch, file_path):
    with open(file_path, "a") as f:
        f.write(f'{epoch} {loss}\n')

def save_checkpoint(net, optimizer, path):
    ckpt = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(ckpt, path)
    print(f'Save model into {path} successfully')

def load_checkpoint(net, path, optimizer=None):
    ckpt = torch.load(path, map_location=config.device)
    net.load_state_dict(ckpt['net'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    print(f'Load model from {path} successfully')