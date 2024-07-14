import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from model import Discriminator, Generator
from torch.optim import lr_scheduler
from data import AlignedDataset
import config
from utils import *

def get_scheduler(optimizer):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + config.load_epoch + 1 - config.num_epochs) / float(config.num_epochs_decay + 1)
        print(f'Percentage of original lr: {lr_l}')
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler

if __name__ == '__main__':
    save_model_dir = os.path.join(config.checkpoint_dir, config.experiment_name)
    log_img_dir = os.path.join(config.log_img_dir, config.experiment_name)
    log_loss_dir = os.path.join(config.log_loss_dir, config.experiment_name)
    
    disc = Discriminator().to(config.device)
    gen = Generator().to(config.device)
    optim_disc = optim.Adam(disc.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
    optim_gen = optim.Adam(gen.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
    scheduler_disc = get_scheduler(optim_disc)
    scheduler_gen = get_scheduler(optim_gen)

    if config.gan_mode=='vanilla':
        gan_loss = nn.BCEWithLogitsLoss()
    elif config.gan_mode=='lsgan':
        gan_loss = nn.MSELoss()
    else:
        raise NotImplementedError(f'{config.gan_mode} not implemented')
    l1_loss = nn.L1Loss()

    if config.continue_train:
        load_checkpoint(disc, os.path.join(save_model_dir, f'disc_epoch_{config.load_epoch}'), optim_disc)
        load_checkpoint(gen, os.path.join(save_model_dir, f'gen_epoch_{config.load_epoch}'), optim_gen)
    
    train_dataset = AlignedDataset(os.path.join('dataset', config.dataset_name, 'train'))
    val_dataset = AlignedDataset(os.path.join('dataset', config.dataset_name, 'val'))
    train_loader = DataLoader(train_dataset, config.batch_size, True, num_workers=config.num_threads)
    val_loader = DataLoader(val_dataset, config.batch_size, False)

    best_gen_loss = 100

    for epoch in range(config.load_epoch + 1, config.num_epochs + config.num_epochs_decay + 1):
        disc.train()
        gen.train()
        disc_losses = []
        gen_losses = []
        for i, (x, y_real) in tqdm(train_loader):
            x = x.to(config.device)
            y_real = y_real.to(config.device)

            y_fake = gen(x)
            pred_fake = disc(x, y_fake.detach())
            pred_real = disc(x, y_real)

            disc_real_loss = gan_loss(pred_real, torch.ones_like(pred_real))
            disc_fake_loss = gan_loss(pred_fake, torch.zeros_like(pred_fake))
            disc_loss = (disc_real_loss + disc_fake_loss) / 2
            disc_losses.append(disc_loss.item())
            disc_loss.backward()
            optim_disc.step()
            optim_disc.zero_grad()

            pred_fake = disc(x, y_fake)
            gen_loss_gan = gan_loss(pred_fake, torch.ones_like(pred_fake))
            gen_loss_l1 = l1_loss(y_fake, y_real)
            gen_loss = gen_loss_gan + gen_loss_l1 * config.l1_lambda
            gen_losses.append(gen_loss.item())
            gen_loss.backward()
            optim_gen.step()
            optim_gen.zero_grad()
        
        scheduler_disc.step()
        scheduler_gen.step()
        
        save_val_predictions(gen, val_loader, epoch, log_img_dir)
        mean_disc_loss = np.mean(np.array(disc_losses)).item()
        mean_gen_loss = np.mean(np.array(gen_losses)).item()
        print(f'epoch {epoch}, disc_loss: {mean_disc_loss}, gen_loss: {mean_gen_loss}')
        save_loss(mean_disc_loss, epoch, os.path.join(log_loss_dir, 'disc.txt'))
        save_loss(mean_gen_loss, epoch, os.path.join(log_loss_dir, 'gen.txt'))

        if mean_gen_loss < best_gen_loss:
            best_gen_loss = mean_gen_loss
            save_checkpoint(disc, optim_disc, os.path.join(save_model_dir, f'disc_best.pth'))
            save_checkpoint(gen, optim_gen, os.path.join(save_model_dir, f'gen_best.pth'))
        
        save_checkpoint(disc, optim_disc, os.path.join(save_model_dir, f'disc_latest.pth'))
        save_checkpoint(gen, optim_gen, os.path.join(save_model_dir, f'gen_latest.pth'))
        
        # if epoch % save_epoch_freq == 0:             
        #     save_checkpoint(disc, optim_disc, os.path.join(save_model_dir, f'disc_epoch_{epoch}.pth'))
        #     save_checkpoint(gen, optim_gen, os.path.join(save_model_dir, f'gen_epoch_{epoch}.pth'))