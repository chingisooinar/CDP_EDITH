#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 13:19:20 2021

@author: chingis
"""

import os
from data_loader import BWAnimeFaceDataset
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from glob import glob
import random
from gradient_penalty import gradient_penalty, save_checkpoint, load_checkpoint
from networks.txtCGAN import Generator
from networks.txtCGAN import Discriminator
import torchvision.utils as vutils
import numpy as np
def mismatch(target):
    batch_size = target.shape[0]
    classes = target.shape[1]
    wrong = torch.zeros(target.shape)
    for i in range(batch_size):
        c = torch.max(target[i, :], 0)[1]
        shifted = (c + np.random.randint(classes - 1)) % classes
        wrong[i][shifted] = 1
    return wrong
def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 3
NUM_CLASSES = 9
GEN_EMBEDDING = 100
Z_DIM = 122
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10
tmp_path = './animefaces256/training_temp_1/'
model_dump_path ='./animefaces256/gan_models'
transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
    ]
)

#females = glob('../Female_Character_Face/*jpg')
males = glob('../animefaces256cleaner_female/*jpg')
#males.extend(females)
random.shuffle(males)
annotation = 'animefaces256cleaner_female_annotation.json'
dataset = BWAnimeFaceDataset(males, annotation, transforms)
#dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
# comment mnist above and uncomment below for training on CelebA
#dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# initialize gen and disc, note: discriminator should be called critic,
# according to WGAN paper (since it no longer outputs between [0, 1])
gen = Generator(latent_dim=Z_DIM, class_dim=7).to(device)#(Z_DIM, CHANNELS_IMG, FEATURES_GEN, NUM_CLASSES, IMAGE_SIZE, GEN_EMBEDDING).to(device)
critic = Discriminator(num_classes=7).to(device)#(CHANNELS_IMG, FEATURES_CRITIC, NUM_CLASSES, IMAGE_SIZE).to(device)

initialize_weights(gen)
initialize_weights(critic)

# initializate optimizer


g_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()

# for tensorboard plotting
fixed_noise = torch.randn(32, Z_DIM).to(device)
logfile = './animefaces256/training.log'
logger = logging.getLogger()
logger.setLevel(logging.INFO)
rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log = logging.FileHandler(logfile, mode='w+')
log.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
log.setFormatter(formatter)

plog = logging.StreamHandler()
plog.setLevel(logging.INFO)
plog.setFormatter(formatter)

logger.addHandler(log)
logger.addHandler(plog)

logger.info('Currently use {} for calculating'.format(device))

step = 0
smooth = 0.9
std = 10
gen.train()
critic.train()
criterion = torch.nn.BCELoss()
start_epoch = 0
checkpoint = torch.load('./animefaces256/gan_models/checkpoint1.tar')
gen.load_state_dict(checkpoint['G'])
critic.load_state_dict(checkpoint['D'])
logger.info('Load Optimizers')
opt_gen = optim.Adam(gen.parameters(), lr=0.0001, betas=(0.5, 0.999))
opt_critic = optim.Adam(critic.parameters(), lr=0.0002, betas=(0.5, 0.999))
#opt_gen.load_state_dict(checkpoint['optimizer_G'])
#opt_critic.load_state_dict(checkpoint['optimizer_D'])
#start_epoch = checkpoint['epoch']
NUM_EPOCHS = start_epoch + 500
for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):

    for batch_idx, (labels, real) in enumerate(loader):
        
        real = real.to(device)
        labels = labels.to(device)
        cur_batch_size = real.shape[0]
        alpha = torch.rand((cur_batch_size, 3, 64, 64)).to(device) * std / (step + 1)
        # Create real and fake labels (0/1)
        real_label = torch.ones(cur_batch_size).to(device)
        fake_label = torch.zeros(cur_batch_size).to(device)
        soft_label = torch.Tensor(cur_batch_size).uniform_(smooth, 1).to(device)
        wrong_class = mismatch(labels).to(device)
       
        noise = torch.randn(cur_batch_size, Z_DIM).to(device)
        # Train Discriminator
        
        #print(fake.shape)
       # with torch.cuda.amp.autocast():
        fake = gen(noise, labels)
        critic_real = critic(real + alpha, labels)#.reshape(-1)
        critic_fake = critic(fake.detach() + alpha , labels)#.reshape(-1)
        critic_w_real = critic(real + alpha, wrong_class)
        #print(critic_fake.min(), critic_fake.max())
        gp = gradient_penalty(critic, labels, real, fake, device=device)
        loss_critic = 0.5 * gp + (criterion(critic_real, soft_label) +
                    (criterion(critic_w_real, fake_label) +
                    criterion(critic_fake, fake_label)) * 0.5)
#(criterion(critic_real, real_label) +
                    #(criterion(critic_w_real, fake_label) +
                    #criterion(critic_fake, fake_label)) * 0.5) + 
        opt_critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        for _ in range(3):
            alpha = torch.rand((cur_batch_size, 3, 64, 64)).to(device) * std / (step + 1)
            noise = torch.randn(cur_batch_size, Z_DIM).to(device)
            fake = gen(noise, labels)
            # Train Discriminator
            gen_fake = critic(fake + alpha, labels)#.reshape(-1)
            loss_gen = criterion(gen_fake, real_label)    #criterion(gen_fake, real_label) +
            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()
   
        # Print losses occasionally and print to tensorboard
        if batch_idx % 20 == 0 and batch_idx > 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )
            torch.save({
                    'epoch': epoch,
                    'D': critic.state_dict(),
                    'G': gen.state_dict(),
                    'optimizer_D': opt_critic.state_dict(),
                    'optimizer_G': opt_gen.state_dict(),
                  }, '{}/checkpoint.tar'.format(model_dump_path))
            with torch.no_grad():
                fake = gen(noise, labels)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                vutils.save_image(img_grid_real.data,
                                  os.path.join(tmp_path, f'real_image_{step}.png'))

                vutils.save_image(img_grid_fake.data,
                                  os.path.join(tmp_path, f'fake_image_{step}.png'))

                logger.info('Saved intermediate file in {}'.format(os.path.join(tmp_path, 'fake_image_{step}.png')))
                        #writer_real.add_image("Real", img_grid_real, global_step=step)
                #writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1