#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 10:07:25 2021

@author: chingis
"""

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
from networks.pix2pix import PixDiscriminator
from networks.pix2pix import GeneratorUNet
import torchvision.utils as vutils
from style_loss import StyleLoss

# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.0002
BATCH_SIZE = 64
IMAGE_SIZE = 256
CHANNELS_IMG = 3
tmp_path = './bw_to_color/training_temp_1/'
model_dump_path ='./bw_to_color/gan_models'
logfile = './bw_to_color/training.log'
males = glob('../Male_Character_Face_looking/*jpg')
females = glob('../animefaces256cleaner_female/*jpg')
MODE ='inpaint'
transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),

        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
    ]
)

total = males + females

dataset = BWAnimeFaceDataset(total, annotation=None, transforms=transforms, mode=MODE)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8
)


unet = GeneratorUNet().to(device)
discriminator = PixDiscriminator().to(device)



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
unet.train()
discriminator.train()
criterion_GAN = torch.nn.MSELoss(reduction='mean')
criterion_pixelwise = torch.nn.L1Loss()
style_loss = StyleLoss()


start_epoch = 0
g_optimizer = torch.optim.Adam(unet.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
NUM_EPOCHS = 500

for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):

    for batch_idx, (real_src, real_trg) in enumerate(loader):
        real_src = real_src.to(device)
        real_trg = real_trg.to(device)
        cur_batch_size = real_src.shape[0]
        # Create real and fake labels (0/1)
        
        # Train Discriminator
        
        #print(fake.shape)
       # with torch.cuda.amp.autocast():

        fake_trg = unet(real_src)
        d_optimizer.zero_grad()
    
        prediction_real = discriminator(real_trg, real_src)
        error_real = criterion_GAN(prediction_real, torch.ones(len(real_src), 1, 16, 16).cuda())
        error_real.backward()
    
        prediction_fake = discriminator(fake_trg.detach(), real_src)
        error_fake = criterion_GAN(prediction_fake, torch.zeros(len(real_src), 1, 16, 16).cuda())
        error_fake.backward()
    
        d_optimizer.step()
        loss_D = error_real + error_fake
        #discriminator.train()
        g_optimizer.zero_grad()
        prediction = discriminator(fake_trg, real_src)
    
        loss_GAN = criterion_GAN(prediction, torch.ones(len(real_src), 1, 16, 16).cuda())
        loss_pixel = criterion_pixelwise(fake_trg, real_trg)
        loss_style = style_loss(fake_trg, real_trg)
        loss_G = loss_GAN + 100 * loss_pixel + loss_style
    
        loss_G.backward()
        g_optimizer.step()
       
        # Print losses occasionally and print to tensorboard
        if batch_idx % 20 == 0 and batch_idx > 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_D:.4f}, loss G: {loss_G:.4f}"
            )
            torch.save({
                    'epoch': epoch,
                    'D': discriminator.state_dict(),
                    'G': unet.state_dict(),
                    'optimizer_D': d_optimizer.state_dict(),
                    'optimizer_G': g_optimizer.state_dict(),
                  }, '{}/checkpoint.tar'.format(model_dump_path))
            with torch.no_grad():
                fake_trg = unet(real_src)

                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real_src[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake_trg[:32], normalize=True)
                vutils.save_image(img_grid_real.data,
                                  os.path.join(tmp_path, f'real_image_{step % 5}.png'))

                vutils.save_image(img_grid_fake.data,
                                  os.path.join(tmp_path, f'fake_image_{step % 5}.png'))

                logger.info('Saved intermediate file in {}'.format(os.path.join(tmp_path, 'fake_image_{step}.png')))
       

            step += 1
