#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 13:18:39 2021

@author: chingis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    
    def __init__(self, latent_dim, class_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        
        self.class_dim = class_dim
        self.class_embedding = nn.Sequential(
                    nn.ConvTranspose2d(in_channels = self.class_dim, 
                                       out_channels = 256, 
                                       kernel_size = 4,
                                       stride = 1,
                                       bias = False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace = True),
                )
        self.feature_embedding = nn.Sequential(
                    nn.ConvTranspose2d(in_channels = self.latent_dim, 
                                       out_channels = 1024, 
                                       kernel_size = 4,
                                       stride = 1,
                                       bias = False),
                    nn.BatchNorm2d(1024),
                    nn.ReLU(inplace = True),
                )
        
        self.gen = nn.Sequential(
                    nn.ConvTranspose2d(in_channels = 1024 + 256,
                                       out_channels = 512,
                                       kernel_size = 4,
                                       stride = 2,
                                       padding = 1,
                                       bias = False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace = True),
                    nn.ConvTranspose2d(in_channels = 512,
                                       out_channels = 256,
                                       kernel_size = 4,
                                       stride = 2,
                                       padding = 1,
                                       bias = False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace = True),
                    nn.ConvTranspose2d(in_channels = 256,
                                       out_channels = 128,
                                       kernel_size = 4,
                                       stride = 2,
                                       padding = 1,
                                       bias = False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace = True),
                    nn.ConvTranspose2d(in_channels = 128,
                                       out_channels = 3,
                                       kernel_size = 4,
                                       stride = 2,
                                       padding = 1),
                    nn.Tanh()
                    )
        return
    
    def forward(self, _input, _class):
        x1 = self.feature_embedding(_input.unsqueeze(2).unsqueeze(3))
        x2 = self.class_embedding(_class.unsqueeze(2).unsqueeze(3))
        concat = torch.cat((x1, x2), dim = 1)
        return self.gen(concat)

class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        
        self.class_embedding = nn.Sequential(
                nn.ConvTranspose2d(in_channels = self.num_classes,
                                   out_channels = 32,
                                   kernel_size = 4,
                                   stride = 1,
                                   bias = False),
                nn.LeakyReLU(0.2, inplace = True),
                nn.ConvTranspose2d(in_channels = 32,
                                   out_channels = 64,
                                   kernel_size = 4,
                                   stride = 2,
                                   padding = 1,
                                   bias = False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace = True),
                nn.ConvTranspose2d(in_channels = 64,
                                   out_channels = 128,
                                   kernel_size = 4,
                                   stride = 2,
                                   padding = 1,
                                   bias = False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace = True)
                )
                
        self.feature_embedding = nn.Sequential(
                 nn.Conv2d(in_channels = 3, 
                             out_channels = 128, 
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                nn.LeakyReLU(0.2, inplace = True),
                nn.Conv2d(in_channels = 128, 
                             out_channels = 256, 
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace = True),
                )
                 
        self.discrim = nn.Sequential(
                    nn.Conv2d(in_channels = 256 + 128, 
                             out_channels = 512, 
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.2, inplace = True),
                    nn.Conv2d(in_channels = 512, 
                             out_channels = 1024, 
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    nn.BatchNorm2d(1024),
                    nn.LeakyReLU(0.2, inplace = True),
                    nn.Conv2d(in_channels = 1024,
                             out_channels = 1, 
                             kernel_size = 4,
                             stride = 1),
                    nn.Sigmoid()
                    )
        
                
                
        return
    
    def forward(self, _input, one_hot):
        x1 = self.feature_embedding(_input)
        x2 = self.class_embedding(one_hot.unsqueeze(2).unsqueeze(3))
        #print(x1.shape, x2.shape)
        #print()
        concat = torch.cat((x1, x2), dim = 1)
        return self.discrim(concat).view(-1)
    
    
if __name__ == '__main__':
    # Image Tensor shape: N * C * H * W
    # Batch size, channels, height, width respectively
    
    latent_dim = 100
    embedding_dim = 20
    num_classes = 22
    
    
    z = torch.randn(5, latent_dim)
    classes = torch.randn(5, num_classes)
    
    
    
    g = Generator(latent_dim = latent_dim, 
                  class_dim = num_classes)
    
    
    
    
    
    d = Discriminator(num_classes = num_classes)
    
    
    
    img = g(z, classes)
    
    print(img.shape)
    print(d(img, classes).shape)
    