#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 10:50:53 2021

@author: chingis
"""

import argparse
from easydict import EasyDict as edict
from model import EmbedderWrapper
from utils import extract_Anime_datasets
from mine_topk import kNN
from torchvision import transforms
import shutil
import torch
from DeepFeatures import DeepFeatures
parser = argparse.ArgumentParser(description='PyTorch kNN')
parser.add_argument('--query', default='../animefaces256cleaner_female/', type=str, help='Directory of images you want to visualize')

args = parser.parse_args()
test_transform = transforms.Compose([
        transforms.Resize(224),             # resize shortest side to 224 pixels
        transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    
transforms = (test_transform, test_transform)
query_loader, _ = extract_Anime_datasets(args.query, args.query, transforms)
args = edict(
    arcface = False,
    low_dim = 512,
    instance = None,
    )

model = EmbedderWrapper(args, 2)
device = 'cuda'
model.to_device(device)
model.resume(args, "ckpt_Anime_Feature_Learning_512_0.878080985915493_100.t7")

model.eval()
DF = DeepFeatures(model = model, 
                  imgs_folder = './Outputs/Anime/Images', 
                  embs_folder = './Outputs/Anime/Embeddings', 
                  tensorboard_folder = './Outputs/Tensorboard', 
                  experiment_name= 'Anime_Female')
images = []
for i in range(10):
    batch_imgs, batch_labels, _ = next(iter(query_loader))
    DF.write_embeddings(x = batch_imgs.to(device))
DF.create_tensorboard_log()
