#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 16:29:47 2021

@author: chingis
"""

from easydict import EasyDict as edict
from model import EmbedderWrapper
from utils import extract_Anime_datasets
from mine_topk import kNN
from torchvision import transforms
import shutil

test_transform = transforms.Compose([
        transforms.Resize(224),             # resize shortest side to 224 pixels
        transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

transforms = (test_transform, test_transform)


query_loader, reference_loader = extract_Anime_datasets('../Male_Character_Reference/', '../Male_Character_Face_Noise/', transforms)




args = edict(
    arcface = False,
    low_dim = 512,
    instance = None,
    
    )
model = EmbedderWrapper(args, 2)
device = 'cuda'
model.to_device(device)
model.resume(args, "ckpt_Anime_Feature_Learning_512_0.878080985915493_100.t7")

indices = kNN(model, query_loader, K=5, reference_loader=reference_loader)
print(indices)
reference_imgs = reference_loader.dataset.samples
for idx in indices:
    src, _ = reference_imgs[idx]
    shutil.move(src, src.replace('Male_Character_Face_Noise', 'Male_Character_Face_KNN'))
