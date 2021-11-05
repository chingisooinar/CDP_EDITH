#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 16:29:47 2021

@author: chingis
"""
import argparse
from easydict import EasyDict as edict
from model import EmbedderWrapper
from utils import extract_Anime_datasets
from mine_topk import kNN
from torchvision import transforms
import shutil
parser = argparse.ArgumentParser(description='PyTorch kNN')

parser.add_argument('--reference', default='../Male_Character_pseudo/', type=str, help='Directory of images where kNN is performed')
parser.add_argument('--query', default='../Male_Character_Face_looking/', type=str, help='Directory of images you want to use to find nearest neighbours')
parser.add_argument('--store', default='../Male_Character_Face_KNN/', type=str, help='where to store kNN')
args = parser.parse_args()
test_transform = transforms.Compose([
        transforms.Resize(224),             # resize shortest side to 224 pixels
        transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

transforms = (test_transform, test_transform)
query_loader, reference_loader = extract_Anime_datasets(args.query, args.reference, transforms)
store_dict = args.store
args = edict(
    arcface = False,
    low_dim = 512,
    instance = None,
    )

model = EmbedderWrapper(args, 2)
device = 'cuda'
model.to_device(device)
model.resume(args, "ckpt_Anime_Feature_Learning_512_0.878080985915493_100.t7")

indices = kNN(model, query_loader, K=1, reference_loader=reference_loader)
print(indices)
reference_imgs = reference_loader.dataset.samples
for idx in indices:
    src, _ = reference_imgs[idx]
    shutil.move(src, store_dict + src.split('/')[-1])
