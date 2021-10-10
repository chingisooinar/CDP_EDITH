#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 14:37:19 2021

@author: chingis
"""
import os
import i2v
from PIL import Image
from glob import glob
import shutil
illust2vec = i2v.make_i2v_with_chainer(
    "illust2vec_tag_ver200.caffemodel", "tag_list.json")

# In the case of caffe, please use i2v.make_i2v_with_caffe instead:
# illust2vec = i2v.make_i2v_with_caffe(
#     "illust2vec_tag.prototxt", "illust2vec_tag_ver200.caffemodel",
#     "tag_list.json")
dataset = 'animefaces256cleaner'
dirs = [f'../{dataset}_male/', f'../{dataset}_noise/', f'../{dataset}_female/']
for d in dirs:
    if not os.path.exists(d):
        os.makedirs(d)
female = 0
male = 0
noise = 0
for idx, image_name in enumerate(glob(f'../{dataset}/*.jpg'), 1):

    print(f'Count: {idx} {image_name}')
    if idx % 10 == 0:
        print(f'Male: {male}, Female: {female}, Noise: {noise}')
    
    img = Image.open(image_name)
    # get specific tags
    pred = illust2vec.estimate_specific_tags([img], ["1girl", "face", "safe", "male", "1boy", 'solo'])[0]
    # check if a guy
    if (pred['1boy'] >= 0.4 or pred['male'] >= 0.4) and pred['1girl'] < max(pred['1boy'], pred['male']):
        # check some general attributes
        if pred["face"] >= 0.25 and pred["safe"] >= 0.5 and pred['solo'] >= 0.25:
            shutil.move(image_name, image_name.replace(dataset, f'{dataset}_male')) 
            male += 1
            continue
        
    elif pred['1girl'] >= 0.4 and pred['1girl'] > max(pred['1boy'], pred['male']):
        if pred["face"] >= 0.25 and pred["safe"] >= 0.5 and pred['solo'] >= 0.25:
            shutil.move(image_name, image_name.replace(dataset, f'{dataset}_female'))
            female += 1
            continue
    
    print(f'The image {image_name}  has a low confidence, so I have to move it')
    shutil.move(image_name, image_name.replace(dataset, f'{dataset}_noise'))
    noise += 1