#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 14:07:08 2021

@author: nuvilabs
"""
import os
from glob import glob
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

class AnimeDataset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.samples = dataset
        self.indices = indices
        self.transform = transform
        self.classes = ['female','male']
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx] 
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)
        return image, target, idx
    
def extract_Anime_datasets(path1, path2, transforms):
    imgs1 = glob(path1 + "*.jpg")
    imgs2 = glob(path2 + "*.jpg")
    
    
    dataset1 = [(img, -1) for img in imgs1]
    dataset2 = [(img, -1) for img in imgs2]


    dataset_1 = AnimeDataset(dataset1, range(len(dataset1)), transforms[1])
    dataset_2 = AnimeDataset(dataset2, range(len(dataset2)), transforms[0])
    
    loader1 = DataLoader(dataset_1, batch_size=256, shuffle=True)
    loader2 = DataLoader(dataset_2, batch_size=256, shuffle=False)
    
    return loader1, loader2

def make_Anime_dataset(male_path, female_path, transforms):
    males = glob(male_path + "*.jpg")
    females = glob(female_path + "*.jpg")
    
    females = females[:len(males)]
    assert len(females) == len(males)
    
    dataset = [(img, 1) for img in males]
    for img in females:
        dataset.append((img, 0 ))
    random.shuffle(dataset)
    
    
    np.random.shuffle(dataset)
    split = int(np.floor(0.9 * len(dataset)))
    tr_dataset, val_dataset = dataset[:split], dataset[split:]

    dataset_train = AnimeDataset(tr_dataset, range(len(tr_dataset)), transforms[1])
    dataset_val = AnimeDataset(val_dataset, range(len(val_dataset)), transforms[0])
    
    train_loader = DataLoader(dataset_train, batch_size=256, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=256, shuffle=False)
    
    return train_loader, val_loader, len(tr_dataset)
    