#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 17:20:27 2021

@author: chingis
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from glob import glob
class AnimeDataset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.data = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path, target = self.data[idx] 
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)
        return image, target, idx

test_transform = transforms.Compose([
        transforms.Resize(112),             # resize shortest side to 224 pixels
        transforms.CenterCrop(112),         # crop longest side to 224 pixels at center
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

male_noise = [(img, -1) for img in glob("Male_Character_Face_looking/*.jpg")]

male_noise_dataset = AnimeDataset(male_noise, range(len(male_noise)), test_transform)
n_loader = DataLoader(male_noise_dataset, batch_size=256, shuffle=False)

net = torch.load('pseudo.pt').to(device)
net.eval()
print("========================Testing=================================")
# Run the testing batches
data = 0


indices = []
with torch.no_grad():
    for X_test, y_test, idx in n_loader:
        data += X_test.shape[0]
        X_test, y_test = X_test.to(device), y_test.to(device)
        # Apply the model
        y_pred = net(X_test)
        y_pred = torch.softmax(y_pred, dim=-1)
        # the number of correct predictions
        score, predicted = torch.max(y_pred.data, 1) 
        
        store = idx[(predicted == 0) & (score > 0.99)]
        indices.extend(store.detach().cpu().numpy().tolist())
for idx in indices:
    img, _ = male_noise[idx]
    os.remove(img)