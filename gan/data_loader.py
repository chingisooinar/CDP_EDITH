__author__ = 'Wendong Xu'
import os
from torch.utils.data import Dataset
import numpy as np
import torch
import pickle
import cv2
import json
import torchvision.transforms as T

import random
from PIL import Image
class BWAnimeFaceDataset(Dataset):
    def __init__(self, images, annotation, transforms, mode='colorize'):
        # tag's one-hot, image-bytes
        self.images = images
        with open(annotation) as f:
            self.annotation = json.load(f)
        self.t = transforms
        self.mode = mode
        self.normalize = lambda x: (x - 127.5)/127.5
    def detect_edges(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gray = cv2.bilateralFilter(img_gray, 5, 50, 50)
        img_gray_edges = cv2.Canny(img_gray, 65, 110)
        img_gray_edges = cv2.bitwise_not(img_gray_edges) # invert black/white
        img_edges = cv2.cvtColor(img_gray_edges, cv2.COLOR_GRAY2RGB)
        return img_edges

    def __getitem__(self, index):
        filename = self.images[index]
        tag_one_hot = np.asarray(self.annotation[filename][:6] + self.annotation[filename][7:8])
        assert len(tag_one_hot) == 7
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_AREA)
        if self.mode == 'colorize':
            img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            img_gray, image = self.normalize(img_gray), self.normalize(image)
            self._draw_color_circles_on_src_img(img_gray, image)
            img = (img_gray * 127.5) + 127.5
            image = (image * 127.5) + 127.5
        elif self.mode == 'sketch':
            img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            edges = self.detect_edges(image)
            img = edges
            image = img_gray
        else:
            img = image
        
        PIL_image = Image.fromarray(np.uint8(img)).convert('RGB')
        if self.mode in ('colorize', 'sketch'):
            og_PIL = Image.fromarray(np.uint8(image)).convert('RGB')
            return self.t(PIL_image), self.t(og_PIL)
        else:
            return tag_one_hot.astype('float32'), self.t(PIL_image)
    
    def _draw_color_circles_on_src_img(self, img_src, img_target):
        non_white_coords = self._get_non_white_coordinates(img_target)
        for center_y, center_x in non_white_coords:
            self._draw_color_circle_on_src_img(img_src, img_target, center_y, center_x)

    def _get_non_white_coordinates(self, img):
        non_white_mask = np.sum(img, axis=-1) < 2.75
        non_white_y, non_white_x = np.nonzero(non_white_mask)
        # randomly sample non-white coordinates
        choices = [300, 400, 800, 600]
        n_non_white = len(non_white_y)
        n_color_points = min(n_non_white, random.choice(choices))
        idxs = np.random.choice(n_non_white, n_color_points, replace=False)
        non_white_coords = list(zip(non_white_y[idxs], non_white_x[idxs]))
        return non_white_coords

    def _draw_color_circle_on_src_img(self, img_src, img_target, center_y, center_x):
        assert img_src.shape == img_target.shape, "Image source and target must have same shape."
        y0, y1, x0, x1 = self._get_color_point_bbox_coords(center_y, center_x)
        color = np.mean(img_target[y0:y1, x0:x1], axis=(0, 1))
        img_src[y0:y1, x0:x1] = color

    def _get_color_point_bbox_coords(self, center_y, center_x):
        radius = 2
        y0 = max(0, center_y-radius+1)
        y1 = min(256, center_y+radius)
        x0 = max(0, center_x-radius+1)
        x1 = min(256, center_x+radius)
        return y0, y1, x0, x1
    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    from glob import glob
    transforms = T.Compose(
    [
        T.RandomHorizontalFlip(),
        T.Resize(64),
        T.ToTensor(),
        T.Normalize(
            [0.5 for _ in range(3)], [0.5 for _ in range(3)]),
    ]
)
    females = glob('../Female_Character_Face/*jpg')
    males = glob('../Male_Character_Face/*jpg')
    males.extend(females)
    random.shuffle(males)
    annotation = 'Anime_annotations.json'
    dataset = BWAnimeFaceDataset(males, annotation, transforms)
