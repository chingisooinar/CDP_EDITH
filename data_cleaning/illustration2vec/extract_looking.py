#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 16:02:51 2021

@author: chingis
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 14:37:19 2021
 
@author: chingis
Extracts images that looks at a viewer
"""

import i2v
from PIL import Image
from glob import glob
import shutil
import json
from tqdm import tqdm
illust2vec = i2v.make_i2v_with_chainer(
    "illust2vec_tag_ver200.caffemodel", "tag_list.json")

# In the case of caffe, please use i2v.make_i2v_with_caffe instead:
# illust2vec = i2v.make_i2v_with_caffe(
#     "illust2vec_tag.prototxt", "illust2vec_tag_ver200.caffemodel",
#     "tag_list.json")
tagged = {}
counter = {}
directory = 'Male_Character_Face_Noise'

needed_tags = ['looking at viewer', 'face']
for idx, image_name in tqdm(enumerate(glob(f'../{directory}/*.jpg'), 1)):

    img = Image.open(image_name)
    # get specific tags
    pred = illust2vec.estimate_specific_tags([img], needed_tags)[0]
    # check if a guy

    looking_at_viewer = (pred['looking at viewer'] >= 0.25) * 1.
    face = (pred['face'] >= 0.25) * 1.
    if looking_at_viewer == 1 and face == 1:
        print(image_name)
        shutil.move(image_name, image_name.replace(directory, 'Male_Character_Face_looking'))