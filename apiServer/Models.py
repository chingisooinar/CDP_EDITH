import cv2
import numpy as np
import torch
import random
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
from common import *

def sketchProcessing(edges, edges_weight=0):
    """
    input:
        edges: edge map
        edges_weight: model generates bw image out of an edge map. It might blur some edges. \
                        if user wants to emphasize edges, making them stronger, then this optional
                        weight parameter can be passed
    output:
        generated image
    
    """
    _transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(3)], [0.5 for _ in range(3)]),
        ]
    )
    
    sketch_model = GeneratorUNet().cuda()
    cp = torch.load('sketch.tar')
    sketch_model.load_state_dict(cp['G'])
    
    # convert image to PIL image to apply pytorch augmentations
    image = Image.fromarray(np.uint8(edges)).convert('RGB')
    # apply augmentations for preprocessing
    inp = _transforms(image)
    # use cuda
    inp = inp.cuda()
    # feed into a model
    out = sketch_model(inp.reshape(1, 3, 256, 256))
    # get a numpy array(image)
    generated_image = out.detach().cpu().numpy()[0].transpose(1, 2, 0)
    # post processing
    generated_image = (generated_image * 127.5 + 127.5).astype('uint8')
    # remove noise
    denoised = cv2.bilateralFilter(generated_image, 15, 75, 75) # preserves edges 
    denoised = cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY)
    denoised = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
    # make edges stronger
    if edges_weight > 0:
        edges_contour = (edges / 255.0) - 1
        edges_contour[edges_contour == -1] = 1
        denoised = denoised - edges_weight * edges_contour
        denoised[denoised < 0] = 0
    return denoised.astype('uint8')

def generateSketch(request):
    #read a sample image
    image = cv2.imread('anime.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_AREA)
    
    
    edges = detect_edges(image)
    bw_edges = sketchProcessing(edges, 45)
    bw_edges_image = Image.fromarray(bw_edges)
    
    return bw_edges_image

def colorizeProcessing(bw):
    _transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(3)], [0.5 for _ in range(3)]),
        ]
    )
    color_model = GeneratorUNet().cuda()
    cp = torch.load('color.tar')
    color_model.load_state_dict(cp['G'])
    """
    input: 
        bw: bw image with color marks!
    output:
        colored RGB image
    """
    # convert image to PIL image to apply pytorch augmentations
    image = Image.fromarray(np.uint8(bw)).convert('RGB')
    # apply augmentations for pre processing
    inp = _transforms(image)
    # cuda
    inp = inp.cuda()
    # feed into a model
    out = color_model(inp.reshape(1, 3, 256, 256))
    # get numpy array (image)
    np_image = out.detach().cpu().numpy()[0].transpose(1, 2, 0)
    # post process
    img =  (np_image * 127.5 + 127.5).astype('uint8')
    # remove nois( salt&pepper noise)
    dst = cv2.fastNlMeansDenoisingColored(img,None,8,8,7,21)
    return dst #, img

def colorizeModel():
    ########################### for debugging
    image = cv2.imread('anime.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_AREA)
    
    
    edges = detect_edges(image)
    bw_edges = sketchProcessing(edges, 45)
    ########################### for debugging
    marked = mark(bw_edges, image)
    color_edges = colorizeProcessing(marked)