import cv2
import numpy as np
import torch
import random
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as vutils
from api.common import *
from io import BytesIO
import base64

def UploadResize(request):
    toCanvas = request.FILES.get('image')
    img_array = cv2.imdecode(np.fromstring(toCanvas.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_array = cv2.resize(img_array, (256, 256), interpolation = cv2.INTER_AREA)

    img = Image.fromarray(img_array)
    buffered = BytesIO()
    img.save(buffered, format="png")
    img_str = base64.b64encode(buffered.getvalue())
    
    return img_str

def inpaintingModel(request):
    image = loadImageWithBg(request)
    image=cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    result = inpaintingProcessing(image)
    result = Image.fromarray(result)
    
    buffered = BytesIO()
    result.save(buffered, format="png")
    img_str = base64.b64encode(buffered.getvalue())
    
    return img_str
    
def colorizeModel(request):
    image = loadImageWithBg(request)
    
    color_edges = colorizeProcessing(image)
    color_edges_image = Image.fromarray(color_edges)
    buffered = BytesIO()
    color_edges_image.save(buffered, format="png")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str

def toSketchModel(request):
    image = loadImageWithBg(request)

    image=cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_AREA)
    
    bw_edges = detect_edges(image)
    bw_edges_image = Image.fromarray(bw_edges)
    buffered = BytesIO()
    bw_edges_image.save(buffered, format="png")
    img_str = base64.b64encode(buffered.getvalue())
    
    return img_str

def toBwModel(request):
    image = loadImageWithBg(request)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_AREA)
    
    img_str = cv2.imencode('.png',image)[1].tostring()
    img_str = base64.b64encode(img_str)
    
    return img_str

def edgeToBwModel(request):
    edge_weight = int(request.POST.get('slider'))
    
    image = loadImageWithBg(request)
    edges=cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    edges = detect_edges(image)
    bw_edges = sketchProcessing(edges, edge_weight)
    bw_edges_image = Image.fromarray(bw_edges)
    buffered = BytesIO()
    bw_edges_image.save(buffered, format="png")
    img_str = base64.b64encode(buffered.getvalue())
    
    return img_str
    