from django.http import HttpResponse
from EDITH.settings import STATIC_ROOT
import os,cv2,base64
from api import AiModels

# Create your views here.
def inpainting(request):
    response = AiModels.inpainting('api/sample_whiteline.jpg')
    return HttpResponse(response,content_type="image/png")

def colorize(request):
    response = AiModels.colorizeModel(request)
    return HttpResponse(response,content_type="image/png")

def toSketch(request):
    response = AiModels.toSketchModel(request)
    return HttpResponse(response,content_type="image/png")

def toBw(request):
    response = AiModels.toBwModel(request)
    return HttpResponse(response,content_type="image/png")

def edgeToBw(request):
    response = AiModels.edgeToBwModel(request)
    return HttpResponse(response,content_type="image/png")




#source_file = request.FILES["source"]
#img_im = cv2.imread(os.path.join(STATIC_ROOT,"image/test.png"))
#image_data = base64.b64encode(cv2.imencode('.png',response)[1]).decode()