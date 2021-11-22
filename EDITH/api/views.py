from django.http import HttpResponse
from EDITH.settings import STATIC_ROOT
import os,cv2,base64
from api import AiModels

# Create your views here.
def sketch(request):
    #source_file = request.FILES["source"]
    
    response = AiModels.generateSketch('api/anime.jpg')
    
    
    #img_im = cv2.imread(os.path.join(STATIC_ROOT,"image/test.png"))
    #image_data = base64.b64encode(cv2.imencode('.png',response)[1]).decode()
    
    return HttpResponse(response,content_type="image/png")

def colorize(request):
    response = AiModels.colorizeModel(request)
    return HttpResponse(response,content_type="image/png")

def toSketch(request):
    response = cv2.imread("api/anime.jpg")
    return HttpResponse(response,content_type="image/png")

def toBw(request):
    response = cv2.imread("api/anime.jpg")
    return HttpResponse(response,content_type="image/png")