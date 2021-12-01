from django.http import HttpResponse
from EDITH.settings import STATIC_ROOT, MEDIA_ROOT
import os,cv2,base64,time
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

def complete(request):
    id = request.session["id"]
    filename = int(round(time.time()*1000000))
    image_data = base64.b64decode(request.POST["image"][22:])
    f = open("{}/user/{}/{}.png".format(MEDIA_ROOT,id,filename),"wb")
    f.write(image_data)
    f.close()
    return HttpResponse(filename)

def deleteHistory(request):
	try:
		os.remove("{}/user/{}".format(MEDIA_ROOT,request.POST["filename"]))
		return HttpResponse(1)
	except:
		return HttpResponse(0)