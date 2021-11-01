from django.http import HttpResponse
from EDITH.settings import STATIC_ROOT
import os,cv2,base64

# Create your views here.
def sketch(request):
    #source_file = request.FILES["source"]
    img_im = cv2.imread(os.path.join(STATIC_ROOT,"image/test.png"))
    image_data = base64.b64encode(cv2.imencode('.png',img_im)[1]).decode()
    return HttpResponse(image_data,content_type="image/png")