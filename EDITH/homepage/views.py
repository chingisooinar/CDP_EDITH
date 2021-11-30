from django.shortcuts import render
from EDITH.settings import MEDIA_ROOT
import time,os,cv2,base64

# Create your views here.
def index(request):
	id = int(round(time.time()*1000000))
	if "id" not in request.session.keys():
		request.session["id"] = id
		request.session.save()
	else:
		id = request.session["id"]
	if os.path.exists("{}/user/{}".format(MEDIA_ROOT,id)) == False:
		os.makedirs("{}/user/{}".format(MEDIA_ROOT,id))
	pic = ""
	if request.method == "POST":
		picFilename = request.POST["image"]
		img = cv2.imread(picFilename)
		pic = base64.b64encode(cv2.imencode('png', img)[1]).decode()
	return render(request,'index.html',locals())
	
def result(request):
	picUrl = "" # The result url
	return render(request,'result.html',locals())
	
def history(request):
	historyList = list() # The history list
	if "id" in request.session.keys():
		historyList = ["/{}/{}".format(request.session["id"],filename) for filename in os.listdir("{}/user/{}/".format(MEDIA_ROOT,request.session["id"]))]
	return render(request,'history.html',locals())