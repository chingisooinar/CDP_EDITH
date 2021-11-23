from django.shortcuts import render
from EDITH.settings import MEDIA_ROOT
import time,os

# Create your views here.
def index(request):
	if "id" in request.session.keys():
		return request.session["id"]
	else:
		id = int(round(time.time()*1000000))
		request.session["id"] = id
		if os.path.exists(MEDIA_ROOT+"/user/{}".format(id)) == False:
			os.makedirs(MEDIA_ROOT+"/user/{}".format(id))
	return render(request,'index.html',locals())
	
def result(request):
	picUrl = "" # The result url
	return render(request,'result.html',locals())
	
def history(request):
	historyList = list() # The history list
	return render(request,'history.html',locals())