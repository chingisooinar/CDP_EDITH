from django.shortcuts import render

# Create your views here.
def index(request):
    return render(request,'index.html',locals())
	
def result(request):
	picUrl = "" # The result url
	return render(request,'result.html',locals())
	
def history(request):
	historyList = list() # The history list
	return render(request,'history.html',locals())