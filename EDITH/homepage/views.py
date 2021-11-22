from django.shortcuts import render

# Create your views here.
def index(request):
    return render(request,'index.html',locals())
	
def result(request):
	return render(request,'result.html',locals())
	
def history(request):
	return render(request,'history.html',locals())