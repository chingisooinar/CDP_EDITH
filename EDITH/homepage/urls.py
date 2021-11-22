from django.urls import path
from homepage import views

app_name = 'homepage'

urlpatterns = [
    path("",views.index,name="index"),
	path("index/",views.index,name="index"),
	path("result/",views.result,name="result"),
	path("history/",views.history,name="history")
]