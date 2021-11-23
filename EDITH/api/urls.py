from django.urls import path
from api import views

app_name = 'api'

urlpatterns = [
    path("sketch_api/",views.sketch,name="sketch"),
    path("colorize_api/",views.colorize,name="colorize"),
    path("convert_to_sketch_api/",views.toSketch,name="toSketch"),
    path("convert_to_bw_api/",views.toBw,name="toBw")
]