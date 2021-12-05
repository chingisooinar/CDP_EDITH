from django.urls import path
from api import views

app_name = 'api'

urlpatterns = [
    path("inpainting_api/",views.inpainting,name="inpainting"),
    path("colorize_api/",views.colorize,name="colorize"),
    path("convert_to_sketch_api/",views.toSketch,name="toSketch"),
    path("convert_to_bw_api/",views.toBw,name="toBw"),
    path("convert_edge_to_bw_api/",views.edgeToBw,name="edgeToBw"),
    path("upload_resize_api/",views.uploadResize,name="uploadResize"),
    path("complete/",views.complete,name="complete"),
    path("deleteHistory/",views.deleteHistory,name="deleteHistory")
]