from django.urls import path
from api import views

app_name = 'api'

urlpatterns = [
    path("sketch_api/",views.sketch,name="sketch"),
]