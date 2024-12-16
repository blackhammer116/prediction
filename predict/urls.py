from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_price, name='predict_price'),
    path('predict_quality', views.predict_quality, name='predict_quality'),
]
