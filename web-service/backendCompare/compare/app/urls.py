from django.urls import path
from app.views import AlgorithmsPostView

urlpatterns = [
    path('methods/', AlgorithmsPostView.as_view(), name='methods'),
]