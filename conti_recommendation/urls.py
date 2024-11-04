from django.urls import path
from .views import CreateContiView

urlpatterns = [
    path('create_conti/', CreateContiView.as_view(), name='create_conti'),
]
