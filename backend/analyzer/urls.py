from django.urls import path
from .views import HelloAnalyze

urlpatterns = [
    path('analyze/hello/', HelloAnalyze.as_view(), name='analyze-hello'),
    path('test/', HelloAnalyze.as_view(), name='analyzer-test'),
]
