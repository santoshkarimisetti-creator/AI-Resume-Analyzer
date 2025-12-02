from django.urls import path
from .views import HelloAnalyze,analyze_resume

urlpatterns = [
    path('analyze/hello/', HelloAnalyze.as_view(), name='analyze-hello'),
    path('test/', HelloAnalyze.as_view(), name='analyzer-test'),
    path('analyze/resume/', analyze_resume, name='analyze-resume'),
]
