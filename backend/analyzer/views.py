from django.shortcuts import render

# Create your views here.
# backend/analyzer/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny

class HelloAnalyze(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        return Response({"ok": True, "msg": "analyzer up"})
