from django.shortcuts import render

import os
import tempfile
import re
import pdfplumber
# Create your views here.
# backend/analyzer/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view,permission_classes
from rest_framework.permissions import AllowAny
from rest_framework import status
from django.views.decorators.csrf import csrf_exempt

from PyPDF2 import PdfReader
from docx import Document

MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXT = ('.pdf', '.docx')

class HelloAnalyze(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        return Response({"msg": "analyzer working"})




def extract_text_from_pdf(path: str) -> str:
    text_parts = []
    try:
        reader = PdfReader(path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    except Exception:
        return ""
    full = "\n".join(text_parts).strip()
    # collapse multiple whitespace (including newlines) into single spaces, then trim
    cleaned = re.sub(r'\s+', ' ', full).strip()
    return cleaned


def extract_text_from_docx(path: str) -> str:
    try:
        doc = Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text]
        full = "\n".join(paragraphs).strip()
    except Exception:
        return ""
    cleaned = re.sub(r'\s+', ' ', full).strip()
    return cleaned

@csrf_exempt
@permission_classes([AllowAny])
@api_view(['POST'])
def analyze_resume(request):
    """
    Analyzes the uploaded resume and returns extracted information.
    """
    resume_file = request.FILES.get('resume')
    job_description = request.data.get('job_description', '') or ''
    jd_present = bool(job_description and job_description.strip())

    if not resume_file:
        return Response({"error": "No resume file provided."}, status=status.HTTP_400_BAD_REQUEST)
    
    if resume_file.size > MAX_UPLOAD_SIZE:
        return Response({"error": "File size exceeds the maximum limit of 5 MB."}, status=status.HTTP_400_BAD_REQUEST)
    
    filename= resume_file.name.lower()
    _, ext = os.path.splitext(filename)
    if ext not in ALLOWED_EXT:
        return Response({"error": "Unsupported file format. Please upload a PDF or DOCX file."}, status=status.HTTP_400_BAD_REQUEST)
    
    # write to a temporary file and extract
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
    os.close(tmp_fd)
    try:
        with open(tmp_path, 'wb') as f:
            for chunk in resume_file.chunks():
                f.write(chunk)

        extracted_text = ""
        if ext == '.pdf':
            extracted_text = extract_text_from_pdf(tmp_path)
        elif ext == '.docx':
            extracted_text = extract_text_from_docx(tmp_path)

        # cleanup temp file
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

    if not extracted_text:
        return Response({"error": "No extractable text found in the uploaded file."}, status=status.HTTP_400_BAD_REQUEST)

    return Response({
        "extracted_text": extracted_text,
        "jd_present": jd_present
    }, status=status.HTTP_200_OK)