from django.shortcuts import render

import requests
from django.conf import settings
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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import text as sklearn_text
from collections import Counter


EXTRA_STOPWORDS = {
    # --- HR / Recruiting Filler ---
    "looking", "ideal", "candidate", "role", "position", "job", "opportunity",
    "team", "company", "client", "clients", "environment", "culture",
    "responsibilities", "responsible", "duties", "requirements", "qualification",
    "qualifications", "preferred", "plus", "advantage", "bonus", "degree",
    "bachelor", "masters", "equivalent", "field", "related", "relevant",
    
    # --- Generic Descriptors ---
    "strong", "excellent", "good", "great", "proficient", "proficiency",
    "demonstrated", "proven", "track", "record", "solid", "deep",
    "understanding", "knowledge", "experience", "familiarity", "expert",
    "hands-on", "passion", "passionate", "motivated", "self-starter",
    "detail", "oriented", "analytical", "problem-solving", "communication",
    "collaborative", "interpersonal", "organizational", "highly",
    
    # --- Common Verbs/Connectors (Non-Technical) ---
    "ability", "able", "use", "using", "used", "work", "working",
    "collaborate", "support", "assist", "ensure", "participate",
    "maintain", "develop", "design", "implement", "build", "create", 
    "manage", "drive", "across", "various", "including", "based",
    "within", "daily", "tasks", "activities", "solutions", "services",
    "projects", "systems", "applications", "years", "best", "practices"
}
STOPWORDS = sklearn_text.ENGLISH_STOP_WORDS.union(EXTRA_STOPWORDS)

from PyPDF2 import PdfReader
from docx import Document

MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXT = ('.pdf', '.docx')

# --- Text Preprocessing Helper Function ---
def preprocess_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def calculate_match_score(
        resume_text: str,
        job_desc: str,
        matched_keywords=None,
        missing_keywords=None
    ) -> int:
    """
    Combines cosine similarity and keyword coverage into a 0-100 match score.
    """
    if not resume_text or not job_desc:
        return 0

    # --- A) Cosine similarity part ---
    try:
        docs = [job_desc, resume_text]
        cv = TfidfVectorizer(
            stop_words=STOPWORDS,
            ngram_range=(1, 2),
            max_features=500
        )
        tfidf = cv.fit_transform(docs)
        cos = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        cos_score = max(0, min(100, round(cos * 100)))
    except Exception:
        cos_score = 0

    # --- B) Keyword coverage part ---
    coverage_score = None
    if matched_keywords is not None and missing_keywords is not None:
        total = len(matched_keywords) + len(missing_keywords)
        if total > 0:
            coverage_score = round(100 * len(matched_keywords) / total)

    # --- C) Final blended score ---
    if coverage_score is not None:
        # Weight coverage slightly more than cosine
        return round(0.4 * cos_score + 0.6 * coverage_score)

    return cos_score

class HelloAnalyze(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        return Response({"msg": "analyzer working"})


def extract_jd_keywords(resume_text: str, job_desc: str, top_n: int = 20):
    """
    Extract top-N meaningful JD keywords using simple frequency
    (no ML) and split them into matched / missing based on resume text.
    This is intentionally simple and robust.
    """
    if not job_desc or not resume_text:
        return [], []

    try:
        # 1. Tokenize JD into words
        jd_tokens = re.findall(r'\b[a-z0-9]+\b', job_desc.lower())

        # 2. Filter out stopwords and very short tokens
        filtered = [
            t for t in jd_tokens
            if len(t) >= 3 and t not in STOPWORDS
        ]

        if not filtered:
            print("extract_jd_keywords: no filtered JD tokens, returning empty.")
            return [], []

        # 3. Count frequency and take top-N
        freq = Counter(filtered)
        jd_keywords = [w for w, _ in freq.most_common(top_n)]

        # 4. Tokenize resume
        resume_tokens = set(re.findall(r'\b[a-z0-9]+\b', resume_text.lower()))

        # 5. Split into matched / missing
        matched = [w for w in jd_keywords if w in resume_tokens]
        missing = [w for w in jd_keywords if w not in resume_tokens]

        print("JD KEYWORDS:", jd_keywords)
        print("MATCHED:", matched)
        print("MISSING:", missing)

        return matched, missing

    except Exception as e:
        print("extract_jd_keywords ERROR:", repr(e))
        return [], []

def compute_ats_readiness(text: str):
    """
    Analyzes resume content for ATS friendliness when no JD is provided.
    Returns a score (0-100) and breakdown metrics.
    """
    if not text:
        return 0, {}, 0, 0, 0

    # --- 1. Section Analysis (Max 40 pts) ---
    # Enhanced mapping to catch different header styles
    SECTION_SYNONYMS = {
        "experience": ["experience", "work experience", "professional experience", "employment", "work history", "internships"],
        "education": ["education", "academic background", "academic qualifications", "academics"],
        "skills": ["skills", "technical skills", "technologies", "competencies", "tech stack"],
        "projects": ["projects", "academic projects", "personal projects", "capstone"],
        "summary": ["summary", "objective", "profile", "about me", "professional summary"]
    }

    present_sections = {}
    for section, synonyms in SECTION_SYNONYMS.items():
        # Check if ANY of the synonyms exist in the text
        # We use strict keyword check (syn in text)
        present_sections[section] = any(syn in text for syn in synonyms)
    
    # 8 points per section found
    section_score = sum(8 for present in present_sections.values() if present)

    # --- 2. Keyword/Action Verb Density (Max 20 pts) ---
    keywords = [
        "developed", "managed", "created", "led", "analyzed", "designed", 
        "python", "java", "c++", "sql", "javascript", "aws", "docker", 
        "communication", "team", "project"
    ]
    
    # Count occurrences of these words
    found_count = sum(1 for word in keywords if word in text)
    # Cap score at 20 (roughly 1 point per keyword hit, max 20 hits)
    keyword_score = min(20, found_count)

    # --- 3. Readability (Max 30 pts) ---
    # Heuristic: Good length (not too short, not too long) and uses punctuation
    word_count = len(text.split())
    
    readability_score = 0
    if 200 < word_count < 2000: # Reasonable resume length
        readability_score += 15
    
    # Sentence/Bullet check (counting periods and hyphens)
    sentence_count = text.count('.') + text.count('-')
    if sentence_count > 10: # Has structure
        readability_score += 15

    # --- 4. Formatting (Max 10 pts) ---
    # Since we cleaned special chars, we check for preserved punctuation typically used in lists
    formatting_score = 0
    if "-" in text or "." in text: # Implies lists or sentences exist
        formatting_score = 10

    # --- Total Score ---
    total_score = section_score + keyword_score + readability_score + formatting_score
    return total_score, present_sections, readability_score, keyword_score, formatting_score


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
    raw_jd = request.data.get('job_description', '')
    job_description = raw_jd.lower().strip()
    jd_present = bool(job_description)

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
    cleaned_text = preprocess_text(extracted_text)
    match_score = 0
    matched_keywords = []
    missing_keywords = []

    ats_score = 0
    section_analysis = {}
    readability_score = 0
    keyword_density_score = 0
    formatting_score = 0

    if jd_present:
        matched_keywords, missing_keywords = extract_jd_keywords(cleaned_text, job_description)
        print("FINAL MATCHED:", matched_keywords)
        print("FINAL MISSING:", missing_keywords)

        match_score = calculate_match_score(
            cleaned_text,
            job_description,
            matched_keywords=matched_keywords,
            missing_keywords=missing_keywords,
        )

        return Response({
            "jd_present": True,
            "extracted_text": cleaned_text,
            "match_score": match_score,
            "matched_keywords": matched_keywords,
            "missing_keywords": missing_keywords
        }, status=status.HTTP_200_OK)
    
    else:
        # --- PATH B: ATS Readiness Mode (No JD) ---
        ats_stats = compute_ats_readiness(cleaned_text)
        # Unpack the results
        ats_score = ats_stats[0]
        section_analysis = ats_stats[1]
        readability_score = ats_stats[2]
        keyword_density_score = ats_stats[3]
        formatting_score = ats_stats[4]

        return Response({
            "jd_present": False,
            "extracted_text": cleaned_text,
            "ats_score": ats_score,
            "section_analysis": section_analysis,
            "readability_score": readability_score,
            "keyword_density_score": keyword_density_score,
            "formatting_score": formatting_score
        }, status=status.HTTP_200_OK)
    
@csrf_exempt
@permission_classes([AllowAny])
@api_view(["POST"])
def ai_suggest(request):
    """
    Calls Bytez chat-completions API (OpenAI-compatible) using
    Qwen/Qwen3-4B-Instruct-2507 to give resume suggestions.
    """

    # LOCAL ONLY: hard-code key + endpoint.
    # DO NOT COMMIT REAL KEY TO GITHUB.
    api_key = os.getenv("BYTEZ_API_KEY")
    base_url = os.getenv("BYTEZ_BASE_URL")
    model_id = os.getenv("BYTEZ_MODEL_ID")
    

    if not api_key or not base_url or not model_id:
        return Response(
            {
                "error": "AI suggestions are not configured on the server.",
                "debug": {
                    "has_key": bool(api_key),
                    "has_url": bool(base_url),
                    "has_model": bool(model_id),
                },
            },
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    resume_text = (request.data.get("resume_text") or "").strip()
    job_description = (request.data.get("job_description") or "").strip()

    if not resume_text:
        return Response(
            {"error": "resume_text is required."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    resume_text_short = resume_text[:4000]
    jd_short = job_description[:2000]

    jd_block = jd_short if jd_short else "No job description was provided."

    user_prompt = f"""
        You are an AI assistant helping a student improve their resume for a job.

        RESUME:
        {resume_text_short}

        JOB DESCRIPTION:
        {jd_block}

        Write your answer ONLY in this format:

        Match summary:
        - ...

        Missing or weak skills / keywords:
        - ...

        Section-wise improvements:
        - Summary: ...
        - Skills: ...
        - Projects: ...
        - Experience: ...

        Example improved bullet points:
        - ...

        Rules:
        - Use clear bullet points.
        - Do NOT repeat the resume text.
        - Do NOT repeat these instructions.
        - Keep everything under 200 words.
    """.strip()

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise, practical resume coach for students.",
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "max_tokens": 300,
        "temperature": 0.4,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(base_url, json=payload, headers=headers, timeout=120)

        if resp.status_code != 200:
            # Special case: Bytez free plan rate-limit / concurrency error
            if resp.status_code == 429:
                return Response(
                    {
                        "error": "AI service is busy right now (free plan rate limit). Please wait 30–60 seconds and try again.",
                        "status_code": 429,
                    },
                    status=status.HTTP_503_SERVICE_UNAVAILABLE,
                )

            # Other HTTP errors
            return Response(
                {
                    "error": "AI API error",
                    "status_code": resp.status_code,
                    "body": resp.text[:500],
                },
                status=status.HTTP_502_BAD_GATEWAY,
            )

        data = resp.json()

        suggestions_text = ""
        if isinstance(data, dict):
            choices = data.get("choices") or []
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message") or {}
                suggestions_text = (msg.get("content") or "").strip()

        if not suggestions_text:
            suggestions_text = resp.text

        return Response(
            {"ok": True, "suggestions_text": suggestions_text},
            status=status.HTTP_200_OK,
        )

    except requests.exceptions.Timeout:
        return Response(
            {
                "error": "AI service took too long to respond (timeout). This is a limitation of the free API. Please try again after some time."
            },
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    except Exception as e:
        return Response(
            {"error": f"AI request failed: {str(e)}"},
            status=status.HTTP_502_BAD_GATEWAY,
        )
