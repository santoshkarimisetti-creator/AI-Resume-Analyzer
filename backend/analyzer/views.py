from django.shortcuts import render

import requests
from django.conf import settings
import os
import tempfile
import re
import pdfplumber

# --- Sentence-Transformers and NLTK Imports (Lazy Load) ---
try:
    # Sentence-transformers (for semantic similarity)
    from sentence_transformers import SentenceTransformer, util as st_util
    import torch # Required by sentence-transformers
    _ST_AVAILABLE = True
except Exception:
    # Fallback if sentence-transformers or torch is not installed
    _ST_AVAILABLE = False

# We include nltk here as requested, though it's not strictly necessary for the core logic below
try:
    import nltk 
    _NLTK_AVAILABLE = True
except Exception:
    _NLTK_AVAILABLE = False
# --- END Imports ---

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


# --- ADDED: synonyms map (expand as you need) ---
# This is used by extract_jd_keywords for improved coverage checks.
SYNONYMS = {
    # core role mappings
    "developer": ["developer", "dev", "engineer", "programmer", "software engineer", "web developer", "fullstack developer", "full-stack developer"],
    "web development": ["web development", "web-develop", "frontend", "front-end", "front end", "front-end development", "front end development", "full stack", "full-stack", "fullstack"],
    "intern": ["intern", "internship", "trainee"],
    "javascript": ["javascript", "js"],
    "react": ["react", "reactjs", "react.js"],
    "node": ["node", "nodejs", "node.js"],
    "machine learning": ["machine learning", "ml", "ai"],
    "data science": ["data science", "datascience", "data scientist"],
    "sql": ["sql", "postgres", "postgresql", "mysql", "t-sql", "ssrs"],
    "python": ["python", "py"],
    "html": ["html"],
    "css": ["css", "scss", "less"],
    "rest": ["rest", "restful", "rest api"],
    "cloud": ["aws", "azure", "gcp", "cloud", "amazon web services"],
    # add more mappings as you encounter missing matches
}

# helper: build flattened set or variants if needed
def _variants_for(keyword: str) -> list[str]:
    """Returns a list containing the keyword itself and all its defined synonyms."""
    # Ensure all variants are lowercased and spaces are used for multi-word phrases
    keyword_norm = keyword.lower().strip()
    return SYNONYMS.get(keyword_norm, [keyword_norm])


# --- Stopwords List (Focus on general English and HR filler) ---
EXTRA_STOPWORDS = {
    "seeking", "performed", # Original crucial fixes
    
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

# --- NEW: Low-Value Terms for Keyword Filtering ---
# These words are typically high-frequency in JD but offer low value for matching technical skills.
# They are removed from the final list of *required* keywords in extract_jd_keywords.
LOW_VALUE_JD_TERMS = {
    # Words reported by user to be problematic:
    "storing", "retrieving", "members", "quickly", 
    "motivation", "continuously", "aligning", "coding", "required", 
    "attitude", "application",
    
    # General Soft Skills / Generic Terms to filter out:
    "growth", "impact", "innovate", "value", "deliver", "delivery", 
    "learn", "learning", "challenge", "challenging", "diverse", "diversity",
    "process", "processes", "method", "methods", "technique", "techniques",
    "systematically", "successfully", "effectively", "efficiently", "seamlessly",
    "customer", "customers", "user", "users", "stakeholder", "stakeholders",
    "client", "clients", "data", # 'data' is too common unless it's a specific phrase like 'data science'
    "information", "business", "company", "technology", "environment",
    "problem", "solving", "critical", "thinking", "adapt", "adaptable",
    "flexible", "flexibility", "resource", "resources", "internal", "external"
}


from PyPDF2 import PdfReader
from docx import Document

MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXT = ('.pdf', '.docx')

# --- ATS Preprocessing Helper (Less aggressive for structural checks) ---
def preprocess_for_ats_readability(text: str) -> str:
    """Preserves punctuation for structural checks."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s\.\,\-\/]', ' ', text, re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Text Preprocessing Helper Function (Aggressive for matching/similarity) ---
def preprocess_text(text: str) -> str:
    """Aggressively cleans text for vectorizers/keyword matching."""
    if not text:
        return ""
    text = text.lower()
    # Remove all punctuation and special characters, leaving only alpha-numeric/space
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Collapse multiple spaces into one, and strip
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Lazy Embedding Model Loader ---
_EMBED_MODEL = None
def _get_embed_model():
    """Loads the sentence transformer model once, only if available."""
    global _EMBED_MODEL, _ST_AVAILABLE
    if not _ST_AVAILABLE:
        return None
    if _EMBED_MODEL is None:
        try:
            # Load small, fast model for semantic similarity
            _EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
            print("SentenceTransformer model loaded successfully.")
        except Exception as e:
            print("Failed to load sentence-transformers model (check model files):", e)
            _EMBED_MODEL = None
            _ST_AVAILABLE = False # Disable future attempts if it fails
    return _EMBED_MODEL


def calculate_match_score(
        resume_text: str,
        job_desc: str,
        matched_keywords=None,
        missing_keywords=None
    ) -> int:
    """
    Combines semantic (embedding) or TF-IDF cosine similarity and keyword coverage
    into a 0-100 match score. Uses embeddings if available; otherwise falls back.
    """
    if not resume_text or not job_desc:
        return 0

    # --- A) Semantic / Cosine similarity part (Weight: 40% or 60%) ---
    cos_score = 0
    model_present = False
    try:
        model = _get_embed_model()
        if model is not None:
            # Use sentence-transformers semantic similarity
            model_present = True
            job_emb = model.encode(job_desc, convert_to_tensor=True)
            res_emb = model.encode(resume_text, convert_to_tensor=True)
            # Calculate similarity and convert to float item
            cos = st_util.cos_sim(job_emb, res_emb).item()
            cos_score = max(0, min(100, round(cos * 100)))
            # print(f"Using Semantic Similarity. Score: {cos_score}")
        else:
            # Fallback: existing TF-IDF approach (less precise, purely lexical)
            docs = [job_desc, resume_text]
            cv = TfidfVectorizer(
                stop_words=STOPWORDS,
                ngram_range=(1, 2), # Using n-grams for better TFIDF
                max_features=500
            )
            tfidf = cv.fit_transform(docs)
            cos = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            cos_score = max(0, min(100, round(cos * 100)))
            # print(f"Using TF-IDF Fallback. Score: {cos_score}")
    except Exception as e:
        print("calculate_match_score (similarity part) error:", e)
        cos_score = 0

    # --- B) Keyword coverage part (Weight: 60% or 40%) ---
    coverage_score = 0
    if matched_keywords is not None and missing_keywords is not None:
        total = len(matched_keywords) + len(missing_keywords)
        if total > 0:
            coverage_score = round(100 * len(matched_keywords) / total)
            # print(f"Keyword Coverage Score: {coverage_score}")

    # --- C) Final blended score ---
    if model_present:
        # Semantic Match is high quality (60%) + Keyword Coverage (40%)
        final_score = round(0.6 * cos_score + 0.4 * coverage_score)
    else:
        # TF-IDF Fallback is less reliable, rely more on specific keywords (40% TFIDF + 60% Coverage)
        final_score = round(0.4 * cos_score + 0.6 * coverage_score)
        
    return max(0, min(100, final_score))

class HelloAnalyze(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        return Response({"msg": "analyzer working"})


def extract_jd_keywords(resume_text: str, job_desc: str, top_n: int = 60):
    """
    Extract top-N meaningful JD keywords using simple frequency, 
    filter out low-value terms using LOW_VALUE_JD_TERMS, 
    and use SYNONYMS/substring matching for the check.
    """
    if not job_desc or not resume_text:
        return [], []

    try:
        # 1. Tokenize JD and filter out general stopwords
        jd_tokens = re.findall(r'\b[a-z0-9]+\b', job_desc.lower())

        filtered = [
            t for t in jd_tokens
            if len(t) >= 3 and t not in STOPWORDS
        ]

        if not filtered:
            print("extract_jd_keywords: no filtered JD tokens, returning empty.")
            return [], []

        # 2. Count frequency and take top-N candidates
        freq = Counter(filtered)
        top_candidates = [w for w, _ in freq.most_common(top_n)]
        
        # 3. Filter out low-value/soft-skill terms from the candidates
        # This handles the generic jargon that shouldn't contribute to the score.
        jd_keywords = [w for w in top_candidates if w not in LOW_VALUE_JD_TERMS]

        # 4. Prepare resume text for smarter matching
        resume_tokens = set(re.findall(r'\b[a-z0-9]+\b', resume_text.lower()))
        resume_text_norm = preprocess_for_ats_readability(resume_text)

        # --- Smarter matching using SYNONYMS and multi-word substring checks ---
        matched = []
        missing = []

        def token_matches(kw):
            """
            Return True if any variant of kw is found in either token set
            or as a substring in the resume text (handles multi-word synonyms).
            """
            variants = _variants_for(kw)
            for v in variants:
                # 1. Exact token match (fastest check)
                if v in resume_tokens:
                    return True
                
                # 2. Multi-word / phrase match (substring in structural text)
                if " " in v and v in resume_text_norm:
                    return True
                
                # 3. Hyphen/period/slash variants (e.g., 'full-stack' vs 'full stack' vs 'full/stack')
                # Check normalized variants for a match in the structural text
                v_spaces = v.replace("-", " ").replace(".", " ").replace("/", " ")
                if v_spaces in resume_text_norm:
                    return True

            return False

        for w in jd_keywords:
            if token_matches(w):
                matched.append(w)
            else:
                missing.append(w)

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
    Scoring weights: Sections (35), Keywords (35), Readability (20), Formatting (10)
    """
    if not text:
        return 0, {}, 0, 0, 0

    # Use the less aggressive preprocessor for structural checks
    ats_text = preprocess_for_ats_readability(text)

    # --- 1. Section Analysis (Max 35 pts) ---
    SECTION_SYNONYMS = {
        "experience": ["experience", "work experience", "professional experience", "employment", "work history", "internships"],
        "education": ["education", "academic background", "academics"],
        "skills": ["skills", "technical skills", "technologies", "competencies", "tech stack"],
        "projects": ["projects", "academic projects", "personal projects", "capstone"],
        "summary": ["summary", "objective", "profile", "about me", "professional summary"]
    }

    present_sections = {}
    
    # Check for presence of key sections
    for section, synonyms in SECTION_SYNONYMS.items():
        # Check if ANY of the synonyms exist in the text
        present_sections[section] = any(syn in ats_text for syn in synonyms)
    
    # 7 points per section found
    section_score = sum(7 for present in present_sections.values() if present)

    # --- 2. Keyword/Action Verb Density (Max 35 pts) ---
    # Expanded list of high-value verbs and technical terms
    keywords = [
        "developed", "managed", "created", "led", "analyzed", "designed", 
        "implemented", "engineered", "optimized", "streamlined", "deployed",
        "python", "java", "c++", "sql", "javascript", "react", "angular", "node",
        "aws", "azure", "gcp", "docker", "kubernetes", "agile", "scrum",
        "communication", "collaboration", "teamwork", "leadership", "mentored"
    ]
    
    # Count occurrences of these words
    found_count = sum(1 for word in keywords if word in ats_text.split())
    
    # Granular scoring: 1 point per keyword hit, capped at 35
    keyword_score = min(35, found_count)

    # --- 3. Readability & Length (Max 20 pts) ---
    word_count = len(text.split())
    readability_score = 0
    
    # Reasonable length score (10 pts)
    if 250 < word_count < 1500: 
        readability_score += 10
    
    # Structural score (10 pts) - checks for periods, commas, or hyphens 
    # indicating bullet points/sentences in the original text.
    structure_indicators = ats_text.count('.') + ats_text.count(',') + ats_text.count('-')
    if structure_indicators > 20: 
        readability_score += 10

    # --- 4. Formatting (Max 10 pts) ---
    # Simple heuristic: Check for a healthy density of structural characters 
    # that usually imply lists/tables were present in the source.
    punctuation_density = (ats_text.count('-') + ats_text.count(':')) / (word_count or 1)
    formatting_score = 0
    # A density between 0.01 and 0.05 is generally good for structured resumes
    if 0.01 < punctuation_density < 0.05:
         formatting_score = 10
    elif punctuation_density > 0.05:
         # Punctuations are too dense, might be corrupted text, give partial credit
         formatting_score = 5
    
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
    cleaned = re.sub(r'[ \t\r\f]+', ' ', full).strip()
    return cleaned


def extract_text_from_docx(path: str) -> str:
    try:
        doc = Document(path)
        paragraphs = [p.text for p in doc.paragraphs if p.text]
        full = "\n".join(paragraphs).strip()
    except Exception:
        return ""
    cleaned = re.sub(r'[ \t\r\f]+', ' ', full).strip()
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
    
    # Use the aggressive preprocess_text for matching/similarity
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
        # PATH A: JD Match Mode
        matched_keywords, missing_keywords = extract_jd_keywords(cleaned_text, job_description)

        # calculate_match_score handles the semantic matching fallback internally
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
        # compute_ats_readiness uses the original extracted text for structural analysis
        ats_stats = compute_ats_readiness(extracted_text)
        
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