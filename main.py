"""
TriageAI — Mass Casualty & Disaster Triage System
Flask backend with multimodal Gemini API integration.
Converts messy inputs (text, images, audio) into structured START triage data.
"""

import os
import json
import uuid
import time
import tempfile
import logging
from datetime import datetime, timezone
from functools import wraps
from typing import List, Literal, Optional

from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory, abort
from pydantic import BaseModel, Field, ValidationError
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    logging.warning("GEMINI_API_KEY not set — API calls will fail.")

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
MAX_TEXT_LENGTH = 5000
MAX_FILE_SIZE_MB = 10
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
ALLOWED_AUDIO_TYPES = {"audio/webm", "audio/ogg", "audio/wav", "audio/mp3",
                       "audio/mpeg", "audio/mp4", "audio/x-m4a"}
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 30     # requests per window

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic Schemas (START Triage Protocol)
# ---------------------------------------------------------------------------

class TriageActionPlan(BaseModel):
    """Structured triage output following the START protocol."""
    incident_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8],
                             description="Unique incident identifier")
    urgency: Literal["RED", "YELLOW", "GREEN", "BLACK"] = Field(
        description="START triage category: RED=Immediate, YELLOW=Delayed, GREEN=Minor, BLACK=Deceased/Expectant"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model confidence in the triage classification (0.0–1.0)"
    )
    patient_count: int = Field(
        ge=0,
        description="Estimated number of patients described"
    )
    hazard_alerts: List[str] = Field(
        description="Environmental hazards detected (fire, gas, chemicals, structural collapse, etc.)"
    )
    patient_vitals_summary: str = Field(
        description="Summary of patient conditions including breathing, consciousness, mobility"
    )
    critical_intervention: str = Field(
        description="Most urgent action first responders should take"
    )
    recommended_resources: List[str] = Field(
        description="Resources needed (ambulance, hazmat, fire, police, helicopter, etc.)"
    )
    location_info: str = Field(
        description="Extracted location information from the input"
    )
    dispatch_code: str = Field(
        description="Standardized dispatch priority code"
    )
    language_detected: str = Field(
        default="en",
        description="ISO 639-1 language code of the original input"
    )
    raw_input_summary: str = Field(
        description="Brief English summary of the original input regardless of source language"
    )


class TriageRecord(BaseModel):
    """A triage record stored in the incident history."""
    id: str
    timestamp: str
    input_type: str
    input_preview: str
    result: dict


# ---------------------------------------------------------------------------
# START Triage System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are TriageAI, an advanced emergency triage system built on the START (Simple Triage and Rapid Treatment) protocol. Your role is to convert chaotic, messy, multilingual emergency inputs into precise structured triage data that saves lives.

## START Triage Classification Rules:

**RED (Immediate / Priority 1):**
- Life-threatening injuries with high survival potential if treated NOW
- Respiratory rate > 30/min OR absent radial pulse OR cannot follow commands
- NOT breathing but starts after airway repositioning

**YELLOW (Delayed / Priority 2):**
- Serious injuries but NOT immediately life-threatening
- Stable respiration and perfusion but cannot walk
- Can follow simple commands

**GREEN (Walking Wounded / Priority 3):**
- Minor injuries, ambulatory
- Can walk, talk, and follow instructions

**BLACK (Deceased / Expectant / Priority 0):**
- Not breathing even after airway repositioning
- Injuries incompatible with survival given current resources

## Your Analysis Process:
1. DETECT the language of input and translate internally
2. EXTRACT all patient information, counts, conditions
3. IDENTIFY environmental hazards (fire, gas, chemicals, structural collapse, water, electrical)
4. CLASSIFY using START protocol — when multiple patients, classify based on the MOST critical
5. RECOMMEND specific resources and interventions
6. ASSESS confidence in your classification (lower if input is vague/contradictory)

## Critical Rules:
- When in doubt between two categories, ALWAYS choose the more urgent one
- Multi-patient scenarios: triage based on the most critical patient
- Always extract location even if partial
- Treat IoT/sensor data (temps, alarms) as legitimate input
- If input describes hiding/trapped persons, factor in psychological state
- Generate a clear dispatch code (e.g., "MCI-R-3" = Mass Casualty Incident, Red priority, 3 patients)"""


# ---------------------------------------------------------------------------
# Gemini Client
# ---------------------------------------------------------------------------

def get_client() -> genai.Client:
    """Create and return a Gemini API client."""
    return genai.Client(api_key=GEMINI_API_KEY)


def process_text_input(text: str) -> dict:
    """Process text input through Gemini and return structured triage data."""
    client = get_client()
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=text,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "response_mime_type": "application/json",
                "response_schema": TriageActionPlan,
            }
        )
        parsed = response.parsed
        if parsed:
            return parsed.model_dump()
        # Fallback: try parsing text as JSON
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"Gemini text processing error: {e}")
        raise


def process_file_input(file_path: str, mime_type: str, context: str = "") -> dict:
    """Process an uploaded file (image/audio) through Gemini multimodal API."""
    client = get_client()
    try:
        # Upload file to Gemini Files API
        uploaded_file = client.files.upload(path=file_path)
        
        prompt_parts = []
        if context:
            prompt_parts.append(f"Additional context from the reporter: {context}")
        
        if mime_type.startswith("image/"):
            prompt_parts.append(
                "Analyze this emergency scene image. Identify victims, hazards, "
                "and environmental conditions. Perform START triage classification."
            )
        elif mime_type.startswith("audio/"):
            prompt_parts.append(
                "Transcribe and analyze this emergency audio recording. "
                "Extract patient information, hazards, and location. "
                "Perform START triage classification."
            )
        
        prompt_parts.append(uploaded_file)
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt_parts,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "response_mime_type": "application/json",
                "response_schema": TriageActionPlan,
            }
        )
        parsed = response.parsed
        if parsed:
            return parsed.model_dump()
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"Gemini file processing error: {e}")
        raise


# ---------------------------------------------------------------------------
# Flask Application
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder="static", static_url_path="")

# In-memory incident history (suitable for hackathon demo)
incident_history: list[dict] = []

# Simple rate limiter
rate_limit_store: dict[str, list[float]] = {}


# -- Security middleware --
@app.after_request
def add_security_headers(response):
    """Add security headers to every response."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), geolocation=(self)"
    return response


def rate_limit(f):
    """Simple in-memory rate limiter decorator."""
    @wraps(f)
    def decorated(*args, **kwargs):
        client_ip = request.remote_addr or "unknown"
        now = time.time()
        
        if client_ip not in rate_limit_store:
            rate_limit_store[client_ip] = []
        
        # Clean old entries
        rate_limit_store[client_ip] = [
            t for t in rate_limit_store[client_ip]
            if now - t < RATE_LIMIT_WINDOW
        ]
        
        if len(rate_limit_store[client_ip]) >= RATE_LIMIT_MAX:
            return jsonify({
                "error": "Rate limit exceeded",
                "message": f"Maximum {RATE_LIMIT_MAX} requests per {RATE_LIMIT_WINDOW}s"
            }), 429
        
        rate_limit_store[client_ip].append(now)
        return f(*args, **kwargs)
    return decorated


def validate_text_input(text: Optional[str]) -> tuple[Optional[str], Optional[tuple]]:
    """Validate text input. Returns (clean_text, error_response) tuple."""
    if not text or not text.strip():
        return None, (jsonify({
            "error": "Empty input",
            "message": "Please provide emergency scenario text."
        }), 400)
    
    text = text.strip()
    if len(text) > MAX_TEXT_LENGTH:
        return None, (jsonify({
            "error": "Input too long",
            "message": f"Text must be under {MAX_TEXT_LENGTH} characters."
        }), 400)
    
    return text, None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def serve_index():
    """Serve the main application page."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/health")
def health_check():
    """Health check endpoint for Cloud Run."""
    return jsonify({
        "status": "healthy",
        "service": "TriageAI",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


@app.route("/api/triage", methods=["POST"])
@rate_limit
def triage_text():
    """Process text-based emergency input.
    
    Expects JSON body: { "text": "emergency description..." }
    Returns structured triage data.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON", "message": "Request body must be valid JSON."}), 400
    
    text, error = validate_text_input(data.get("text"))
    if error:
        return error
    
    try:
        result = process_text_input(text)
        
        # Store in history
        record = {
            "id": result.get("incident_id", str(uuid.uuid4())[:8]),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_type": "text",
            "input_preview": text[:100] + ("..." if len(text) > 100 else ""),
            "result": result
        }
        incident_history.insert(0, record)
        
        logger.info(f"Triage completed: {result.get('urgency')} | {result.get('dispatch_code')}")
        return jsonify({"success": True, "data": result})
    
    except Exception as e:
        logger.error(f"Triage text error: {e}")
        return jsonify({
            "error": "Processing failed",
            "message": "Failed to process the emergency input. Please try again."
        }), 500


@app.route("/api/triage/image", methods=["POST"])
@rate_limit
def triage_image():
    """Process image-based emergency input.
    
    Expects multipart form data with 'image' file and optional 'context' text.
    Returns structured triage data.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image", "message": "Please upload an image file."}), 400
    
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Empty file", "message": "The uploaded file is empty."}), 400
    
    # Validate MIME type
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        return jsonify({
            "error": "Invalid file type",
            "message": f"Allowed image types: {', '.join(ALLOWED_IMAGE_TYPES)}"
        }), 400
    
    # Validate file size
    file.seek(0, 2)
    size_mb = file.tell() / (1024 * 1024)
    file.seek(0)
    if size_mb > MAX_FILE_SIZE_MB:
        return jsonify({
            "error": "File too large",
            "message": f"Maximum file size is {MAX_FILE_SIZE_MB}MB."
        }), 400
    
    context = request.form.get("context", "")
    
    try:
        # Save to temp file for Gemini processing
        suffix = os.path.splitext(file.filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            result = process_file_input(tmp_path, file.content_type, context)
        finally:
            os.unlink(tmp_path)
        
        record = {
            "id": result.get("incident_id", str(uuid.uuid4())[:8]),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_type": "image",
            "input_preview": f"📷 {file.filename}" + (f" — {context[:60]}" if context else ""),
            "result": result
        }
        incident_history.insert(0, record)
        
        logger.info(f"Image triage: {result.get('urgency')} | {file.filename}")
        return jsonify({"success": True, "data": result})
    
    except Exception as e:
        logger.error(f"Image triage error: {e}")
        return jsonify({
            "error": "Processing failed",
            "message": "Failed to analyze the image. Please try again."
        }), 500


@app.route("/api/triage/audio", methods=["POST"])
@rate_limit
def triage_audio():
    """Process audio-based emergency input.
    
    Expects multipart form data with 'audio' file and optional 'context' text.
    Returns structured triage data.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio", "message": "Please upload an audio file."}), 400
    
    file = request.files["audio"]
    
    # Validate MIME type
    content_type = file.content_type or "audio/webm"
    if content_type not in ALLOWED_AUDIO_TYPES:
        return jsonify({
            "error": "Invalid file type",
            "message": f"Allowed audio types: {', '.join(ALLOWED_AUDIO_TYPES)}"
        }), 400
    
    # Validate file size
    file.seek(0, 2)
    size_mb = file.tell() / (1024 * 1024)
    file.seek(0)
    if size_mb > MAX_FILE_SIZE_MB:
        return jsonify({
            "error": "File too large",
            "message": f"Maximum file size is {MAX_FILE_SIZE_MB}MB."
        }), 400
    
    context = request.form.get("context", "")
    
    try:
        suffix = os.path.splitext(file.filename or "recording")[1] or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            result = process_file_input(tmp_path, content_type, context)
        finally:
            os.unlink(tmp_path)
        
        record = {
            "id": result.get("incident_id", str(uuid.uuid4())[:8]),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input_type": "audio",
            "input_preview": f"🎤 Audio recording" + (f" — {context[:60]}" if context else ""),
            "result": result
        }
        incident_history.insert(0, record)
        
        logger.info(f"Audio triage: {result.get('urgency')}")
        return jsonify({"success": True, "data": result})
    
    except Exception as e:
        logger.error(f"Audio triage error: {e}")
        return jsonify({
            "error": "Processing failed",
            "message": "Failed to analyze the audio. Please try again."
        }), 500


@app.route("/api/history", methods=["GET"])
def get_history():
    """Return the incident triage history."""
    return jsonify({
        "success": True,
        "data": incident_history,
        "count": len(incident_history)
    })


@app.route("/api/history", methods=["DELETE"])
def clear_history():
    """Clear the incident triage history."""
    incident_history.clear()
    return jsonify({"success": True, "message": "History cleared."})


# ---------------------------------------------------------------------------
# CLI Test Suite (preserved from original)
# ---------------------------------------------------------------------------

def run_cli_tests():
    """Run the original CLI test suite for development/CI."""
    print("=" * 60)
    print("  TriageAI — CLI Test Suite")
    print("=" * 60)
    
    test_input = (
        "Subway crash at 42nd st. Smoke everywhere. "
        "One victim bleeding from arm, unconscious. "
        "I see a downed power line."
    )
    print(f"\n--- Processing Triage 1 ---")
    result = process_text_input(test_input)
    print(json.dumps(result, indent=2))

    advanced_tests = [
        "¡Ayuda! Hubo una explosión en la cocina. Mi compañero no puede respirar por el humo.",
        "ALARM_ID: 992. TEMP: 105C. SMOKE: POSITIVE. 2 technicians trapped in server room.",
        "Car sinking in river, but another car on the bridge is on fire and about to explode!",
        "Can't talk. Hiding in closet. 3rd floor, Apt 4B. Smelling gas.",
        "Bus flipped. 20 people. Most walking. One woman not breathing. One kid with broken leg.",
    ]

    print(f"\n{'=' * 20} NEXUS BRIDGE STRESS TEST {'=' * 20}")
    for i, test in enumerate(advanced_tests, 1):
        print(f"\nTEST CASE #{i}: {test[:50]}...")
        result = process_text_input(test)
        print(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    
    if "--test" in sys.argv:
        run_cli_tests()
    else:
        port = int(os.environ.get("PORT", 8080))
        logger.info(f"Starting TriageAI on port {port}")
        app.run(host="0.0.0.0", port=port, debug=True)