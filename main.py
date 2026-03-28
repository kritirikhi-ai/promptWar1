"""
TriageAI — Mass Casualty & Disaster Triage System

A Gemini-powered application that converts chaotic, unstructured emergency
inputs (text reports, images, audio recordings) into structured, verified,
life-saving triage actions using the START protocol.

Google Cloud Services Integrated:
    - Gemini API: Multimodal AI for input analysis
    - Cloud Run: Serverless container deployment
    - Cloud Logging: Structured operational logging
    - Secret Manager: Secure API key management
    - Firestore: Persistent incident data storage
    - Cloud Storage: Emergency media file archival
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Dict, FrozenSet, List, Literal, Optional, Tuple

from dotenv import load_dotenv
from flask import Flask, Response, g, jsonify, request, send_from_directory
from pydantic import BaseModel, Field, ValidationError
from google import genai
from google.genai import types as genai_types

# ── Google Cloud imports (graceful fallback for local dev) ────────────
_GCP_LOGGING = False
_GCP_SECRETS = False
_GCP_FIRESTORE = False
_GCP_STORAGE = False

try:
    import google.cloud.logging as cloud_logging
    _GCP_LOGGING = True
except ImportError:
    pass

try:
    from google.cloud import secretmanager
    _GCP_SECRETS = True
except ImportError:
    pass

try:
    from google.cloud import firestore as cloud_firestore
    _GCP_FIRESTORE = True
except ImportError:
    pass

try:
    from google.cloud import storage as cloud_storage
    _GCP_STORAGE = True
except ImportError:
    pass

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────
APP_VERSION = "2.0.0"

# ── Logging ───────────────────────────────────────────────────────────
logger = logging.getLogger("triageai")


def _init_logging() -> None:
    """Set up logging: Cloud Logging on GCP, standard otherwise."""
    if _GCP_LOGGING and os.environ.get("K_SERVICE"):
        client = cloud_logging.Client()
        client.setup_logging(log_level=logging.INFO)
        logger.info("Cloud Logging integration active")
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )


_init_logging()


# ── Configuration ─────────────────────────────────────────────────────
@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration with secure defaults."""

    GEMINI_MODEL: str = "gemini-2.5-flash"
    MAX_TEXT_LENGTH: int = 5000
    MAX_FILE_SIZE_MB: int = 10
    RATE_LIMIT_WINDOW: int = 60
    RATE_LIMIT_MAX: int = 30
    FIRESTORE_COLLECTION: str = "triage_incidents"
    GCS_BUCKET: str = ""
    ALLOWED_IMAGE_TYPES: FrozenSet[str] = frozenset(
        {"image/jpeg", "image/png", "image/webp", "image/gif"}
    )
    ALLOWED_AUDIO_TYPES: FrozenSet[str] = frozenset(
        {"audio/webm", "audio/ogg", "audio/wav", "audio/mp3",
         "audio/mpeg", "audio/mp4", "audio/x-m4a"}
    )


CONFIG = AppConfig(
    GCS_BUCKET=os.environ.get("GCS_BUCKET", ""),
)


# ── Custom Exceptions ─────────────────────────────────────────────────
class TriageError(Exception):
    """Base exception for triage processing errors."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class InputValidationError(TriageError):
    """Raised when user input fails validation."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400)


class AIProcessingError(TriageError):
    """Raised when Gemini API processing fails."""

    def __init__(self, message: str = "AI analysis failed.") -> None:
        super().__init__(message, status_code=502)


# ── Pydantic Models ───────────────────────────────────────────────────
class TriageActionPlan(BaseModel):
    """Structured triage output following the START protocol."""

    incident_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique incident identifier",
    )
    urgency: Literal["RED", "YELLOW", "GREEN", "BLACK"] = Field(
        description="START triage: RED=Immediate, YELLOW=Delayed, "
                    "GREEN=Minor, BLACK=Expectant",
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model confidence in classification (0.0–1.0)",
    )
    patient_count: int = Field(ge=0, description="Estimated patient count")
    hazard_alerts: List[str] = Field(description="Environmental hazards")
    patient_vitals_summary: str = Field(description="Patient conditions")
    critical_intervention: str = Field(description="Urgent action needed")
    recommended_resources: List[str] = Field(description="Resources needed")
    location_info: str = Field(description="Extracted location")
    dispatch_code: str = Field(description="Priority dispatch code")
    language_detected: str = Field(default="en", description="ISO 639-1")
    raw_input_summary: str = Field(description="English summary of input")


class TriageRecord(BaseModel):
    """A triage incident record for persistence."""

    id: str
    timestamp: str
    input_type: str
    input_preview: str
    result: Dict[str, Any]


# ── Google Services Layer ─────────────────────────────────────────────
class GoogleServices:
    """Centralised Google Cloud service integrations with fallbacks."""

    def __init__(self) -> None:
        self._gemini_client: Optional[genai.Client] = None
        self._firestore_db: Optional[Any] = None
        self._storage_client: Optional[Any] = None
        self._memory_store: deque[dict] = deque(maxlen=200)
        self._api_key: str = self._resolve_api_key()

    # ── Secret Manager ────────────────────────────────────────────────
    def _resolve_api_key(self) -> str:
        """Retrieve API key: Secret Manager → env var fallback."""
        if _GCP_SECRETS and os.environ.get("GOOGLE_CLOUD_PROJECT"):
            try:
                client = secretmanager.SecretManagerServiceClient()
                project = os.environ["GOOGLE_CLOUD_PROJECT"]
                name = f"projects/{project}/secrets/gemini-api-key/versions/latest"
                resp = client.access_secret_version(request={"name": name})
                logger.info("API key loaded from Secret Manager")
                return resp.payload.data.decode("UTF-8")
            except Exception as exc:
                logger.warning("Secret Manager fallback to env: %s", exc)
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            logger.warning("GEMINI_API_KEY not set — API calls will fail")
        return key

    # ── Gemini Client (singleton) ─────────────────────────────────────
    @property
    def gemini(self) -> genai.Client:
        """Lazy singleton Gemini client for connection reuse."""
        if self._gemini_client is None:
            self._gemini_client = genai.Client(api_key=self._api_key)
        return self._gemini_client

    # ── Firestore ─────────────────────────────────────────────────────
    def _get_firestore(self) -> Optional[Any]:
        """Get or create Firestore client."""
        if self._firestore_db is None and _GCP_FIRESTORE:
            try:
                self._firestore_db = cloud_firestore.Client()
                logger.info("Firestore client initialised")
            except Exception as exc:
                logger.warning("Firestore unavailable: %s", exc)
        return self._firestore_db

    def save_incident(self, record: dict) -> None:
        """Persist incident to Firestore (fallback: in-memory)."""
        self._memory_store.appendleft(record)
        db = self._get_firestore()
        if db is not None:
            try:
                db.collection(CONFIG.FIRESTORE_COLLECTION).document(
                    record["id"]
                ).set(record)
                logger.info("Incident %s saved to Firestore", record["id"])
            except Exception as exc:
                logger.error("Firestore write failed: %s", exc)

    def get_incidents(self, limit: int = 50) -> List[dict]:
        """Read incidents: Firestore → in-memory fallback."""
        db = self._get_firestore()
        if db is not None:
            try:
                docs = (
                    db.collection(CONFIG.FIRESTORE_COLLECTION)
                    .order_by("timestamp", direction=cloud_firestore.Query.DESCENDING)
                    .limit(limit)
                    .stream()
                )
                return [doc.to_dict() for doc in docs]
            except Exception as exc:
                logger.warning("Firestore read failed: %s", exc)
        return list(self._memory_store)

    def clear_incidents(self) -> None:
        """Delete all incidents from Firestore and memory."""
        self._memory_store.clear()
        db = self._get_firestore()
        if db is not None:
            try:
                docs = db.collection(CONFIG.FIRESTORE_COLLECTION).limit(500).stream()
                for doc in docs:
                    doc.reference.delete()
                logger.info("Firestore incidents cleared")
            except Exception as exc:
                logger.error("Firestore clear failed: %s", exc)

    # ── Cloud Storage ─────────────────────────────────────────────────
    def _get_storage(self) -> Optional[Any]:
        """Get or create Cloud Storage client."""
        if self._storage_client is None and _GCP_STORAGE:
            try:
                self._storage_client = cloud_storage.Client()
            except Exception:
                pass
        return self._storage_client

    def upload_to_gcs(self, local_path: str, filename: str) -> Optional[str]:
        """Archive uploaded file to Cloud Storage. Returns GCS URI."""
        if not CONFIG.GCS_BUCKET:
            return None
        client = self._get_storage()
        if client is None:
            return None
        try:
            bucket = client.bucket(CONFIG.GCS_BUCKET)
            blob_name = f"uploads/{datetime.now(timezone.utc).strftime('%Y%m%d')}/{filename}"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_path)
            uri = f"gs://{CONFIG.GCS_BUCKET}/{blob_name}"
            logger.info("File archived to %s", uri)
            return uri
        except Exception as exc:
            logger.warning("GCS upload failed: %s", exc)
            return None


# Singleton services instance
services = GoogleServices()


# ── START Triage System Prompt ────────────────────────────────────────
SYSTEM_PROMPT = """You are TriageAI, an advanced emergency triage system built on the START (Simple Triage and Rapid Treatment) protocol.

## START Triage Classification Rules:
**RED (Immediate):** Life-threatening, high survival if treated NOW. Resp rate >30, no radial pulse, or cannot follow commands.
**YELLOW (Delayed):** Serious but stable. Can follow commands but cannot walk.
**GREEN (Minor):** Walking wounded. Ambulatory, minor injuries.
**BLACK (Expectant):** Not breathing after airway repositioning. Injuries incompatible with survival.

## Process:
1. DETECT input language and translate internally
2. EXTRACT all patient info, counts, conditions
3. IDENTIFY hazards (fire, gas, chemicals, collapse, water, electrical)
4. CLASSIFY using START — when multiple patients, use MOST critical
5. RECOMMEND specific resources and interventions
6. ASSESS confidence (lower if input is vague or contradictory)

## Rules:
- When in doubt, choose the MORE urgent category
- Multi-patient: triage on most critical patient
- Always extract location even if partial
- IoT/sensor data is legitimate input
- Generate dispatch code: e.g. "MCI-R-3" = Mass Casualty, Red, 3 patients"""


# ── Triage Engine ─────────────────────────────────────────────────────
def process_text_input(text: str) -> dict:
    """Process text through Gemini and return structured triage data."""
    try:
        response = services.gemini.models.generate_content(
            model=CONFIG.GEMINI_MODEL,
            contents=text,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "response_mime_type": "application/json",
                "response_schema": TriageActionPlan,
            },
        )
        return response.parsed.model_dump() if response.parsed else json.loads(response.text)
    except Exception as exc:
        logger.error("Gemini text processing error: %s", exc)
        raise AIProcessingError() from exc


def process_file_input(file_path: str, mime_type: str, context: str = "") -> dict:
    """Process an uploaded file (image/audio) through Gemini multimodal."""
    try:
        uploaded = services.gemini.files.upload(path=file_path)
        parts: list = []
        if context:
            parts.append(f"Additional context: {context}")
        if mime_type.startswith("image/"):
            parts.append("Analyze this emergency scene image. Perform START triage.")
        else:
            parts.append("Transcribe and analyze this emergency audio. Perform START triage.")
        parts.append(uploaded)

        response = services.gemini.models.generate_content(
            model=CONFIG.GEMINI_MODEL,
            contents=parts,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "response_mime_type": "application/json",
                "response_schema": TriageActionPlan,
            },
        )
        return response.parsed.model_dump() if response.parsed else json.loads(response.text)
    except Exception as exc:
        logger.error("Gemini file processing error: %s", exc)
        raise AIProcessingError() from exc


# ── Flask Application ─────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", static_url_path="")

# Rate limiter store
_rate_store: Dict[str, List[float]] = {}


@app.before_request
def before_request_hook() -> None:
    """Add request ID and start timer for tracing."""
    g.request_id = request.headers.get(
        "X-Cloud-Trace-Context", str(uuid.uuid4())[:8]
    )
    g.request_start = time.monotonic()


@app.after_request
def after_request_hook(response: Response) -> Response:
    """Add security headers, CORS, caching, and log request."""
    # ── Security Headers ──
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(self), geolocation=()"
    )
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' blob: data:; "
        "media-src 'self' blob:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"

    # ── CORS (same-origin) ──
    origin = request.host_url.rstrip("/")
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Request-ID"

    # ── Cache Control ──
    if request.path.endswith((".css", ".js")):
        response.headers["Cache-Control"] = "public, max-age=3600"
    elif request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"

    # ── Request ID header ──
    response.headers["X-Request-ID"] = g.get("request_id", "")

    # ── Structured log ──
    duration = (time.monotonic() - g.get("request_start", 0)) * 1000
    if request.path != "/health":
        logger.info(
            "request_completed",
            extra={
                "request_id": g.get("request_id"),
                "method": request.method,
                "path": request.path,
                "status": response.status_code,
                "duration_ms": round(duration, 2),
            },
        )
    return response


def rate_limit(f):
    """In-memory rate limiter decorator."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        ip = request.remote_addr or "unknown"
        now = time.time()
        _rate_store.setdefault(ip, [])
        _rate_store[ip] = [t for t in _rate_store[ip] if now - t < CONFIG.RATE_LIMIT_WINDOW]
        if len(_rate_store[ip]) >= CONFIG.RATE_LIMIT_MAX:
            return jsonify({"error": "Rate limit exceeded"}), 429
        _rate_store[ip].append(now)
        return f(*args, **kwargs)
    return wrapper


def validate_text(text: Optional[str]) -> str:
    """Validate and sanitise text input. Raises InputValidationError."""
    if not text or not text.strip():
        raise InputValidationError("Please provide emergency scenario text.")
    text = text.strip()
    if len(text) > CONFIG.MAX_TEXT_LENGTH:
        raise InputValidationError(f"Text must be under {CONFIG.MAX_TEXT_LENGTH} characters.")
    return text


@app.errorhandler(TriageError)
def handle_triage_error(exc: TriageError):
    """Centralised error handler for custom exceptions."""
    return jsonify({"error": exc.message}), exc.status_code


# ── Routes ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the main application page."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/health")
def health():
    """Health check for Cloud Run."""
    return jsonify({
        "status": "healthy",
        "service": "TriageAI",
        "version": APP_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/api/info")
def info():
    """Report integrated Google Cloud services status."""
    return jsonify({
        "service": "TriageAI",
        "version": APP_VERSION,
        "google_services": {
            "gemini_api": True,
            "cloud_run": bool(os.environ.get("K_SERVICE")),
            "cloud_logging": _GCP_LOGGING,
            "secret_manager": _GCP_SECRETS,
            "firestore": _GCP_FIRESTORE,
            "cloud_storage": _GCP_STORAGE,
        },
    })


@app.route("/api/triage", methods=["POST"])
@rate_limit
def triage_text():
    """Process text-based emergency input."""
    data = request.get_json(silent=True)
    if not data:
        raise InputValidationError("Request body must be valid JSON.")
    text = validate_text(data.get("text"))
    result = process_text_input(text)
    _save_record(result, "text", text[:100])
    return jsonify({"success": True, "data": result})


@app.route("/api/triage/image", methods=["POST"])
@rate_limit
def triage_image():
    """Process image-based emergency input."""
    if "image" not in request.files:
        raise InputValidationError("Please upload an image file.")
    file = request.files["image"]
    if file.content_type not in CONFIG.ALLOWED_IMAGE_TYPES:
        raise InputValidationError(f"Allowed types: {', '.join(CONFIG.ALLOWED_IMAGE_TYPES)}")
    file.seek(0, 2)
    if file.tell() / (1024 * 1024) > CONFIG.MAX_FILE_SIZE_MB:
        raise InputValidationError(f"Max file size is {CONFIG.MAX_FILE_SIZE_MB}MB.")
    file.seek(0)
    context = request.form.get("context", "")
    suffix = os.path.splitext(file.filename or ".jpg")[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    try:
        services.upload_to_gcs(tmp_path, f"{uuid.uuid4()}{suffix}")
        result = process_file_input(tmp_path, file.content_type, context)
    finally:
        os.unlink(tmp_path)
    _save_record(result, "image", f"📷 {file.filename}")
    return jsonify({"success": True, "data": result})


@app.route("/api/triage/audio", methods=["POST"])
@rate_limit
def triage_audio():
    """Process audio-based emergency input."""
    if "audio" not in request.files:
        raise InputValidationError("Please upload an audio file.")
    file = request.files["audio"]
    content_type = file.content_type or "audio/webm"
    if content_type not in CONFIG.ALLOWED_AUDIO_TYPES:
        raise InputValidationError(f"Allowed types: {', '.join(CONFIG.ALLOWED_AUDIO_TYPES)}")
    file.seek(0, 2)
    if file.tell() / (1024 * 1024) > CONFIG.MAX_FILE_SIZE_MB:
        raise InputValidationError(f"Max file size is {CONFIG.MAX_FILE_SIZE_MB}MB.")
    file.seek(0)
    context = request.form.get("context", "")
    suffix = os.path.splitext(file.filename or ".webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    try:
        services.upload_to_gcs(tmp_path, f"{uuid.uuid4()}{suffix}")
        result = process_file_input(tmp_path, content_type, context)
    finally:
        os.unlink(tmp_path)
    _save_record(result, "audio", "🎤 Audio recording")
    return jsonify({"success": True, "data": result})


@app.route("/api/history", methods=["GET"])
def get_history():
    """Return triage incident history from Firestore."""
    data = services.get_incidents()
    return jsonify({"success": True, "data": data, "count": len(data)})


@app.route("/api/history", methods=["DELETE"])
def clear_history():
    """Clear all incident history."""
    services.clear_incidents()
    return jsonify({"success": True, "message": "History cleared."})


def _save_record(result: dict, input_type: str, preview: str) -> None:
    """Create and persist a triage record."""
    record = {
        "id": result.get("incident_id", str(uuid.uuid4())[:8]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_type": input_type,
        "input_preview": preview,
        "result": result,
    }
    services.save_incident(record)
    logger.info("Triage: %s | %s", result.get("urgency"), result.get("dispatch_code"))


# ── CLI Test Suite ────────────────────────────────────────────────────
def run_cli_tests() -> None:
    """Run CLI stress tests for development and CI."""
    print("=" * 60)
    print("  TriageAI — CLI Test Suite")
    print("=" * 60)
    tests = [
        "Subway crash at 42nd st. Smoke everywhere. One victim bleeding, unconscious. Downed power line.",
        "¡Ayuda! Explosión en la cocina. Mi compañero no puede respirar.",
        "ALARM_ID: 992. TEMP: 105C. SMOKE: POSITIVE. 2 trapped in server room.",
        "Bus flipped. 20 people. Most walking. One woman not breathing. Kid with broken leg.",
    ]
    for i, test in enumerate(tests, 1):
        print(f"\nTEST #{i}: {test[:60]}...")
        result = process_text_input(test)
        print(json.dumps(result, indent=2))


# ── Entry Point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--test" in sys.argv:
        run_cli_tests()
    else:
        port = int(os.environ.get("PORT", 8080))
        logger.info("Starting TriageAI v%s on port %d", APP_VERSION, port)
        app.run(host="0.0.0.0", port=port, debug=True)