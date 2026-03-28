"""
TriageAI — Mass Casualty & Disaster Triage System

Entry point and Flask Application Factory.
Provides multimodal API routes to process emergency inputs into structured data.
"""

from __future__ import annotations

import logging
import os
import sys
import json
import tempfile
import time
import uuid
from datetime import datetime, timezone
from functools import wraps
from typing import Dict, List, Optional

from flask import Flask, Response, g, jsonify, request, send_from_directory

from config import CONFIG, APP_VERSION
from models import TriageError, InputValidationError
from services import services
from engine import process_text_input, process_file_input

logger = logging.getLogger("triageai.app")

app = Flask(__name__, static_folder="static", static_url_path="")

# Rate limiter store (in-memory)
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
    # Security Headers
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

    # CORS
    origin = request.host_url.rstrip("/")
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Request-ID"

    # Cache Control
    if request.path.endswith((".css", ".js")):
        response.headers["Cache-Control"] = "public, max-age=3600"
    elif request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"

    response.headers["X-Request-ID"] = g.get("request_id", "")

    # Structured log
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
    """Validate and sanitise text input."""
    if not text or not text.strip():
        raise InputValidationError("Please provide emergency scenario text.")
    text = text.strip()
    if len(text) > CONFIG.MAX_TEXT_LENGTH:
        raise InputValidationError(f"Text must be under {CONFIG.MAX_TEXT_LENGTH} characters.")
    return text


@app.errorhandler(TriageError)
def handle_triage_error(exc: TriageError):
    """Centralised error handler for custom logic exceptions."""
    return jsonify({"error": exc.message}), exc.status_code


# ── Routes ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main application page."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/health")
def health():
    """Health check for Cloud Run routing."""
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
        "google_services": services.get_service_status(),
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
    """Create and persist a triage record locally and stream to BQ."""
    record = {
        "id": result.get("incident_id", str(uuid.uuid4())[:8]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_type": input_type,
        "input_preview": preview,
        "result": result,
    }
    services.save_incident(record)
    services.stream_to_bigquery(record)
    logger.info("Triage: %s | %s", result.get("urgency"), result.get("dispatch_code"))


# ── CLI Test Driver ───────────────────────────────────────────────────

def run_cli_tests() -> None:
    """Run simulated test queries and dump outputs to stdout."""
    print("=" * 60)
    print("  TriageAI — Modular Test Run Engine")
    print("=" * 60)
    tests = [
        "Subway crash at 42nd st. Smoke everywhere. One victim bleeding, unconscious. Downed power line.",
        "¡Ayuda! Explosión en la cocina. Mi compañero no puede respirar.",
        "ALARM_ID: 992. TEMP: 105C. SMOKE: POSITIVE. 2 trapped in server room.",
    ]
    for i, test_val in enumerate(tests, 1):
        print(f"\nTEST #{i}: {test_val[:60]}...")
        res = process_text_input(test_val)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_cli_tests()
    else:
        port = int(os.environ.get("PORT", 8080))
        logger.info("Starting TriageAI v%s on port %d", APP_VERSION, port)
        app.run(host="0.0.0.0", port=port, debug=True)