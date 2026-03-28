"""
engine.py
---------
The core Triage Engine handling the AI execution via Gemini, Translate, DLP,
STT, TTS, Google Maps, and Google Search Grounding.

Optimised for performance with:
- Pre-computed model fallback chain (avoids per-request allocation)
- Parallel post-processing enrichment (geocoding + TTS concurrently)
- In-memory GCS upload for TTS (avoids temp file I/O)
- LRU caching for repeated identical text inputs
- Automatic model fallback on rate-limit errors
"""

import hashlib
import io
import json
import logging
import os
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from config import CONFIG
from models import AIProcessingError, TriageActionPlan
from services import services

logger = logging.getLogger("triageai.engine")

# ── Thread Pool for Parallel Enrichment ──────────────────────────────
# Reused across requests — avoids thread creation overhead per call.
_enrichment_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="enrich")

# ── Model Fallback Chain (pre-computed once at module load) ──────────
# When the primary model is rate-limited (429), try the next model.
# gemini-2.5-flash (best) → gemini-2.0-flash → gemini-2.5-flash-lite (highest quota)
_MODEL_CHAIN: Tuple[str, ...] = tuple(dict.fromkeys([
    CONFIG.GEMINI_MODEL,
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
]))


def _call_gemini_with_fallback(*, contents, config, operation: str = "generate"):
    """Call Gemini generate_content with automatic model fallback on 429.

    Uses the pre-computed _MODEL_CHAIN to avoid per-request list allocation.
    Minimises sleep between fallback attempts (0.5s vs 2s) since model quotas
    are independent.
    """
    last_exc = None

    for model in _MODEL_CHAIN:
        try:
            logger.info("Trying model %s for %s", model, operation)
            response = services.gemini.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return response
        except Exception as exc:
            exc_str = str(exc)
            if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str:
                logger.warning(
                    "Model %s rate-limited for %s, trying next fallback...",
                    model, operation,
                )
                last_exc = exc
                time.sleep(0.5)  # Brief pause — models have independent quotas
                continue
            # Non-rate-limit error — raise immediately
            raise

    # All models exhausted
    raise last_exc


# ── START Triage System Prompt ────────────────────────────────────────
SYSTEM_PROMPT = """You are TriageAI, an advanced emergency triage system built on the START (Simple Triage and Rapid Treatment) protocol.

## START Triage Classification Rules:
**RED (Immediate):** Life-threatening, high survival if treated NOW. Resp rate >30, no radial pulse, or cannot follow commands.
**YELLOW (Delayed):** Serious but stable. Can follow commands but cannot walk.
**GREEN (Minor):** Walking wounded. Ambulatory, minor injuries.
**BLACK (Expectant):** Not breathing after airway repositioning. Injuries incompatible with survival.

## Process:
1. **LIVE CONTEXT**: Use Google Search grounding to map the location to real-world weather and traffic data. Factor this into hazards.
2. DETECT input language and translate internally.
3. EXTRACT all patient info, conditions, and REDACTED medical histories.
4. IDENTIFY hazards (fire, traffic, severe weather context).
5. CLASSIFY using START — when multiple patients, use MOST critical.
6. ASSESS confidence and generate dispatch codes.
"""


def _geocode_task(location_info: str) -> Tuple[Optional[float], Optional[float]]:
    """Geocode in a thread — returns (lat, lng) or (None, None)."""
    return services.geocode_location(location_info)


def _tts_and_upload_task(
    alert_text: str, language_code: str
) -> Optional[str]:
    """Generate TTS audio and upload directly from memory to GCS.

    Avoids temp file I/O by using in-memory bytes upload when available,
    falling back to temp file only when GCS requires a file path.
    """
    audio_bytes = services.generate_voice_alert(alert_text, language_code=language_code)
    if not audio_bytes:
        return None

    # Try in-memory upload first (avoids disk I/O)
    signed_url = services.upload_bytes_to_gcs(
        data=audio_bytes,
        filename=f"tts_{uuid.uuid4().hex}.mp3",
        content_type="audio/mpeg",
    )
    return signed_url


def _enrich_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Appends Coordinates and Voice Alerts to the LLM response.

    Runs geocoding and TTS generation in parallel using a thread pool
    to reduce total enrichment latency from ~sequential to ~max(geocode, tts).
    """
    location_info = result.get("location_info", "")
    alert_text = f"Priority {result.get('urgency')}. {result.get('critical_intervention')}."
    lang_code = result.get("language_detected", "en")

    # Submit both I/O tasks concurrently
    geo_future = _enrichment_pool.submit(_geocode_task, location_info)
    tts_future = _enrichment_pool.submit(_tts_and_upload_task, alert_text, lang_code)

    # Collect results (blocks until both complete)
    try:
        lat, lng = geo_future.result(timeout=10)
    except Exception:
        lat, lng = None, None

    try:
        tts_url = tts_future.result(timeout=15)
    except Exception:
        tts_url = None

    result["latitude"] = lat
    result["longitude"] = lng
    if tts_url:
        result["tts_audio_url"] = tts_url

    return result


# ── LRU Cache for repeated identical text inputs ─────────────────────
@lru_cache(maxsize=64)
def _cached_text_hash(text_hash: str, text: str) -> Dict[str, Any]:
    """Internal cached processor — keyed by content hash to avoid
    redundant API calls for the same emergency text.

    Returns a fresh dict (caller must not mutate the cache entry).
    """
    return _process_text_uncached(text)


def _process_text_uncached(text: str) -> Dict[str, Any]:
    """Core 2-step Gemini pipeline without caching wrapper."""
    # Step 1: Security Redaction
    redacted_text = services.redact_pii(text)

    # Step 2: Translation Normalization
    translated_text = services.translate_text_to_english(redacted_text)

    # Step 3a: Gemini with Google Search Grounding (no response_schema)
    grounded_response = _call_gemini_with_fallback(
        contents=translated_text,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "tools": [{"google_search": {}}],
        },
        operation="text_grounding",
    )
    grounded_text = grounded_response.text

    # Step 3b: Structured extraction using response_schema (no grounding)
    structured_prompt = (
        f"Based on the following grounded triage analysis, produce the "
        f"structured JSON output:\n\n{grounded_text}"
    )
    structured_response = _call_gemini_with_fallback(
        contents=structured_prompt,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "response_mime_type": "application/json",
            "response_schema": TriageActionPlan,
        },
        operation="text_structuring",
    )

    return (
        structured_response.parsed.model_dump()
        if structured_response.parsed
        else json.loads(structured_response.text)
    )


def process_text_input(text: str) -> Dict[str, Any]:
    """Process text through DLP, Translate & Gemini with Search Grounding.

    Uses a two-step pipeline because response_schema and google_search
    grounding cannot be combined in a single Gemini API call.
    Includes LRU caching and automatic model fallback on rate-limit errors.
    """
    try:
        # Use content hash for cache key (avoids storing full text as key)
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        result_dict = _cached_text_hash(text_hash, text)
        # Return a copy so callers can mutate without polluting cache
        return _enrich_result(dict(result_dict))

    except Exception as exc:
        logger.error("Gemini text processing error: %s", exc, exc_info=True)
        raise AIProcessingError() from exc


def process_file_input(file_path: str, mime_type: str, context: str = "") -> Dict[str, Any]:
    """Process an uploaded file (image/audio/pdf) through multimodal AI.

    Includes automatic model fallback on rate-limit errors.
    """
    try:
        parts: list = []
        transcript = None

        # Audio Pre-processing (Cloud Speech-to-Text Integration)
        if mime_type.startswith("audio/"):
            transcript = services.transcribe_audio(file_path)
            if transcript:
                parts.append(f"[Cloud STT Pre-Transcription for accuracy]: '{transcript}'")

        # Security Redaction on context
        if context:
            redacted_context = services.redact_pii(context)
            context = services.translate_text_to_english(redacted_context)
            parts.append(f"Additional context: {context}")

        uploaded = services.gemini.files.upload(file=file_path)

        if mime_type.startswith("image/"):
            parts.append("Analyze this emergency scene image. Perform START triage.")
        elif mime_type == "application/pdf":
            parts.append(
                "Analyze this patient medical history document. Identify "
                "vulnerabilities and allergies, then perform START triage "
                "based on the context."
            )
        else:
            parts.append("Transcribe and analyze this emergency audio. Perform START triage.")

        parts.append(uploaded)

        response = _call_gemini_with_fallback(
            contents=parts,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "response_mime_type": "application/json",
                "response_schema": TriageActionPlan,
            },
            operation="file_analysis",
        )

        result_dict = (
            response.parsed.model_dump()
            if response.parsed
            else json.loads(response.text)
        )
        if transcript:
            result_dict["stt_transcript"] = transcript

        return _enrich_result(result_dict)

    except Exception as exc:
        logger.error("Gemini file processing error: %s", exc, exc_info=True)
        raise AIProcessingError() from exc
