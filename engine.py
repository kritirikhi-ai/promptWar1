"""
engine.py
---------
The core Triage Engine handling the AI execution via Gemini, Translate, DLP,
STT, TTS, Google Maps, and Google Search Grounding.

Includes automatic model fallback when rate limits are hit.
"""

import json
import logging
import os
import time
import uuid
from typing import Dict, Any, List

from config import CONFIG
from models import TriageActionPlan, AIProcessingError
from services import services

logger = logging.getLogger("triageai.engine")

# ── Model Fallback Chain ─────────────────────────────────────────────
# When the primary model is rate-limited (429), try the next model.
# gemini-2.5-flash (best) → gemini-2.0-flash → gemini-2.5-flash-lite (highest quota)
FALLBACK_MODELS: List[str] = [
    CONFIG.GEMINI_MODEL,
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
]

def _get_model_chain() -> List[str]:
    """Return a de-duplicated ordered list of models to try."""
    seen = set()
    chain = []
    for m in FALLBACK_MODELS:
        if m not in seen:
            seen.add(m)
            chain.append(m)
    return chain


def _call_gemini_with_fallback(*, contents, config, operation: str = "generate"):
    """Call Gemini generate_content with automatic model fallback on 429."""
    models = _get_model_chain()
    last_exc = None

    for model in models:
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
                time.sleep(2)  # Brief pause before trying next model
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

def _enrich_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Appends Coordinates and Voice Alerts to the LLM response."""
    # 1. Geocode Location
    lat, lng = services.geocode_location(result.get("location_info", ""))
    result["latitude"] = lat
    result["longitude"] = lng
    
    # 2. Text-to-Speech (Voice Alert)
    alert_text = f"Priority {result.get('urgency')}. {result.get('critical_intervention')}."
    lang_code = result.get("language_detected", "en")
    
    audio_bytes = services.generate_voice_alert(alert_text, language_code=lang_code)
    
    if audio_bytes:
        # Save TTS to cloud storage so frontend can play it
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        try:
            filename = f"tts_{uuid.uuid4().hex}.mp3"
            _, signed_url = services.upload_to_gcs(tmp_path, filename, content_type="audio/mp3")
            result["tts_audio_url"] = signed_url
        finally:
            os.unlink(tmp_path)
            
    return result


def process_text_input(text: str) -> Dict[str, Any]:
    """Process text through DLP, Translate & Gemini with Search Grounding.
    
    Uses a two-step pipeline because response_schema and google_search
    grounding cannot be combined in a single Gemini API call.
    Includes automatic model fallback on rate-limit errors.
    """
    try:
        # Step 1: Security Redaction
        redacted_text = services.redact_pii(text)

        # Step 2: Translation Normalization
        translated_text = services.translate_text_to_english(redacted_text)

        # Step 3a: Gemini with Google Search Grounding (no response_schema)
        # response_schema + google_search are incompatible in the same call,
        # so we split into two steps.
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

        result_dict = (
            structured_response.parsed.model_dump()
            if structured_response.parsed
            else json.loads(structured_response.text)
        )
        return _enrich_result(result_dict)
        
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
            parts.append("Analyze this patient medical history document. Identify vulnerabilities and allergies, then perform START triage based on the context.")
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
        
        result_dict = response.parsed.model_dump() if response.parsed else json.loads(response.text)
        if transcript:
            result_dict["stt_transcript"] = transcript
            
        return _enrich_result(result_dict)
        
    except Exception as exc:
        logger.error("Gemini file processing error: %s", exc, exc_info=True)
        raise AIProcessingError() from exc
