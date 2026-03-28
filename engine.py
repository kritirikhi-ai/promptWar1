"""
engine.py
---------
The core Triage Engine handling the AI execution via Gemini, Translate, DLP,
STT, TTS, Google Maps, and Google Search Grounding.
"""

import json
import logging
import os
import uuid
from typing import Dict, Any

from config import CONFIG
from models import TriageActionPlan, AIProcessingError
from services import services

logger = logging.getLogger("triageai.engine")

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
    """Process text through DLP, Translate & Gemini with Search Grounding."""
    try:
        # Step 1: Security Redaction
        redacted_text = services.redact_pii(text)

        # Step 2: Translation Normalization
        translated_text = services.translate_text_to_english(redacted_text)

        # Step 3: Gemini Analysis (with Google Search Grounding enabled)
        response = services.gemini.models.generate_content(
            model=CONFIG.GEMINI_MODEL,
            contents=translated_text,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "response_mime_type": "application/json",
                "response_schema": TriageActionPlan,
                "tools": [{"google_search": {}}]  # Hackathon Flex: Search Grounding
            },
        )
        
        result_dict = response.parsed.model_dump() if response.parsed else json.loads(response.text)
        return _enrich_result(result_dict)
        
    except Exception as exc:
        logger.error("Gemini text processing error: %s", exc)
        raise AIProcessingError() from exc


def process_file_input(file_path: str, mime_type: str, context: str = "") -> Dict[str, Any]:
    """Process an uploaded file (image/audio/pdf) through multimodal AI."""
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
            
        uploaded = services.gemini.files.upload(path=file_path)
        
        if mime_type.startswith("image/"):
            parts.append("Analyze this emergency scene image. Perform START triage.")
        elif mime_type == "application/pdf":
            parts.append("Analyze this patient medical history document. Identify vulnerabilities and allergies, then perform START triage based on the context.")
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
                "tools": [{"google_search": {}}]  # Hackathon Flex: Search Grounding
            },
        )
        
        result_dict = response.parsed.model_dump() if response.parsed else json.loads(response.text)
        if transcript:
            result_dict["stt_transcript"] = transcript
            
        return _enrich_result(result_dict)
        
    except Exception as exc:
        logger.error("Gemini file processing error: %s", exc)
        raise AIProcessingError() from exc
