"""
engine.py
---------
The core Triage Engine handling the AI execution via Gemini and Translate.
Converts unstructured inputs into structured Pydantic models.
"""

import json
import logging
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
- General IoT or sensor data is legitimate input
- Generate dispatch code: e.g. "MCI-R-3" = Mass Casualty, Red, 3 patients"""


def process_text_input(text: str) -> Dict[str, Any]:
    """Process text through Translate & Gemini and return structured triage data."""
    try:
        # Explicit Google Service interaction: Pre-processing with Cloud Translation
        translated_text = services.translate_text_to_english(text)

        response = services.gemini.models.generate_content(
            model=CONFIG.GEMINI_MODEL,
            contents=translated_text,
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


def process_file_input(file_path: str, mime_type: str, context: str = "") -> Dict[str, Any]:
    """Process an uploaded file (image/audio) through Gemini multimodal."""
    try:
        # 1. Translate context via explicit Translation API
        if context:
            context = services.translate_text_to_english(context)
            
        # 2. Upload file via Gemini API
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
