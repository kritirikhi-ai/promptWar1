"""
models.py
---------
Pydantic schemas for the TriageAI application and custom domain exceptions.
"""

import uuid
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


# ── Custom Exceptions ─────────────────────────────────────────────────

class TriageError(Exception):
    """Base exception for triage processing errors."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class InputValidationError(TriageError):
    """Raised when user input fails validation or is malformed."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400)


class AIProcessingError(TriageError):
    """Raised when Gemini API or Cloud Translation processing fails."""

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
        description="START triage classification (RED=Immediate, YELLOW=Delayed, GREEN=Minor, BLACK=Expectant)",
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model confidence in classification bounds (0.01 to 1.0)",
    )
    patient_count: int = Field(ge=0, description="Estimated number of patients")
    hazard_alerts: List[str] = Field(description="Environmental hazards identified if any")
    patient_vitals_summary: str = Field(description="Patient conditions and vitals")
    critical_intervention: str = Field(description="Most urgent action needed")
    recommended_resources: List[str] = Field(description="Specific resources needed like hazmat, ems")
    location_info: str = Field(description="Extracted location details")
    dispatch_code: str = Field(description="Priority dispatch code (e.g. MCI-R-3)")
    language_detected: str = Field(default="en", description="ISO 639-1 code of original intent")
    raw_input_summary: str = Field(description="Brief English summary of the original text")


class TriageRecord(BaseModel):
    """A triage incident record for persistence format validation."""

    id: str
    timestamp: str
    input_type: str
    input_preview: str
    result: Dict[str, Any]
