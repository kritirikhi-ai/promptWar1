"""
config.py
---------
Centralized configuration definitions and secure defaults for TriageAI.
"""

import os
from dataclasses import dataclass, field
from typing import FrozenSet
from dotenv import load_dotenv

import logging

load_dotenv()

APP_VERSION = "4.0.0"

_GCP_LOGGING = False
try:
    import google.cloud.logging as cloud_logging
    _GCP_LOGGING = True
except ImportError:
    pass

def init_logging() -> None:
    """Set up logging: Cloud Logging on GCP, standard otherwise."""
    if _GCP_LOGGING and os.environ.get("K_SERVICE"):
        client = cloud_logging.Client()
        client.setup_logging(log_level=logging.INFO)
        logging.getLogger("triageai").info("Cloud Logging integration active")
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

init_logging()


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration with secure defaults."""

    GEMINI_MODEL: str = "gemini-2.5-flash"
    MAX_TEXT_LENGTH: int = 5000
    MAX_FILE_SIZE_MB: int = 10
    RATE_LIMIT_WINDOW: int = 60
    RATE_LIMIT_MAX: int = 30
    
    FIRESTORE_COLLECTION: str = "triage_incidents"
    BIGQUERY_DATASET: str = os.environ.get("BIGQUERY_DATASET", "triage_analytics")
    BIGQUERY_TABLE: str = os.environ.get("BIGQUERY_TABLE", "incidents")
    GCS_BUCKET: str = os.environ.get("GCS_BUCKET", "")
    GOOGLE_CLOUD_PROJECT: str = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    GOOGLE_MAPS_API_KEY: str = os.environ.get("GOOGLE_MAPS_API_KEY", "")

    ALLOWED_IMAGE_TYPES: FrozenSet[str] = frozenset(
        {"image/jpeg", "image/png", "image/webp", "image/gif"}
    )
    ALLOWED_AUDIO_TYPES: FrozenSet[str] = frozenset(
        {"audio/webm", "audio/ogg", "audio/wav", "audio/mp3",
         "audio/mpeg", "audio/mp4", "audio/x-m4a"}
    )
    ALLOWED_DOC_TYPES: FrozenSet[str] = frozenset(
        {"application/pdf", "text/plain", "text/markdown"}
    )


CONFIG = AppConfig()
