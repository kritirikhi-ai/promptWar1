"""
services.py
-----------
Centralised Google Cloud service integrations with fallbacks.
Includes Gemini, Firestore, Cloud Storage, Secret Manager, Translation, and BigQuery.
"""

import logging
import os
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from google.cloud import firestore as cloud_firestore
from google.cloud import storage as cloud_storage
from google.cloud import secretmanager
from google.cloud import bigquery
from google.cloud import translate_v2 as translate
from google import genai

from config import CONFIG
from models import AIProcessingError

logger = logging.getLogger("triageai.services")

class GoogleServices:
    """Manages all Google Cloud service connections and fallbacks."""

    def __init__(self) -> None:
        self._gemini_client: Optional[genai.Client] = None
        self._firestore_db: Optional[Any] = None
        self._storage_client: Optional[Any] = None
        self._bq_client: Optional[Any] = None
        self._translate_client: Optional[Any] = None
        self._memory_store: deque[dict] = deque(maxlen=200)
        self._api_key: str = self._resolve_api_key()

        self._has_bq = False
        self._has_translate = False
        self._has_firestore = False
        self._has_secrets = False
        self._has_storage = False

        self._init_clients()

    def _init_clients(self) -> None:
        """Initialize all optional GCP clients safely."""
        try:
            self._bq_client = bigquery.Client()
            self._has_bq = True
        except Exception:
            pass

        try:
            self._translate_client = translate.Client()
            self._has_translate = True
        except Exception:
            pass

        try:
            self._firestore_db = cloud_firestore.Client()
            self._has_firestore = True
        except Exception:
            pass

        try:
            self._storage_client = cloud_storage.Client()
            self._has_storage = True
        except Exception:
            pass
            
        try:
            secretmanager.SecretManagerServiceClient()
            self._has_secrets = True
        except Exception:
            pass

    def get_service_status(self) -> dict:
        """Returns boolean status of all integrated services."""
        return {
            "gemini_api": True,
            "cloud_logging": bool(os.environ.get("K_SERVICE")),
            "secret_manager": self._has_secrets,
            "firestore": self._has_firestore,
            "cloud_storage": self._has_storage,
            "bigquery": self._has_bq,
            "translate_api": self._has_translate
        }

    # ── Secret Manager ────────────────────────────────────────────────
    def _resolve_api_key(self) -> str:
        """Retrieve API key: Secret Manager → env var fallback."""
        if os.environ.get("GOOGLE_CLOUD_PROJECT"):
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
            logger.warning("GEMINI_API_KEY not set \u2014 API calls will fail")
        return key

    # ── Gemini ────────────────────────────────────────────────────────
    @property
    def gemini(self) -> genai.Client:
        """Lazy singleton Gemini client for connection reuse."""
        if self._gemini_client is None:
            self._gemini_client = genai.Client(api_key=self._api_key)
        return self._gemini_client

    # ── Translation API ───────────────────────────────────────────────
    def translate_text_to_english(self, text: str) -> str:
        """Uses Translation API to ensure input is in English if possible."""
        if not self._translate_client:
            return text  # fallback to Gemini directly handling multi-lingual
            
        try:
            result = self._translate_client.translate(text, target_language="en")
            return result.get("translatedText", text)
        except Exception as exc:
            logger.warning("Translation API failed: %s", exc)
            return text

    # ── Firestore ─────────────────────────────────────────────────────
    def save_incident(self, record: dict) -> None:
        """Persist incident to Firestore (fallback: in-memory)."""
        self._memory_store.appendleft(record)
        if self._firestore_db:
            try:
                self._firestore_db.collection(CONFIG.FIRESTORE_COLLECTION).document(
                    record["id"]
                ).set(record)
            except Exception as exc:
                logger.error("Firestore write failed: %s", exc)

    def get_incidents(self, limit: int = 50) -> List[dict]:
        """Read incidents: Firestore → in-memory fallback."""
        if self._firestore_db:
            try:
                docs = (
                    self._firestore_db.collection(CONFIG.FIRESTORE_COLLECTION)
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
        if self._firestore_db:
            try:
                docs = self._firestore_db.collection(CONFIG.FIRESTORE_COLLECTION).limit(500).stream()
                for doc in docs:
                    doc.reference.delete()
                logger.info("Firestore incidents cleared")
            except Exception as exc:
                logger.error("Firestore clear failed: %s", exc)

    # ── BigQuery ──────────────────────────────────────────────────────
    def stream_to_bigquery(self, record: dict) -> None:
        """Stream the incident record into BigQuery for analytics tracking."""
        if not self._bq_client or not CONFIG.GOOGLE_CLOUD_PROJECT:
            return
            
        try:
            table_id = f"{CONFIG.GOOGLE_CLOUD_PROJECT}.{CONFIG.BIGQUERY_DATASET}.{CONFIG.BIGQUERY_TABLE}"
            # Ensure row is formatted cleanly for BQ (flattens complex dicts to strings)
            row = {
                "incident_id": record["id"],
                "timestamp": record["timestamp"],
                "input_type": record["input_type"],
                "urgency": record.get("result", {}).get("urgency", "UNKNOWN"),
                "patient_count": record.get("result", {}).get("patient_count", 0),
                "dispatch_code": record.get("result", {}).get("dispatch_code", ""),
                "location_info": record.get("result", {}).get("location_info", ""),
            }
            errors = self._bq_client.insert_rows_json(table_id, [row])
            if errors:
                logger.error("BigQuery insert errors: %s", errors)
            else:
                logger.info("Streamed record %s to BigQuery", record["id"])
        except Exception as exc:
            # Swallow BQ errors to not crash the main triage flow
            logger.warning("BigQuery streaming failed or table not found: %s", exc)

    # ── Cloud Storage ─────────────────────────────────────────────────
    def upload_to_gcs(self, local_path: str, filename: str) -> Optional[str]:
        """Archive uploaded file to Cloud Storage. Returns GCS URI."""
        if not CONFIG.GCS_BUCKET or not self._storage_client:
            return None
            
        try:
            bucket = self._storage_client.bucket(CONFIG.GCS_BUCKET)
            blob_name = f"uploads/{datetime.now(timezone.utc).strftime('%Y%m%d')}/{filename}"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_path)
            uri = f"gs://{CONFIG.GCS_BUCKET}/{blob_name}"
            logger.info("File archived to %s", uri)
            return uri
        except Exception as exc:
            logger.warning("GCS upload failed: %s", exc)
            return None


services = GoogleServices()
