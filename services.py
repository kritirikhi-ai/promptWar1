"""
services.py
-----------
Centralised Google Cloud service integrations with fallbacks.
Includes Gemini, Firestore, Cloud Storage, Secret Manager, Translation, BigQuery,
DLP, Text-to-Speech (TTS), Speech-to-Text (STT), and Google Maps.
"""

import logging
import os
import datetime
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from google.cloud import firestore as cloud_firestore
from google.cloud import storage as cloud_storage
from google.cloud import secretmanager
from google.cloud import bigquery
from google.cloud import translate_v2 as translate
from google.cloud import dlp_v2
from google.cloud import speech
from google.cloud import texttospeech
from google import genai
import googlemaps

from config import CONFIG

logger = logging.getLogger("triageai.services")

class GoogleServices:
    """Manages 12 Google Cloud/Maps service connections and fallbacks."""

    def __init__(self) -> None:
        self._gemini_client: Optional[genai.Client] = None
        self._firestore_db: Optional[Any] = None
        self._storage_client: Optional[Any] = None
        self._bq_client: Optional[Any] = None
        self._translate_client: Optional[Any] = None
        self._dlp_client: Optional[Any] = None
        self._speech_client: Optional[Any] = None
        self._tts_client: Optional[Any] = None
        self._maps_client: Optional[Any] = None
        
        self._memory_store: deque[dict] = deque(maxlen=200)
        self._api_key: str = self._resolve_api_key()

        self._has_bq = False
        self._has_translate = False
        self._has_firestore = False
        self._has_secrets = False
        self._has_storage = False
        self._has_dlp = False
        self._has_stt = False
        self._has_tts = False
        self._has_maps = False

        self._init_clients()

    def _init_clients(self) -> None:
        """Initialize all optional GCP clients safely."""
        try:
            self._bq_client = bigquery.Client()
            self._has_bq = True
        except Exception: pass

        try:
            self._translate_client = translate.Client()
            self._has_translate = True
        except Exception: pass

        try:
            self._firestore_db = cloud_firestore.Client()
            self._has_firestore = True
        except Exception: pass

        try:
            self._storage_client = cloud_storage.Client()
            self._has_storage = True
        except Exception: pass
            
        try:
            secretmanager.SecretManagerServiceClient()
            self._has_secrets = True
        except Exception: pass
            
        try:
            self._dlp_client = dlp_v2.DlpServiceClient()
            self._has_dlp = True
        except Exception: pass

        try:
            self._speech_client = speech.SpeechClient()
            self._has_stt = True
        except Exception: pass

        try:
            self._tts_client = texttospeech.TextToSpeechClient()
            self._has_tts = True
        except Exception: pass
        
        if CONFIG.GOOGLE_MAPS_API_KEY:
            try:
                self._maps_client = googlemaps.Client(key=CONFIG.GOOGLE_MAPS_API_KEY)
                self._has_maps = True
            except Exception: pass

    def get_service_status(self) -> dict:
        """Returns boolean status of all 12 integrated services."""
        return {
            "gemini_api": True,
            "cloud_logging": bool(os.environ.get("K_SERVICE")),
            "secret_manager": self._has_secrets,
            "firestore": self._has_firestore,
            "cloud_storage": self._has_storage,
            "bigquery": self._has_bq,
            "translate_api": self._has_translate,
            "dlp_api": self._has_dlp,
            "speech_to_text": self._has_stt,
            "text_to_speech": self._has_tts,
            "google_maps": self._has_maps
        }

    # ── Secret Manager ────────────────────────────────────────────────
    def _resolve_api_key(self) -> str:
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
            logger.warning("GEMINI_API_KEY not set")
        return key

    # ── Gemini ────────────────────────────────────────────────────────
    @property
    def gemini(self) -> genai.Client:
        if self._gemini_client is None:
            self._gemini_client = genai.Client(api_key=self._api_key)
        return self._gemini_client

    # ── DLP (Security) ────────────────────────────────────────────────
    def redact_pii(self, text: str) -> str:
        """Uses Cloud DLP to redact PII (names, phones) from medical history."""
        if not self._dlp_client or not CONFIG.GOOGLE_CLOUD_PROJECT:
            return text
            
        try:
            parent = f"projects/{CONFIG.GOOGLE_CLOUD_PROJECT}/locations/global"
            item = {"value": text}
            inspect_config = {
                "info_types": [{"name": "PERSON_NAME"}, {"name": "PHONE_NUMBER"}, {"name": "EMAIL_ADDRESS"}],
                "min_likelihood": dlp_v2.Likelihood.LIKELY,
            }
            deidentify_config = {
                "info_type_transformations": {
                    "transformations": [
                        {
                            "primitive_transformation": {"replace_with_info_type_config": {}}
                        }
                    ]
                }
            }
            request = {
                "parent": parent,
                "item": item,
                "inspect_config": inspect_config,
                "deidentify_config": deidentify_config,
            }
            response = self._dlp_client.deidentify_content(request=request)
            return response.item.value
        except Exception as exc:
            logger.warning("DLP redaction failed: %s", exc)
            return text

    # ── Text-to-Speech (TTS Voice Alerts) ──────────────────────────────
    def generate_voice_alert(self, text: str, language_code: str = "en-IN") -> Optional[bytes]:
        """Synthesizes an audio alert for the critical intervention."""
        if not self._tts_client:
            return None
            
        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Map common languages to high-quality voices
            voice_name = "en-IN-Wavenet-B"
            if language_code.startswith("hi"):
                language_code = "hi-IN"
                voice_name = "hi-IN-Wavenet-A"
            elif language_code.startswith("es"):
                language_code = "es-US"
                voice_name = "es-US-Wavenet-B"
                
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code,
                name=voice_name
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
            response = self._tts_client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
            return response.audio_content
        except Exception as exc:
            logger.warning("TTS generation failed: %s", exc)
            return None

    # ── Speech-to-Text (STT) ──────────────────────────────────────────
    def transcribe_audio(self, local_path: str) -> Optional[str]:
        """Transcribes audio explicitly using Google Cloud Speech-to-Text."""
        if not self._speech_client:
            return None
            
        try:
            with open(local_path, "rb") as f:
                content = f.read()
                
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
                sample_rate_hertz=48000,
                language_code="en-US",
                alternative_language_codes=["hi-IN", "es-US"]
            )
            
            response = self._speech_client.recognize(config=config, audio=audio)
            
            transcript = " ".join([result.alternatives[0].transcript for result in response.results])
            return transcript.strip() if transcript else None
        except Exception as exc:
            logger.warning("STT transcription failed: %s", exc)
            return None

    # ── Google Maps (Geocoding) ───────────────────────────────────────
    def geocode_location(self, location_text: str) -> Tuple[Optional[float], Optional[float]]:
        """Converts an unstructured location string into precise Lat/Lng coords."""
        if not self._maps_client or not location_text or location_text.lower() == "unknown":
            return None, None
            
        try:
            result = self._maps_client.geocode(location_text)
            if result:
                loc = result[0]["geometry"]["location"]
                return loc["lat"], loc["lng"]
        except Exception as exc:
            logger.warning("Google Maps geocoding failed: %s", exc)
        return None, None

    # ── Translation API ───────────────────────────────────────────────
    def translate_text_to_english(self, text: str) -> str:
        if not self._translate_client:
            return text
        try:
            result = self._translate_client.translate(text, target_language="en")
            return result.get("translatedText", text)
        except Exception:
            return text

    # ── Firestore ─────────────────────────────────────────────────────
    def save_incident(self, record: dict) -> None:
        self._memory_store.appendleft(record)
        if self._firestore_db:
            try:
                self._firestore_db.collection(CONFIG.FIRESTORE_COLLECTION).document(
                    record["id"]
                ).set(record)
            except Exception as exc:
                logger.error("Firestore write failed: %s", exc)

    def get_incidents(self, limit: int = 50) -> List[dict]:
        if self._firestore_db:
            try:
                docs = (
                    self._firestore_db.collection(CONFIG.FIRESTORE_COLLECTION)
                    .order_by("timestamp", direction=cloud_firestore.Query.DESCENDING)
                    .limit(limit)
                    .stream()
                )
                return [doc.to_dict() for doc in docs]
            except Exception:
                pass
        return list(self._memory_store)

    def clear_incidents(self) -> None:
        self._memory_store.clear()
        if self._firestore_db:
            try:
                docs = self._firestore_db.collection(CONFIG.FIRESTORE_COLLECTION).limit(500).stream()
                for doc in docs:
                    doc.reference.delete()
            except Exception:
                pass

    # ── BigQuery ──────────────────────────────────────────────────────
    def stream_to_bigquery(self, record: dict) -> None:
        if not self._bq_client or not CONFIG.GOOGLE_CLOUD_PROJECT:
            return
        try:
            table_id = f"{CONFIG.GOOGLE_CLOUD_PROJECT}.{CONFIG.BIGQUERY_DATASET}.{CONFIG.BIGQUERY_TABLE}"
            # Ensure safe BQ types
            row = {
                "incident_id": record["id"],
                "timestamp": record["timestamp"],
                "input_type": record["input_type"],
                "urgency": record.get("result", {}).get("urgency", "UNKNOWN"),
                "patient_count": record.get("result", {}).get("patient_count", 0),
                "dispatch_code": record.get("result", {}).get("dispatch_code", ""),
                "location_info": record.get("result", {}).get("location_info", ""),
            }
            self._bq_client.insert_rows_json(table_id, [row])
        except Exception:
            pass

    # ── Cloud Storage (Signed URLs) ───────────────────────────────────
    def upload_to_gcs(self, local_path: str, filename: str, content_type: str = "application/octet-stream") -> Tuple[Optional[str], Optional[str]]:
        """Archives to GCS and returns both the GS URI and a 2-hour Signed URL."""
        if not CONFIG.GCS_BUCKET or not self._storage_client:
            return None, None
            
        try:
            bucket = self._storage_client.bucket(CONFIG.GCS_BUCKET)
            blob_name = f"uploads/{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d')}/{filename}"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_path, content_type=content_type)
            
            uri = f"gs://{CONFIG.GCS_BUCKET}/{blob_name}"
            # Generate signed URL
            signed_url = blob.generate_signed_url(
                version="v4",
                expiration=datetime.timedelta(hours=2),
                method="GET"
            )
            return uri, signed_url
        except Exception as exc:
            logger.warning("GCS upload/signing failed: %s", exc)
            return None, None

services = GoogleServices()
