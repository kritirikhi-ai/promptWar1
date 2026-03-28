"""
Tests for Flask API endpoints — security headers, input validation,
error handling, and Google services integration.
"""

import pytest
import json
import io
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import app, TriageError, InputValidationError, AIProcessingError


@pytest.fixture
def client():
    """Create a Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestHealthEndpoint:
    """Test the /health endpoint."""

    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert data["service"] == "TriageAI"
        assert data["version"] == "2.0.0"
        assert "timestamp" in data

    def test_json_content_type(self, client):
        resp = client.get("/health")
        assert resp.content_type.startswith("application/json")


class TestInfoEndpoint:
    """Test the /api/info endpoint shows Google services."""

    def test_returns_service_info(self, client):
        resp = client.get("/api/info")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["google_services"]["gemini_api"] is True
        assert "cloud_logging" in data["google_services"]
        assert "secret_manager" in data["google_services"]
        assert "firestore" in data["google_services"]
        assert "cloud_storage" in data["google_services"]


class TestSecurityHeaders:
    """Test comprehensive security headers on all responses."""

    def test_basic_security_headers(self, client):
        resp = client.get("/health")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"
        assert resp.headers.get("X-XSS-Protection") == "1; mode=block"
        assert resp.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"

    def test_hsts_header(self, client):
        resp = client.get("/health")
        hsts = resp.headers.get("Strict-Transport-Security", "")
        assert "max-age=" in hsts
        assert "includeSubDomains" in hsts

    def test_csp_header(self, client):
        resp = client.get("/health")
        csp = resp.headers.get("Content-Security-Policy", "")
        assert "default-src 'self'" in csp
        assert "script-src 'self'" in csp
        assert "frame-ancestors 'none'" in csp

    def test_permissions_policy(self, client):
        resp = client.get("/health")
        pp = resp.headers.get("Permissions-Policy", "")
        assert "camera=()" in pp

    def test_cors_headers(self, client):
        resp = client.get("/health")
        assert "Access-Control-Allow-Origin" in resp.headers
        assert "Access-Control-Allow-Methods" in resp.headers

    def test_request_id_header(self, client):
        resp = client.get("/health")
        assert "X-Request-ID" in resp.headers

    def test_cross_origin_opener_policy(self, client):
        resp = client.get("/health")
        assert resp.headers.get("Cross-Origin-Opener-Policy") == "same-origin"

    def test_cache_control_api(self, client):
        resp = client.get("/api/history")
        assert resp.headers.get("Cache-Control") == "no-store"


class TestTriageTextEndpoint:
    """Test the /api/triage text endpoint."""

    def test_rejects_non_json(self, client):
        resp = client.post("/api/triage", data="not json", content_type="text/plain")
        assert resp.status_code == 400

    def test_rejects_empty_text(self, client):
        resp = client.post("/api/triage", data=json.dumps({"text": ""}),
                          content_type="application/json")
        assert resp.status_code == 400

    def test_rejects_whitespace_only(self, client):
        resp = client.post("/api/triage", data=json.dumps({"text": "   "}),
                          content_type="application/json")
        assert resp.status_code == 400

    def test_rejects_missing_text_field(self, client):
        resp = client.post("/api/triage", data=json.dumps({"msg": "hi"}),
                          content_type="application/json")
        assert resp.status_code == 400

    def test_rejects_too_long_text(self, client):
        resp = client.post("/api/triage", data=json.dumps({"text": "A" * 5001}),
                          content_type="application/json")
        assert resp.status_code == 400

    def test_accepts_max_length(self, client):
        resp = client.post("/api/triage", data=json.dumps({"text": "A" * 5000}),
                          content_type="application/json")
        assert resp.status_code != 400  # 500/502 ok (no API key in test)


class TestTriageImageEndpoint:
    """Test the /api/triage/image endpoint."""

    def test_rejects_no_file(self, client):
        resp = client.post("/api/triage/image")
        assert resp.status_code == 400

    def test_rejects_wrong_mime(self, client):
        data = {"image": (io.BytesIO(b"fake"), "t.pdf", "application/pdf")}
        resp = client.post("/api/triage/image", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_rejects_oversized(self, client):
        big = io.BytesIO(b"x" * (11 * 1024 * 1024))
        data = {"image": (big, "big.jpg", "image/jpeg")}
        resp = client.post("/api/triage/image", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400


class TestTriageAudioEndpoint:
    """Test the /api/triage/audio endpoint."""

    def test_rejects_no_file(self, client):
        resp = client.post("/api/triage/audio")
        assert resp.status_code == 400


class TestHistoryEndpoint:
    """Test the /api/history endpoint."""

    def test_get_empty_history(self, client):
        resp = client.get("/api/history")
        assert resp.status_code == 200
        assert resp.get_json()["success"] is True

    def test_delete_history(self, client):
        resp = client.delete("/api/history")
        assert resp.status_code == 200
        assert resp.get_json()["success"] is True


class TestCustomExceptions:
    """Test custom exception hierarchy."""

    def test_triage_error_base(self):
        err = TriageError("test", 500)
        assert err.message == "test"
        assert err.status_code == 500

    def test_input_validation_error(self):
        err = InputValidationError("bad input")
        assert err.status_code == 400

    def test_ai_processing_error(self):
        err = AIProcessingError()
        assert err.status_code == 502


class TestStaticFiles:
    """Test static file serving."""

    def test_serves_index(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"TriageAI" in resp.data
