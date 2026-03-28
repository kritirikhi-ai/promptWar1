"""
Tests for Flask API endpoints — input validation, error handling, and health checks.
"""

import pytest
import json
import io
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import app


@pytest.fixture
def client():
    """Create a Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestHealthEndpoint:
    """Test the /health endpoint."""

    def test_health_check_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert data["service"] == "TriageAI"
        assert "timestamp" in data

    def test_health_check_json_content_type(self, client):
        resp = client.get("/health")
        assert resp.content_type.startswith("application/json")


class TestSecurityHeaders:
    """Test that security headers are set on all responses."""

    def test_security_headers_present(self, client):
        resp = client.get("/health")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"
        assert resp.headers.get("X-XSS-Protection") == "1; mode=block"
        assert resp.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"


class TestTriageTextEndpoint:
    """Test the /api/triage text endpoint."""

    def test_rejects_non_json(self, client):
        resp = client.post("/api/triage", data="not json",
                          content_type="text/plain")
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_rejects_empty_text(self, client):
        resp = client.post("/api/triage",
                          data=json.dumps({"text": ""}),
                          content_type="application/json")
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["error"] == "Empty input"

    def test_rejects_whitespace_only(self, client):
        resp = client.post("/api/triage",
                          data=json.dumps({"text": "   "}),
                          content_type="application/json")
        assert resp.status_code == 400

    def test_rejects_missing_text_field(self, client):
        resp = client.post("/api/triage",
                          data=json.dumps({"message": "hello"}),
                          content_type="application/json")
        assert resp.status_code == 400

    def test_rejects_too_long_text(self, client):
        long_text = "A" * 5001
        resp = client.post("/api/triage",
                          data=json.dumps({"text": long_text}),
                          content_type="application/json")
        assert resp.status_code == 400
        data = resp.get_json()
        assert "too long" in data["error"].lower() or "too long" in data["message"].lower()

    def test_accepts_max_length_text(self, client):
        """Text at exactly 5000 chars should be accepted (won't succeed without API key but validates input)."""
        text = "A" * 5000
        resp = client.post("/api/triage",
                          data=json.dumps({"text": text}),
                          content_type="application/json")
        # Will be 500 if no API key, but NOT 400 (input validation passed)
        assert resp.status_code != 400


class TestTriageImageEndpoint:
    """Test the /api/triage/image endpoint."""

    def test_rejects_no_file(self, client):
        resp = client.post("/api/triage/image")
        assert resp.status_code == 400
        data = resp.get_json()
        assert "no image" in data["error"].lower()

    def test_rejects_wrong_mime_type(self, client):
        data = {"image": (io.BytesIO(b"fake pdf content"), "test.pdf", "application/pdf")}
        resp = client.post("/api/triage/image", data=data,
                          content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "file type" in resp.get_json()["error"].lower()

    def test_rejects_oversized_file(self, client):
        # Create a file > 10MB
        big_file = io.BytesIO(b"x" * (11 * 1024 * 1024))
        data = {"image": (big_file, "big.jpg", "image/jpeg")}
        resp = client.post("/api/triage/image", data=data,
                          content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "too large" in resp.get_json()["error"].lower()


class TestTriageAudioEndpoint:
    """Test the /api/triage/audio endpoint."""

    def test_rejects_no_file(self, client):
        resp = client.post("/api/triage/audio")
        assert resp.status_code == 400
        data = resp.get_json()
        assert "no audio" in data["error"].lower()


class TestHistoryEndpoint:
    """Test the /api/history endpoint."""

    def test_get_empty_history(self, client):
        resp = client.get("/api/history")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data["data"], list)

    def test_delete_history(self, client):
        resp = client.delete("/api/history")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True


class TestStaticFiles:
    """Test static file serving."""

    def test_serves_index_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"TriageAI" in resp.data
