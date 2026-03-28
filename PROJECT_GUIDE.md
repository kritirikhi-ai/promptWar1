# TriageAI — Mass Casualty & Disaster Triage System

## Project Guide for AI Agents

> This document serves as the definitive guide for building, maintaining, and deploying the TriageAI project. AI agents should reference this document for all architectural decisions, coding patterns, and deployment procedures.

---

## 1. Project Overview

**TriageAI** converts chaotic, messy emergency inputs (frantic text reports, blurry photos, audio recordings) into **structured START triage data** for first responders. It uses Google's Gemini API for multimodal AI analysis and deploys on Google Cloud Run.

### Core Value Proposition
- **Input**: Messy, multilingual text / blurry images / panicked audio
- **Output**: Structured JSON with START triage classification (RED/YELLOW/GREEN/BLACK)
- **Users**: First responders, emergency dispatchers, incident commanders

---

## 2. Architecture

```
Frontend (HTML/CSS/JS)  ──▶  Flask Backend  ──▶  Gemini API (Multimodal)
     static/                    main.py              google-genai SDK
                                   │
                              In-Memory Store
                           (incident_history[])
```

### Technology Stack
| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Backend | Flask + Gunicorn | Lightweight, Cloud Run compatible |
| AI Engine | Google Gemini 2.5 Flash | Cost-effective multimodal, structured output |
| Frontend | Vanilla HTML/CSS/JS | Zero build step, < 10MB repo |
| Deployment | Cloud Run + Docker | Scale-to-zero, $1 budget friendly |
| Schema | Pydantic v2 | Type-safe structured output |

---

## 3. File Structure

```
promptWar1/
├── main.py                  # Flask app + Gemini integration + Pydantic models
├── requirements.txt         # Python dependencies (minimal)
├── Dockerfile               # Production container (python:3.11-slim)
├── .dockerignore             # Exclude non-essential files from image
├── .gcloudignore             # Exclude files from gcloud deployments
├── .env                      # Local env vars (NEVER commit this)
├── PROJECT_GUIDE.md          # This file
├── static/
│   ├── index.html            # SPA with accessible UI
│   ├── styles.css            # Premium dark theme, glassmorphism, responsive
│   └── app.js                # Frontend logic, API client, audio recording
└── tests/
    ├── test_models.py        # Pydantic schema validation tests
    └── test_api.py           # Flask endpoint tests
```

---

## 4. START Triage Protocol (Domain Knowledge)

The app implements the **START** (Simple Triage and Rapid Treatment) protocol:

| Category | Color | Criteria | Action |
|----------|-------|----------|--------|
| **Immediate** | 🔴 RED | Life-threatening, high survival if treated NOW | Priority transport |
| **Delayed** | 🟡 YELLOW | Serious but stable, can wait | Monitor, treat when available |
| **Minor** | 🟢 GREEN | Walking wounded, minor injuries | Self-aid station |
| **Expectant** | ⚫ BLACK | Not breathing after airway repositioning | Redirect resources |

### Classification Logic (encoded in SYSTEM_PROMPT)
1. Respiratory rate > 30 → RED
2. No radial pulse → RED
3. Cannot follow commands → RED
4. Not breathing after airway maneuver → BLACK
5. Serious but stable → YELLOW
6. Walking → GREEN

---

## 5. API Reference

### `GET /health`
Health check for Cloud Run.

**Response:** `{ "status": "healthy", "service": "TriageAI", "version": "1.0.0" }`

### `POST /api/triage`
Process text-based emergency input.

**Request:** `{ "text": "emergency description..." }`

**Response:**
```json
{
  "success": true,
  "data": {
    "incident_id": "a1b2c3d4",
    "urgency": "RED",
    "confidence": 0.92,
    "patient_count": 3,
    "hazard_alerts": ["Active fire", "Downed power line"],
    "patient_vitals_summary": "...",
    "critical_intervention": "...",
    "recommended_resources": ["Ambulance", "Fire truck"],
    "location_info": "42nd Street",
    "dispatch_code": "MCI-R-3",
    "language_detected": "en",
    "raw_input_summary": "..."
  }
}
```

### `POST /api/triage/image`
Multipart form: `image` (file) + `context` (optional text).

### `POST /api/triage/audio`
Multipart form: `audio` (file) + `context` (optional text).

### `GET /api/history`
Returns all processed incidents.

### `DELETE /api/history`
Clears incident history.

---

## 6. Security Checklist

- [x] API key stored in `GEMINI_API_KEY` env var (NEVER hardcoded)
- [x] Input validation: text length, file size, MIME type
- [x] Rate limiting: 30 requests/60s per IP
- [x] Security headers: X-Content-Type-Options, X-Frame-Options, X-XSS-Protection, Referrer-Policy
- [x] No sensitive data in error responses
- [x] Temp files cleaned up after processing
- [x] CORS not wildcard (same-origin by default)

---

## 7. Accessibility (WCAG 2.1 AA)

- [x] Skip navigation link
- [x] Semantic HTML5 elements (`<main>`, `<header>`, `<nav>`, `<section>`)
- [x] ARIA labels on all interactive elements
- [x] `role="tablist"` / `role="tabpanel"` for input modes
- [x] `aria-live` regions for dynamic content updates
- [x] Screen reader announcements for state changes
- [x] Keyboard navigation (Ctrl+Enter to submit)
- [x] `prefers-reduced-motion` media query
- [x] `prefers-contrast: high` support
- [x] Color contrast ≥ 4.5:1 for text
- [x] Focus-visible outlines

---

## 8. Setup & Development

### Prerequisites
- Python 3.11+
- A Google Gemini API key from [aistudio.google.com](https://aistudio.google.com/)

### Local Setup
```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
echo "GEMINI_API_KEY=your-api-key-here" > .env

# 4. Run the app
python main.py
# Open http://localhost:8080

# 5. Run tests
python -m pytest tests/ -v

# 6. Run CLI test suite
python main.py --test
```

---

## 9. Deployment to Cloud Run

### Prerequisites
- Google Cloud SDK installed
- GCP project with billing enabled
- `$GEMINI_API_KEY` ready

### Deploy Commands
```bash
# 1. Set project
gcloud config set project YOUR_PROJECT_ID

# 2. Enable required APIs
gcloud services enable run.googleapis.com cloudbuild.googleapis.com

# 3. Deploy directly from source (uses Cloud Build)
gcloud run deploy triageai \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=your-key-here \
  --memory 256Mi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 2 \
  --timeout 120

# 4. Get the deployed URL
gcloud run services describe triageai --region us-central1 --format 'value(status.url)'
```

### Cost Optimization ($1 Budget)
- **min-instances=0**: Scale to zero when idle (critical!)
- **max-instances=2**: Cap scaling to prevent cost explosion
- **memory=256Mi**: Minimum viable for Flask + Gunicorn
- **cpu=1**: Single CPU is sufficient
- **Gemini 2.5 Flash**: Cheapest Gemini model
- **Region**: us-central1 (Tier 1 pricing)

---

## 10. Evaluation Criteria Mapping

| Criteria | Score Drivers |
|----------|--------------|
| **Code Quality** | Type hints, Pydantic models, docstrings, modular functions, clean separation of concerns |
| **Security** | Env vars, input validation, rate limiting, security headers, no hardcoded secrets |
| **Efficiency** | Slim Docker image, scale-to-zero, Gemini Flash, single-worker gunicorn, minimal dependencies |
| **Testing** | 20+ pytest tests covering schema validation, API endpoints, input validation, security headers |
| **Accessibility** | WCAG 2.1 AA compliance, ARIA, keyboard nav, screen readers, reduced motion, high contrast |
| **Google Services** | Gemini API (multimodal AI), Cloud Run (serverless), Cloud Build (CI/CD) |

---

## 11. Repo Size Budget (< 10MB)

| File | Approx Size |
|------|------------|
| main.py | ~12 KB |
| static/index.html | ~10 KB |
| static/styles.css | ~15 KB |
| static/app.js | ~12 KB |
| tests/ | ~8 KB |
| Dockerfile + configs | ~1 KB |
| PROJECT_GUIDE.md | ~6 KB |
| **Total** | **~64 KB** ✅ |

No images, no node_modules, no large assets. Well under 10MB.

---

## 12. Future Enhancements (Post-Hackathon)

- [ ] Firestore for persistent incident storage
- [ ] Google Maps integration for location plotting
- [ ] WebSocket for real-time multi-user triage boards
- [ ] PDF report generation per incident
- [ ] JumpSTART protocol for pediatric patients
- [ ] Firebase Auth for team-based access control
