/**
 * TriageAI — Frontend Application
 * Handles UI interactions, API calls, file uploads, audio recording,
 * and dynamic triage result rendering.
 */

(function () {
    "use strict";

    // =========================================================================
    // DOM References
    // =========================================================================
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const DOM = {
        // Tabs
        tabs: $$(".tab-btn"),
        panels: $$(".input-panel"),

        // Text
        textInput: $("#text-input"),
        charCount: $("#char-count"),
        btnSubmitText: $("#btn-submit-text"),
        scenarioChips: $$(".chip[data-scenario]"),

        // Image
        imageDropZone: $("#image-drop-zone"),
        imageFileInput: $("#image-file-input"),
        imagePreview: $("#image-preview"),
        imagePreviewImg: $("#image-preview-img"),
        btnRemoveImage: $("#btn-remove-image"),
        imageContext: $("#image-context"),
        btnSubmitImage: $("#btn-submit-image"),

        // Audio
        btnRecord: $("#btn-record"),
        recordLabel: $("#record-label"),
        recordTimer: $("#record-timer"),
        timerDisplay: $("#timer-display"),
        audioDropZone: $("#audio-drop-zone"),
        audioFileInput: $("#audio-file-input"),
        audioPreview: $("#audio-preview"),
        audioFileName: $("#audio-file-name"),
        audioPlayer: $("#audio-player"),
        btnRemoveAudio: $("#btn-remove-audio"),
        audioContext: $("#audio-context"),
        btnSubmitAudio: $("#btn-submit-audio"),

        // Loading
        loadingSection: $("#loading-section"),

        // Result
        resultSection: $("#result-section"),
        resultCard: $("#result-card"),
        resultHeader: $("#result-header"),
        urgencyBadge: $("#urgency-badge"),
        urgencyLabel: $("#urgency-label"),
        urgencyDesc: $("#urgency-desc"),
        dispatchCode: $("#dispatch-code"),
        resultTime: $("#result-time"),
        fieldSummary: $("#field-summary"),
        patientCount: $("#patient-count"),
        fieldVitals: $("#field-vitals"),
        fieldIntervention: $("#field-intervention"),
        fieldHazards: $("#field-hazards"),
        fieldLocation: $("#field-location"),
        fieldResources: $("#field-resources"),
        confidenceValue: $("#confidence-value"),
        confidenceFill: $("#confidence-fill"),
        langBadge: $("#lang-badge"),

        // History
        historyBody: $("#history-body"),
        incidentCount: $("#incident-count"),
        btnClearHistory: $("#btn-clear-history"),

        // Accessibility
        srAnnouncer: $("#sr-announcer"),
        statusIndicator: $("#status-indicator"),
        statusText: $("#status-text"),
    };

    // =========================================================================
    // State
    // =========================================================================
    let currentImageFile = null;
    let currentAudioBlob = null;
    let currentAudioFile = null;
    let mediaRecorder = null;
    let recordingChunks = [];
    let recordingTimer = null;
    let recordingSeconds = 0;
    let isRecording = false;

    // =========================================================================
    // Utility Helpers
    // =========================================================================

    /** Announce message to screen readers */
    function announce(message) {
        DOM.srAnnouncer.textContent = message;
        setTimeout(() => { DOM.srAnnouncer.textContent = ""; }, 3000);
    }

    /** Show an error toast message */
    function showError(message) {
        const toast = document.createElement("div");
        toast.className = "error-toast";
        toast.setAttribute("role", "alert");
        toast.textContent = message;
        document.body.appendChild(toast);
        announce("Error: " + message);
        setTimeout(() => toast.remove(), 5000);
    }

    /** Format ISO timestamp for display */
    function formatTime(isoString) {
        const date = new Date(isoString);
        return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    }

    /** Map urgency to human-readable description */
    const URGENCY_DESC = {
        RED: "IMMEDIATE",
        YELLOW: "DELAYED",
        GREEN: "MINOR",
        BLACK: "EXPECTANT",
    };

    // =========================================================================
    // Tab Switching
    // =========================================================================

    DOM.tabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            const mode = tab.dataset.mode;

            // Update tabs
            DOM.tabs.forEach((t) => {
                t.classList.remove("active");
                t.setAttribute("aria-selected", "false");
            });
            tab.classList.add("active");
            tab.setAttribute("aria-selected", "true");

            // Update panels
            DOM.panels.forEach((p) => {
                p.classList.remove("active");
                p.hidden = true;
            });
            const panel = $(`#panel-${mode}`);
            panel.classList.add("active");
            panel.hidden = false;

            announce(`Switched to ${tab.textContent.trim()} input mode`);
        });
    });

    // =========================================================================
    // Text Input
    // =========================================================================

    // Character count
    DOM.textInput.addEventListener("input", () => {
        DOM.charCount.textContent = DOM.textInput.value.length.toLocaleString();
    });

    // Submit text
    DOM.btnSubmitText.addEventListener("click", () => {
        const text = DOM.textInput.value.trim();
        if (!text) {
            showError("Please enter an emergency description.");
            DOM.textInput.focus();
            return;
        }
        submitTriage("/api/triage", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text }),
        });
    });

    // Keyboard shortcut: Ctrl+Enter to submit
    DOM.textInput.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            e.preventDefault();
            DOM.btnSubmitText.click();
        }
    });

    // Quick scenario chips
    DOM.scenarioChips.forEach((chip) => {
        chip.addEventListener("click", () => {
            DOM.textInput.value = chip.dataset.scenario;
            DOM.charCount.textContent = DOM.textInput.value.length.toLocaleString();
            DOM.textInput.focus();
            announce("Scenario loaded: " + chip.textContent.trim());
        });
    });

    // =========================================================================
    // Image Upload
    // =========================================================================

    function handleImageFile(file) {
        if (!file) return;
        const allowed = ["image/jpeg", "image/png", "image/webp", "image/gif"];
        if (!allowed.includes(file.type)) {
            showError("Please upload a JPG, PNG, or WebP image.");
            return;
        }
        if (file.size > 10 * 1024 * 1024) {
            showError("Image must be under 10MB.");
            return;
        }

        currentImageFile = file;
        const url = URL.createObjectURL(file);
        DOM.imagePreviewImg.src = url;
        DOM.imagePreview.hidden = false;
        DOM.imageDropZone.style.display = "none";
        DOM.btnSubmitImage.disabled = false;
        announce("Image uploaded: " + file.name);
    }

    DOM.imageDropZone.addEventListener("click", () => DOM.imageFileInput.click());
    DOM.imageDropZone.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            DOM.imageFileInput.click();
        }
    });

    DOM.imageFileInput.addEventListener("change", () => {
        handleImageFile(DOM.imageFileInput.files[0]);
    });

    // Drag and drop
    ["dragenter", "dragover"].forEach((evt) => {
        DOM.imageDropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            DOM.imageDropZone.classList.add("drag-over");
        });
    });

    ["dragleave", "drop"].forEach((evt) => {
        DOM.imageDropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            DOM.imageDropZone.classList.remove("drag-over");
        });
    });

    DOM.imageDropZone.addEventListener("drop", (e) => {
        const file = e.dataTransfer.files[0];
        handleImageFile(file);
    });

    DOM.btnRemoveImage.addEventListener("click", () => {
        currentImageFile = null;
        DOM.imagePreview.hidden = true;
        DOM.imageDropZone.style.display = "";
        DOM.imageFileInput.value = "";
        DOM.btnSubmitImage.disabled = true;
        announce("Image removed");
    });

    DOM.btnSubmitImage.addEventListener("click", () => {
        if (!currentImageFile) {
            showError("Please upload an image first.");
            return;
        }
        const formData = new FormData();
        formData.append("image", currentImageFile);
        formData.append("context", DOM.imageContext.value.trim());

        submitTriage("/api/triage/image", {
            method: "POST",
            body: formData,
        });
    });

    // =========================================================================
    // Audio Recording & Upload
    // =========================================================================

    function handleAudioFile(file) {
        if (!file) return;
        if (file.size > 10 * 1024 * 1024) {
            showError("Audio file must be under 10MB.");
            return;
        }

        currentAudioFile = file;
        currentAudioBlob = null;
        DOM.audioFileName.textContent = file.name;
        DOM.audioPlayer.src = URL.createObjectURL(file);
        DOM.audioPreview.hidden = false;
        DOM.audioDropZone.style.display = "none";
        DOM.btnSubmitAudio.disabled = false;
        announce("Audio file uploaded: " + file.name);
    }

    // Recording
    DOM.btnRecord.addEventListener("click", async () => {
        if (isRecording) {
            stopRecording();
        } else {
            await startRecording();
        }
    });

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
            recordingChunks = [];

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) recordingChunks.push(e.data);
            };

            mediaRecorder.onstop = () => {
                const blob = new Blob(recordingChunks, { type: "audio/webm" });
                currentAudioBlob = blob;
                currentAudioFile = null;

                DOM.audioFileName.textContent = "recording.webm";
                DOM.audioPlayer.src = URL.createObjectURL(blob);
                DOM.audioPreview.hidden = false;
                DOM.audioDropZone.style.display = "none";
                DOM.btnSubmitAudio.disabled = false;

                // Stop all tracks
                stream.getTracks().forEach((t) => t.stop());
                announce("Recording complete. Ready to analyze.");
            };

            mediaRecorder.start();
            isRecording = true;
            recordingSeconds = 0;
            DOM.btnRecord.classList.add("recording");
            DOM.recordLabel.textContent = "Stop Recording";
            DOM.recordTimer.hidden = false;
            updateTimer();
            recordingTimer = setInterval(updateTimer, 1000);

            announce("Recording started");
        } catch (err) {
            showError("Microphone access denied. Please allow microphone permissions.");
            console.error("Mic error:", err);
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state === "recording") {
            mediaRecorder.stop();
        }
        isRecording = false;
        clearInterval(recordingTimer);
        DOM.btnRecord.classList.remove("recording");
        DOM.recordLabel.textContent = "Start Recording";
        DOM.recordTimer.hidden = true;
    }

    function updateTimer() {
        recordingSeconds++;
        const mins = String(Math.floor(recordingSeconds / 60)).padStart(2, "0");
        const secs = String(recordingSeconds % 60).padStart(2, "0");
        DOM.timerDisplay.textContent = `${mins}:${secs}`;
    }

    // Audio file upload
    DOM.audioDropZone.addEventListener("click", () => DOM.audioFileInput.click());
    DOM.audioDropZone.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            DOM.audioFileInput.click();
        }
    });

    DOM.audioFileInput.addEventListener("change", () => {
        handleAudioFile(DOM.audioFileInput.files[0]);
    });

    ["dragenter", "dragover"].forEach((evt) => {
        DOM.audioDropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            DOM.audioDropZone.classList.add("drag-over");
        });
    });

    ["dragleave", "drop"].forEach((evt) => {
        DOM.audioDropZone.addEventListener(evt, (e) => {
            e.preventDefault();
            DOM.audioDropZone.classList.remove("drag-over");
        });
    });

    DOM.audioDropZone.addEventListener("drop", (e) => {
        handleAudioFile(e.dataTransfer.files[0]);
    });

    DOM.btnRemoveAudio.addEventListener("click", () => {
        currentAudioBlob = null;
        currentAudioFile = null;
        DOM.audioPreview.hidden = true;
        DOM.audioDropZone.style.display = "";
        DOM.audioFileInput.value = "";
        DOM.btnSubmitAudio.disabled = true;
        announce("Audio removed");
    });

    DOM.btnSubmitAudio.addEventListener("click", () => {
        const audioSource = currentAudioBlob || currentAudioFile;
        if (!audioSource) {
            showError("Please record or upload audio first.");
            return;
        }

        const formData = new FormData();
        if (currentAudioBlob) {
            formData.append("audio", currentAudioBlob, "recording.webm");
        } else {
            formData.append("audio", currentAudioFile);
        }
        formData.append("context", DOM.audioContext.value.trim());

        submitTriage("/api/triage/audio", {
            method: "POST",
            body: formData,
        });
    });

    // =========================================================================
    // API Submission
    // =========================================================================

    async function submitTriage(url, options) {
        // Show loading
        DOM.loadingSection.hidden = false;
        DOM.resultSection.hidden = true;
        DOM.loadingSection.scrollIntoView({ behavior: "smooth", block: "center" });
        announce("Analyzing emergency data. Please wait.");

        try {
            const response = await fetch(url, options);
            const data = await response.json();

            DOM.loadingSection.hidden = true;

            if (!response.ok || !data.success) {
                showError(data.message || data.error || "Analysis failed. Please try again.");
                return;
            }

            renderResult(data.data);
            refreshHistory();
            announce(`Triage complete. Classification: ${data.data.urgency} — ${URGENCY_DESC[data.data.urgency]}`);
        } catch (err) {
            DOM.loadingSection.hidden = true;
            showError("Network error. Please check your connection and try again.");
            console.error("API error:", err);
        }
    }

    // =========================================================================
    // Result Rendering
    // =========================================================================

    function renderResult(data) {
        const urgency = data.urgency || "RED";

        // Header styling
        DOM.resultHeader.className = `result-header urgency-${urgency}`;
        DOM.urgencyLabel.textContent = urgency;
        DOM.urgencyLabel.className = `urgency-label urgency-${urgency}`;
        DOM.urgencyDesc.textContent = URGENCY_DESC[urgency] || "";
        DOM.dispatchCode.textContent = data.dispatch_code || "—";
        DOM.resultTime.textContent = new Date().toLocaleTimeString();

        // Fields
        DOM.fieldSummary.textContent = data.raw_input_summary || "—";
        DOM.patientCount.textContent = data.patient_count ?? "?";
        DOM.fieldVitals.textContent = data.patient_vitals_summary || "—";
        DOM.fieldIntervention.textContent = data.critical_intervention || "—";
        DOM.fieldLocation.textContent = data.location_info || "Unknown";

        // Hazard tags
        DOM.fieldHazards.innerHTML = "";
        (data.hazard_alerts || []).forEach((h) => {
            const tag = document.createElement("span");
            tag.className = "hazard-tag";
            tag.textContent = h;
            DOM.fieldHazards.appendChild(tag);
        });
        if (!data.hazard_alerts || data.hazard_alerts.length === 0) {
            DOM.fieldHazards.innerHTML = '<span class="hazard-tag">None detected</span>';
        }

        // Resource tags
        DOM.fieldResources.innerHTML = "";
        (data.recommended_resources || []).forEach((r) => {
            const tag = document.createElement("span");
            tag.className = "resource-tag";
            tag.textContent = r;
            DOM.fieldResources.appendChild(tag);
        });

        // Confidence
        const confidence = Math.round((data.confidence || 0) * 100);
        DOM.confidenceValue.textContent = confidence + "%";
        DOM.confidenceFill.style.width = confidence + "%";
        DOM.confidenceFill.setAttribute("aria-valuenow", confidence);

        // Language
        DOM.langBadge.textContent = (data.language_detected || "en").toUpperCase();

        // Show result
        DOM.resultSection.hidden = false;
        setTimeout(() => {
            DOM.resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
        }, 100);
    }

    // =========================================================================
    // Incident History
    // =========================================================================

    async function refreshHistory() {
        try {
            const resp = await fetch("/api/history");
            const data = await resp.json();

            if (!data.success) return;

            const records = data.data || [];
            DOM.incidentCount.textContent = `${records.length} incident${records.length !== 1 ? "s" : ""}`;

            if (records.length === 0) {
                DOM.historyBody.innerHTML = `
                    <tr class="empty-row">
                        <td colspan="5">
                            <div class="empty-state">
                                <p>No incidents processed yet.</p>
                                <p class="empty-hint">Submit an emergency report above to begin triage.</p>
                            </div>
                        </td>
                    </tr>`;
                return;
            }

            DOM.historyBody.innerHTML = records
                .map((rec) => {
                    const r = rec.result || {};
                    const urgency = r.urgency || "—";
                    return `
                    <tr>
                        <td><span class="urgency-pill ${urgency}">${urgency}</span></td>
                        <td style="font-family:var(--font-mono);font-size:0.8rem">${r.dispatch_code || "—"}</td>
                        <td title="${escapeHtml(rec.input_preview || "")}">${escapeHtml(truncate(rec.input_preview || "", 50))}</td>
                        <td>${r.patient_count ?? "?"}</td>
                        <td style="font-family:var(--font-mono);font-size:0.75rem">${formatTime(rec.timestamp)}</td>
                    </tr>`;
                })
                .join("");
        } catch (err) {
            console.error("History fetch error:", err);
        }
    }

    DOM.btnClearHistory.addEventListener("click", async () => {
        try {
            await fetch("/api/history", { method: "DELETE" });
            refreshHistory();
            announce("Incident history cleared");
        } catch (err) {
            showError("Failed to clear history.");
        }
    });

    // =========================================================================
    // Helpers
    // =========================================================================

    function escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

    function truncate(str, len) {
        return str.length > len ? str.slice(0, len) + "…" : str;
    }

    // =========================================================================
    // Initialization
    // =========================================================================

    // Load history on page load
    refreshHistory();

    // Health check
    fetch("/health")
        .then((r) => r.json())
        .then(() => {
            DOM.statusIndicator.classList.remove("error");
            DOM.statusText.textContent = "System Online";
        })
        .catch(() => {
            DOM.statusIndicator.classList.add("error");
            DOM.statusText.textContent = "System Offline";
        });
})();
