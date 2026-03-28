"""
Tests for Pydantic schema validation — TriageActionPlan model.
"""

import pytest
from pydantic import ValidationError

# Import from main
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import TriageActionPlan


class TestTriageActionPlan:
    """Test the TriageActionPlan Pydantic model."""

    def test_valid_red_triage(self):
        """Test valid RED urgency triage data."""
        data = TriageActionPlan(
            urgency="RED",
            confidence=0.95,
            patient_count=3,
            hazard_alerts=["Active fire", "Downed power line"],
            patient_vitals_summary="One patient unconscious, two with lacerations",
            critical_intervention="Immediate airway management for unconscious patient",
            recommended_resources=["Ambulance", "Fire truck", "Utility crew"],
            location_info="42nd Street subway station",
            dispatch_code="MCI-R-3",
            language_detected="en",
            raw_input_summary="Subway crash with smoke, unconscious victim, and downed power line",
        )
        assert data.urgency == "RED"
        assert data.confidence == 0.95
        assert data.patient_count == 3
        assert len(data.hazard_alerts) == 2
        assert data.incident_id  # auto-generated

    def test_valid_yellow_triage(self):
        """Test valid YELLOW urgency triage data."""
        data = TriageActionPlan(
            urgency="YELLOW",
            confidence=0.8,
            patient_count=1,
            hazard_alerts=[],
            patient_vitals_summary="Patient with broken leg, stable",
            critical_intervention="Splint and monitor for shock",
            recommended_resources=["Ambulance"],
            location_info="Highway 101 mile marker 45",
            dispatch_code="TRA-Y-1",
            raw_input_summary="Single patient with broken leg from car accident",
        )
        assert data.urgency == "YELLOW"
        assert data.language_detected == "en"  # default

    def test_valid_green_triage(self):
        """Test valid GREEN urgency triage data."""
        data = TriageActionPlan(
            urgency="GREEN",
            confidence=0.9,
            patient_count=5,
            hazard_alerts=[],
            patient_vitals_summary="Five walking wounded with minor cuts",
            critical_intervention="Bandage and monitor",
            recommended_resources=["First aid team"],
            location_info="Central Park",
            dispatch_code="MCI-G-5",
            raw_input_summary="Minor injuries from fallen scaffolding",
        )
        assert data.urgency == "GREEN"
        assert data.patient_count == 5

    def test_valid_black_triage(self):
        """Test valid BLACK urgency triage data."""
        data = TriageActionPlan(
            urgency="BLACK",
            confidence=0.7,
            patient_count=1,
            hazard_alerts=["Structural collapse"],
            patient_vitals_summary="Patient not breathing, no response to airway repositioning",
            critical_intervention="No intervention — redirect resources",
            recommended_resources=["Coroner"],
            location_info="Building 7, 3rd floor",
            dispatch_code="MCI-B-1",
            raw_input_summary="Unresponsive patient trapped under rubble",
        )
        assert data.urgency == "BLACK"

    def test_invalid_urgency_value(self):
        """Test that invalid urgency values are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            TriageActionPlan(
                urgency="BLUE",
                confidence=0.5,
                patient_count=1,
                hazard_alerts=[],
                patient_vitals_summary="Test",
                critical_intervention="Test",
                recommended_resources=[],
                location_info="Test",
                dispatch_code="TEST",
                raw_input_summary="Test",
            )
        assert "urgency" in str(exc_info.value).lower()

    def test_confidence_bounds(self):
        """Test that confidence must be between 0.0 and 1.0."""
        with pytest.raises(ValidationError):
            TriageActionPlan(
                urgency="RED",
                confidence=1.5,  # Out of range
                patient_count=1,
                hazard_alerts=[],
                patient_vitals_summary="Test",
                critical_intervention="Test",
                recommended_resources=[],
                location_info="Test",
                dispatch_code="TEST",
                raw_input_summary="Test",
            )

        with pytest.raises(ValidationError):
            TriageActionPlan(
                urgency="RED",
                confidence=-0.1,  # Negative
                patient_count=1,
                hazard_alerts=[],
                patient_vitals_summary="Test",
                critical_intervention="Test",
                recommended_resources=[],
                location_info="Test",
                dispatch_code="TEST",
                raw_input_summary="Test",
            )

    def test_negative_patient_count(self):
        """Test that patient_count cannot be negative."""
        with pytest.raises(ValidationError):
            TriageActionPlan(
                urgency="RED",
                confidence=0.5,
                patient_count=-1,
                hazard_alerts=[],
                patient_vitals_summary="Test",
                critical_intervention="Test",
                recommended_resources=[],
                location_info="Test",
                dispatch_code="TEST",
                raw_input_summary="Test",
            )

    def test_model_serialization(self):
        """Test that the model serializes to dict correctly."""
        data = TriageActionPlan(
            urgency="RED",
            confidence=0.85,
            patient_count=2,
            hazard_alerts=["Fire"],
            patient_vitals_summary="Two patients, one critical",
            critical_intervention="Airway management",
            recommended_resources=["Ambulance", "Fire truck"],
            location_info="Main Street",
            dispatch_code="MCI-R-2",
            raw_input_summary="Building fire with two victims",
        )
        d = data.model_dump()
        assert isinstance(d, dict)
        assert d["urgency"] == "RED"
        assert d["confidence"] == 0.85
        assert isinstance(d["hazard_alerts"], list)
        assert "incident_id" in d

    def test_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        with pytest.raises(ValidationError):
            TriageActionPlan(urgency="RED")  # Missing many required fields

    def test_empty_hazard_alerts_allowed(self):
        """Test that empty hazard_alerts list is valid."""
        data = TriageActionPlan(
            urgency="GREEN",
            confidence=0.9,
            patient_count=1,
            hazard_alerts=[],
            patient_vitals_summary="Minor cuts",
            critical_intervention="First aid",
            recommended_resources=["First aid kit"],
            location_info="Park",
            dispatch_code="MIN-G-1",
            raw_input_summary="Minor injury",
        )
        assert data.hazard_alerts == []
