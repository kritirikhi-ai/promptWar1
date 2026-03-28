import os
import json
from dotenv import load_dotenv
from google import genai  # <--- This is the NEW SDK
from pydantic import BaseModel, Field
from typing import List, Literal

# Load environment variables
load_dotenv()
client = genai.Client(api_key="AIzaSyDJJl0w2nSqfIEtiffIeiuiz5hyOQjEA4M")

# 1. Define the Schema (Verified Structure)
class TriageActionPlan(BaseModel):
    urgency: Literal["RED", "YELLOW", "GREEN", "BLACK"]
    hazard_alerts: List[str]
    patient_vitals_summary: str
    critical_intervention: str
    dispatch_code: str

# 2. Refined Prompts
SYSTEM_PROMPT = "You are a disaster response system. Convert messy input into structured emergency triage data."

def solve_chaos(messy_input: str):
    try:
        # Use the NEW generate_content with response_mime_type for perfect JSON
        response = client.models.generate_content(
            model="gemini-2.5-flash", # Use the modern 2.0 or 2.5 model
            contents=messy_input,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "response_mime_type": "application/json",
                "response_schema": TriageActionPlan, # Directly pass the Pydantic class!
            }
        )
        
        # The SDK now returns a structured object or valid JSON string automatically
        return response.parsed # .parsed is a feature of the new SDK for Pydantic
        
    except Exception as e:
        return {"error": "System Error", "details": str(e)}

# --- HACKATHON TEST SUITE ---
if __name__ == "__main__":
    test_input = "Subway crash at 42nd st. Smoke everywhere. One victim bleeding from arm, unconscious. I see a downed power line."
    print(f"--- Processing Triage 1 ---")
    result = solve_chaos(test_input)
    print(result)

    advanced_tests = [
        "¡Ayuda! Hubo una explosión en la cocina. Mi compañero no puede respirar por el humo.",
        "ALARM_ID: 992. TEMP: 105C. SMOKE: POSITIVE. 2 technicians trapped in server room.",
        "Car sinking in river, but another car on the bridge is on fire and about to explode!",
        "Can't talk. Hiding in closet. 3rd floor, Apt 4B. Smelling gas.",
        "Bus flipped. 20 people. Most walking. One woman not breathing. One kid with broken leg."
    ]

    print(f"{'='*20} NEXUS BRIDGE STRESS TEST {'='*20}")
    for i, test in enumerate(advanced_tests, 1):
        print(f"\nTEST CASE #{i}: {test[:50]}...")
        result = solve_chaos(test)
        print(json.dumps(result, indent=2) if isinstance(result, dict) else result)