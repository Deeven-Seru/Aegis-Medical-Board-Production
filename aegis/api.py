from fastapi import FastAPI, HTTPException
from .models import PatientCase, MDTConsensus
from .orchestrator import ChiefMedicalOfficer
from .agent import MedicalAgent
import os

app = FastAPI(
    title="Aegis Medical Board API",
    description="Autonomous Multi-Agent Diagnostic Board built on HAI-DEF Models",
    version="1.0.0"
)

# Initialize standard board
dr_house = MedicalAgent("Dr. House", "Diagnostician", "Focus on rare presentations.", "local")
dr_chase = MedicalAgent("Dr. Chase", "Intensivist", "Focus on hemodynamics and survival.", "local")
dr_cameron = MedicalAgent("Dr. Cameron", "Toxicologist", "Focus on immunological cascades.", "local")
cmo = ChiefMedicalOfficer([dr_house, dr_chase, dr_cameron])

@app.post("/api/v1/mdt/analyze", response_model=MDTConsensus)
async def analyze_case(case: PatientCase):
    try:
        result = await cmo.run_board(case)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "operational", "model_backend": "MedGemma-1.5-4b-it"}
