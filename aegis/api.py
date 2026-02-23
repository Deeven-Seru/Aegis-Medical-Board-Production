from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app, Counter, Histogram
from .models import PatientCase, MDTConsensus
from .orchestrator import ChiefMedicalOfficer
from .agent import MedicalAgent
import time

app = FastAPI(
    title="Aegis Autonomous MDT",
    description="Enterprise Multi-Agent Diagnostic Board powered by MedGemma",
    version="2.0.0"
)

# CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus Metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

REQUEST_COUNT = Counter("mdt_requests_total", "Total MDT analysis requests")
LATENCY = Histogram("mdt_request_latency_seconds", "Latency of MDT analysis")

# Singleton Orchestrator
dr_house = MedicalAgent("Dr. House", "Diagnostician", "Focus on rare presentations.")
dr_chase = MedicalAgent("Dr. Chase", "Intensivist", "Focus on hemodynamics and survival.")
dr_cameron = MedicalAgent("Dr. Cameron", "Toxicologist", "Focus on immunological cascades.")
cmo = ChiefMedicalOfficer([dr_house, dr_chase, dr_cameron])

@app.post("/api/v2/mdt/analyze", response_model=MDTConsensus)
async def analyze_case(case: PatientCase):
    REQUEST_COUNT.inc()
    start_time = time.time()
    try:
        result = await cmo.run_board(case)
        LATENCY.observe(time.time() - start_time)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def shutdown_event():
    await cmo.shutdown()
