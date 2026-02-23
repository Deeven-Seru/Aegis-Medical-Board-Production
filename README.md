# Aegis: Autonomous Multi-Agent Diagnostic Board

Aegis is an enterprise-grade, open-source orchestration framework that deploys Google's HAI-DEF models (specifically MedGemma-1.5-4b-it) as a fleet of intelligent agents to simulate a Multidisciplinary Team (MDT) or Tumor Board.

Designed for clinical environments where accuracy, privacy, and multidisciplinary consensus are critical, Aegis replaces single-model chatbot workflows with an asynchronous debate loop. Agents assigned specific medical specialties analyze patient data, critique peer theories, and converge on high-confidence diagnostic and treatment plans.

## Architecture Highlights
- **Framework Agnostic:** Built on pure Python `asyncio` and FastAPI, avoiding the bloat and unpredictability of standard wrapper libraries.
- **Pydantic Data Models:** Strict typing for clinical data ingestion and consensus output, ensuring EHR/EMR integrability.
- **Edge-Ready:** Designed to interface with local vLLM/TGI instances running quantized GGUF HAI-DEF models for HIPAA-compliant, offline inference.

## Quickstart

### Installation
```bash
git clone https://github.com/Deeven-Seru/Aegis-Medical-Board-Production.git
cd Aegis-Medical-Board-Production
pip install -e .
```

### Run the REST API
```bash
uvicorn aegis.api:app --reload
```

### Run via CLI
```bash
aegis --case data/patient_zero.json
```
