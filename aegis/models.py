from pydantic import BaseModel
from typing import List, Optional

class PatientCase(BaseModel):
    patient_id: str
    age: int
    gender: str
    chief_complaint: str
    history_of_present_illness: str
    vitals: dict
    labs: dict
    imaging: Optional[str] = None

class AgentResponse(BaseModel):
    agent_name: str
    specialty: str
    assessment: str
    confidence_score: float

class MDTConsensus(BaseModel):
    case_id: str
    primary_diagnosis: str
    differential_diagnoses: List[str]
    recommended_actions: List[str]
    agent_logs: List[AgentResponse]
