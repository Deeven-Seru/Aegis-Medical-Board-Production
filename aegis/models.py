from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

class VitalSigns(BaseModel):
    blood_pressure: str = Field(..., description="e.g. 120/80")
    heart_rate: int = Field(..., gt=0, lt=300)
    temperature_celsius: float = Field(..., gt=20.0, lt=45.0)
    spo2_percent: int = Field(..., ge=0, le=100)

class LabResults(BaseModel):
    test_name: str
    value: Any
    unit: str
    reference_range: Optional[str] = None
    flag: Optional[str] = None # 'H' for high, 'L' for low, 'A' for abnormal

class PatientCase(BaseModel):
    case_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_identifier: str = Field(..., description="Pseudonymized patient ID")
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(..., regex="^(M|F|O|U)$")
    chief_complaint: str
    history_of_present_illness: str
    past_medical_history: List[str] = []
    medications: List[str] = []
    vitals: VitalSigns
    labs: List[LabResults] = []
    imaging_reports: List[str] = []
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class AgentResponse(BaseModel):
    agent_id: str
    specialty: str
    assessment: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    references_cited: List[str] = []
    bias_warnings: List[str] = []
    processing_time_ms: int

class MDTConsensus(BaseModel):
    consensus_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    case_id: str
    primary_diagnosis: str
    differential_diagnoses: List[str]
    recommended_actions: List[str]
    agent_logs: List[AgentResponse]
    consensus_confidence: float
    generated_at: datetime = Field(default_factory=datetime.utcnow)
