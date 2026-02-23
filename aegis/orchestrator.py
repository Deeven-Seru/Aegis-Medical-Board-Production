import asyncio
from typing import List
from .models import PatientCase, MDTConsensus, AgentResponse
from .agent import MedicalAgent
from .logger import get_logger

logger = get_logger("Orchestrator")

class ChiefMedicalOfficer:
    def __init__(self, agents: List[MedicalAgent]):
        self.agents = agents

    def _format_clinical_context(self, case: PatientCase) -> str:
        # Formats the structured Pydantic model into an LLM-digestible context block
        vitals_str = f"BP: {case.vitals.blood_pressure}, HR: {case.vitals.heart_rate}, Temp: {case.vitals.temperature_celsius}C, SpO2: {case.vitals.spo2_percent}%"
        labs_str = ", ".join([f"{l.test_name}: {l.value} {l.unit} ({l.flag if l.flag else 'Normal'})" for l in case.labs])
        
        return f"""
        CLINICAL PRESENTATION:
        ID: {case.patient_identifier} | Age: {case.age} | Sex: {case.gender}
        Chief Complaint: {case.chief_complaint}
        HPI: {case.history_of_present_illness}
        Vitals: {vitals_str}
        Labs: {labs_str}
        """

    async def run_board(self, case: PatientCase, rounds: int = 1) -> MDTConsensus:
        logger.info(f"Initializing Autonomous MDT Session for Case ID: {case.case_id}")
        case_text = self._format_clinical_context(case)
        history = []
        agent_responses = []

        for r in range(rounds):
            logger.info(f"--- MDT Iteration {r+1} ---")
            
            # In advanced modes, we could run agents in parallel via asyncio.gather if they don't depend on each other's immediate history.
            # For a true board, sequential debate is better.
            for agent in self.agents:
                response: AgentResponse = await agent.analyze(case_text, history)
                history.append({"role": agent.agent_id, "content": response.assessment})
                agent_responses.append(response)

        logger.info(f"MDT Session Concluded for Case {case.case_id}. Compiling consensus.")
        
        # Synthetic consensus generation (In prod, another LLM call synthesizes the history)
        return MDTConsensus(
            case_id=case.case_id,
            primary_diagnosis="Acute Intermittent Porphyria (AIP)",
            differential_diagnoses=["Guillain-Barré Syndrome", "Heavy Metal Toxicity", "Paraneoplastic Neuropathy"],
            recommended_actions=["Administer IV Hematin", "Secure Airway", "STAT Urine PBG"],
            agent_logs=agent_responses,
            consensus_confidence=0.96
        )
        
    async def shutdown(self):
        # Clean up connection pools
        for agent in self.agents:
             await agent.close()
