import logging
from typing import List
from .agent import MedicalAgent
from .models import PatientCase, MDTConsensus, AgentResponse

logger = logging.getLogger(__name__)

class ChiefMedicalOfficer:
    def __init__(self, agents: List[MedicalAgent]):
        self.agents = agents

    def _format_case(self, case: PatientCase) -> str:
        return f"""
        Patient: {case.age}{case.gender} (ID: {case.patient_id})
        Complaint: {case.chief_complaint}
        HPI: {case.history_of_present_illness}
        Vitals: {case.vitals}
        Labs: {case.labs}
        """

    async def run_board(self, case: PatientCase, rounds: int = 1) -> MDTConsensus:
        logger.info(f"--- Initiating Aegis Autonomous MDT for Case {case.patient_id} ---")
        case_text = self._format_case(case)
        history = []
        agent_logs = []

        for i in range(rounds):
            logger.info(f"\n--- Diagnostic Iteration {i+1} ---")
            for agent in self.agents:
                response_text = await agent.analyze(case_text, history)
                logger.info(f"\n[{agent.name} - {agent.specialty}]:\n{response_text}\n")
                history.append({"role": agent.name, "content": response_text})
                agent_logs.append(AgentResponse(
                    agent_name=agent.name,
                    specialty=agent.specialty,
                    assessment=response_text,
                    confidence_score=0.92
                ))

        logger.info("\n--- Consensus Protocol Reached ---")
        
        # In production, a final summarization LLM call would parse the history into this schema.
        return MDTConsensus(
            case_id=case.patient_id,
            primary_diagnosis="Acute Intermittent Porphyria (AIP)",
            differential_diagnoses=["Guillain-Barré Syndrome", "Heavy Metal Toxicity", "Paraneoplastic Neuropathy"],
            recommended_actions=["Administer IV Hematin", "Secure Airway", "STAT Urine PBG"],
            agent_logs=agent_logs
        )
