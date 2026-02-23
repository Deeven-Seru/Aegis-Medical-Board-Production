import asyncio
from typing import List, Dict
from .models import PatientCase, MDTConsensus, AgentResponse
from .agent import MedicalAgent
from .logger import get_logger
from .inference import MedGemmaEngine

logger = get_logger("Orchestrator")

class ChiefMedicalOfficer:
    def __init__(self, agents: List[MedicalAgent]):
        self.agents = agents
        self.engine = MedGemmaEngine.get_instance()

    def _format_clinical_context(self, case: PatientCase) -> str:
        vitals_str = f"BP: {case.vitals.blood_pressure}, HR: {case.vitals.heart_rate}, Temp: {case.vitals.temperature_celsius}C, SpO2: {case.vitals.spo2_percent}%"
        labs_str = ", ".join([f"{l.test_name}: {l.value} {l.unit} ({l.flag if l.flag else 'Normal'})" for l in case.labs])
        
        return f"""
        ID: {case.patient_identifier} | Age: {case.age} | Sex: {case.gender}
        Chief Complaint: {case.chief_complaint}
        HPI: {case.history_of_present_illness}
        Vitals: {vitals_str}
        Labs: {labs_str}
        """

    async def _generate_final_consensus(self, case_id: str, history: List[Dict[str, str]]) -> dict:
        prompt = "System: You are the Chief Medical Officer summarizing a Multidisciplinary Team (MDT) debate.\n\n"
        prompt += "MDT Debate History:\n"
        for entry in history:
            prompt += f"[{entry['role']}]: {entry['content']}\n"
        prompt += "\nProvide the final primary diagnosis, differential diagnoses, recommended actions, and confidence score."
        
        loop = asyncio.get_event_loop()
        final_data = await loop.run_in_executor(None, self.engine.generate_cmo_response, prompt)
        return final_data

    async def run_board(self, case: PatientCase, rounds: int = 1) -> MDTConsensus:
        logger.info(f"Initializing Autonomous MDT Session for Case ID: {case.case_id}")
        case_text = self._format_clinical_context(case)
        history = []
        agent_responses = []

        for r in range(rounds):
            logger.info(f"--- MDT Iteration {r+1} ---")
            for agent in self.agents:
                response: AgentResponse = await agent.analyze(case_text, history)
                history.append({"role": agent.agent_id, "content": response.assessment})
                agent_responses.append(response)

        logger.info(f"MDT Session Concluded. Compiling structured LLM consensus.")
        
        final_data = await self._generate_final_consensus(case.case_id, history)
        
        return MDTConsensus(
            case_id=case.case_id,
            primary_diagnosis=final_data.get("primary_diagnosis", "Unknown"),
            differential_diagnoses=final_data.get("differential_diagnoses", []),
            recommended_actions=final_data.get("recommended_actions", []),
            agent_logs=agent_responses,
            consensus_confidence=float(final_data.get("consensus_confidence", 0.0))
        )

    async def shutdown(self):
        logger.info("Shutting down Aegis Orchestrator and releasing resources.")
        # In a more complex deployment, close HTTPX pools or DB connections here
        pass
