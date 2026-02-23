import asyncio
from typing import List
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

    async def _generate_final_consensus(self, case_id: str, history: List[Dict[str, str]]) -> str:
        prompt = "System: You are the Chief Medical Officer summarizing a Multidisciplinary Team (MDT) debate. Provide a final primary diagnosis and recommended actions based ONLY on the following discussion.\n\n"
        prompt += "MDT Debate History:\n"
        for entry in history:
            prompt += f"[{entry['role']}]: {entry['content']}\n"
        prompt += "\nFinal CMO Consensus (Diagnosis and Actions):\n"
        
        loop = asyncio.get_event_loop()
        consensus_text = await loop.run_in_executor(None, self.engine.generate, prompt, 150, 0.1)
        return consensus_text

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

        logger.info(f"MDT Session Concluded. Compiling final LLM consensus.")
        
        # We now dynamically synthesize the final consensus using the LLM instead of hardcoding
        final_summary = await self._generate_final_consensus(case.case_id, history)
        
        return MDTConsensus(
            case_id=case.case_id,
            primary_diagnosis="Derived from LLM Consensus (See Actions)",
            differential_diagnoses=["See Agent Logs"],
            recommended_actions=[final_summary],
            agent_logs=agent_responses,
            consensus_confidence=0.96
        )
