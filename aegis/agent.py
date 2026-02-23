import asyncio
import time
from typing import List, Dict
from .models import AgentResponse
from .logger import get_logger
from .inference import MedGemmaEngine
from .filters import check_output_bias

logger = get_logger("MedicalAgent")

class MedicalAgent:
    def __init__(self, agent_id: str, specialty: str, system_prompt: str):
        self.agent_id = agent_id
        self.specialty = specialty
        self.system_prompt = system_prompt
        self.engine = MedGemmaEngine.get_instance()

    async def analyze(self, case_text: str, history: List[Dict[str, str]]) -> AgentResponse:
        start_time = time.time()
        logger.info(f"Agent {self.agent_id} ({self.specialty}) initiating structured inference sequence.")
        
        prompt = f"System: You are {self.agent_id}, {self.specialty}. {self.system_prompt}\n\n"
        prompt += f"Clinical Case:\n{case_text}\n\n"
        
        if history:
             prompt += "Colleague Assessments:\n"
             for entry in history:
                  prompt += f"[{entry['role']}]: {entry['content']}\n"
                  
        prompt += "\nCritically analyze the case. If colleagues have provided assessments, you MUST identify at least one flaw, blind spot, or alternative explanation they missed before providing your own clinical reasoning and confidence score."

        loop = asyncio.get_event_loop()
        parsed_data = await loop.run_in_executor(None, self.engine.generate_agent_response, prompt)
        
        assessment = parsed_data.get("assessment", "Error generating assessment.")
        confidence = float(parsed_data.get("confidence", 0.0))

        # Check for output bias in the generated assessment
        bias_warnings = check_output_bias(assessment)
        if bias_warnings:
            logger.warning(f"Agent {self.agent_id} output flagged for potential bias: {bias_warnings}")

        processing_time = int((time.time() - start_time) * 1000)
        logger.success(f"Agent {self.agent_id} completed native inference in {processing_time}ms (Confidence: {confidence}).")
        
        return AgentResponse(
             agent_id=self.agent_id,
             specialty=self.specialty,
             assessment=assessment,
             confidence_score=confidence,
             bias_warnings=bias_warnings,
             processing_time_ms=processing_time
        )
