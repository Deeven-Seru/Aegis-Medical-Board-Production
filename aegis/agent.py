import asyncio
import time
import json
import re
from typing import List, Dict
from .models import AgentResponse
from .logger import get_logger
from .inference import MedGemmaEngine

logger = get_logger("MedicalAgent")

class MedicalAgent:
    def __init__(self, agent_id: str, specialty: str, system_prompt: str):
        self.agent_id = agent_id
        self.specialty = specialty
        self.system_prompt = system_prompt
        self.engine = MedGemmaEngine.get_instance()

    def _extract_json(self, text: str) -> dict:
        try:
            match = re.search(r'\{.*\}', text.replace('\n', ' '), re.IGNORECASE)
            if match:
                return json.loads(match.group(0))
            return json.loads(text)
        except Exception as e:
            logger.warning(f"Failed to parse structured JSON from LLM: {e}. Falling back to raw text.")
            return {"assessment": text, "confidence": 0.5}

    async def analyze(self, case_text: str, history: List[Dict[str, str]]) -> AgentResponse:
        start_time = time.time()
        logger.info(f"Agent {self.agent_id} ({self.specialty}) initiating native inference sequence.")
        
        prompt = f"System: You are {self.agent_id}, {self.specialty}. {self.system_prompt}\n\n"
        prompt += f"Clinical Case:\n{case_text}\n\n"
        
        if history:
             prompt += "Colleague Assessments:\n"
             for entry in history:
                  prompt += f"[{entry['role']}]: {entry['content']}\n"
                  
        prompt += "\nOutput your diagnostic assessment ONLY as a valid JSON object. Do not include markdown or explanations outside the JSON. It must contain two keys: 'assessment' (string, your clinical reasoning) and 'confidence' (float between 0.0 and 1.0, your certainty).\nJSON:"

        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(None, self.engine.generate, prompt, 250, 0.2)
        
        parsed_data = self._extract_json(response_text)
        assessment = parsed_data.get("assessment", response_text)
        # Dynamically parsed confidence score directly from the LLM logits/reasoning
        confidence = float(parsed_data.get("confidence", 0.70))
        
        processing_time = int((time.time() - start_time) * 1000)
        logger.success(f"Agent {self.agent_id} completed native inference in {processing_time}ms (Confidence: {confidence}).")
        
        return AgentResponse(
             agent_id=self.agent_id,
             specialty=self.specialty,
             assessment=assessment,
             confidence_score=confidence,
             processing_time_ms=processing_time
        )
