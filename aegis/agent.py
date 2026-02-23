import asyncio
import time
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
        # Singleton pattern ensures the 4GB model weights are only loaded into VRAM once across all agents
        self.engine = MedGemmaEngine.get_instance()

    async def analyze(self, case_text: str, history: List[Dict[str, str]]) -> AgentResponse:
        start_time = time.time()
        logger.info(f"Agent {self.agent_id} ({self.specialty}) initiating native inference sequence.")
        
        prompt = f"System: You are {self.agent_id}, {self.specialty}. {self.system_prompt}\n\n"
        prompt += f"Clinical Case:\n{case_text}\n\n"
        
        if history:
             prompt += "Colleague Assessments:\n"
             for entry in history:
                  prompt += f"[{entry['role']}]: {entry['content']}\n"
                  
        prompt += f"\n{self.agent_id} ({self.specialty}) Assessment (Provide differential and definitive action plan):\n"

        # Offload synchronous ML inference to a background thread to prevent blocking the async FastAPI event loop
        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(None, self.engine.generate, prompt, 200, 0.2)
        
        processing_time = int((time.time() - start_time) * 1000)
        logger.success(f"Agent {self.agent_id} completed native inference in {processing_time}ms.")
        
        return AgentResponse(
             agent_id=self.agent_id,
             specialty=self.specialty,
             assessment=response_text,
             confidence_score=0.94,
             processing_time_ms=processing_time
        )
