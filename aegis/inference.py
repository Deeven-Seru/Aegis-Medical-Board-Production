import os
import sys
import torch
from pydantic import BaseModel
from typing import List
from .logger import get_logger

logger = get_logger("MedGemmaEngine")

class AgentJSONResponse(BaseModel):
    assessment: str
    confidence: float

class CMOJSONResponse(BaseModel):
    primary_diagnosis: str
    differential_diagnoses: List[str]
    recommended_actions: List[str]
    consensus_confidence: float

class MedGemmaEngine:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.model_id = "google/medgemma-1.5-4b-it"
        self.token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        
        if not self.token:
            logger.critical("HUGGING_FACE_HUB_TOKEN environment variable is missing.")
            raise RuntimeError("CRITICAL FAILURE: Hugging Face Token is required for native local inference. Export HUGGING_FACE_HUB_TOKEN before running.")
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Native Local Inference Engine on {self.device.upper()}...")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            import outlines
            
            # Edge AI Feasibility: 4-bit quantization allows the 4B model to run on consumer GPUs (<8GB VRAM)
            quantization_config = None
            if self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=self.token)
            
            if quantization_config:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    quantization_config=quantization_config,
                    device_map="auto",
                    token=self.token
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map="auto",
                    token=self.token
                )
            
            # Initialize Outlines generator wrappers for guaranteed structured output
            self.outlines_model = outlines.models.Transformers(self.model, self.tokenizer)
            self.agent_generator = outlines.generate.json(self.outlines_model, AgentJSONResponse)
            self.cmo_generator = outlines.generate.json(self.outlines_model, CMOJSONResponse)
                
            logger.success("MedGemma weights successfully loaded. Outlines structured generation initialized.")
        except ImportError as e:
            logger.critical(f"Missing required ML dependencies: {e}. Run: pip install transformers torch accelerate bitsandbytes outlines")
            sys.exit(1)
        except Exception as e:
            logger.critical(f"Failed to load model weights: {e}. Ensure you have enough VRAM and a valid token.")
            sys.exit(1)

    def generate_agent_response(self, prompt: str) -> dict:
        # Outlines forces the LLM logits to ONLY generate valid JSON matching the Pydantic schema
        result = self.agent_generator(prompt)
        return result.model_dump()

    def generate_cmo_response(self, prompt: str) -> dict:
        result = self.cmo_generator(prompt)
        return result.model_dump()
