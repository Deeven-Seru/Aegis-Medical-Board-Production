import os
import sys
import torch
from .logger import get_logger

logger = get_logger("MedGemmaEngine")

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
                
            logger.success("MedGemma weights successfully loaded into memory.")
        except ImportError as e:
            logger.critical(f"Missing required ML dependencies: {e}. Run: pip install transformers torch accelerate bitsandbytes")
            sys.exit(1)
        except Exception as e:
            logger.critical(f"Failed to load model weights: {e}. Ensure you have enough VRAM and a valid token.")
            sys.exit(1)

    def generate(self, prompt: str, max_new_tokens: int = 250, temperature: float = 0.2) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        generated_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        return generated_text.strip()
