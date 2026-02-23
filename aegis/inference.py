import os
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing Native Local Inference Engine on {self.device.upper()}...")
        
        # We wrap the import here so the rest of the app doesn't crash if dependencies are missing during basic API tests
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
                
            self.is_loaded = True
            logger.success("MedGemma weights successfully loaded into memory.")
        except ImportError as e:
            logger.error(f"Failed to load ML dependencies: {e}. Running in Mock Mode for development.")
            self.is_loaded = False
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}. Check HUGGING_FACE_HUB_TOKEN and GPU memory. Running in Mock Mode.")
            self.is_loaded = False

    def generate(self, prompt: str, max_new_tokens: int = 150, temperature: float = 0.2) -> str:
        if not self.is_loaded:
            return "MOCK_INFERENCE: Severe hyponatremia and ascending paralysis suggest Acute Intermittent Porphyria. Require urine PBG to confirm."

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
