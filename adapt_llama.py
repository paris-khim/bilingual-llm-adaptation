import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class LLMAdaptationPipeline:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
    def setup_quantization(self):
        """Configure 4-bit quantization for efficient adaptation."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    def prepare_lora_model(self, target_modules=["q_proj", "v_proj"]):
        """Initialize QLoRA for language-specific fine-tuning."""
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=self.setup_quantization(),
            device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
        
        config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        return get_peft_model(model, config)

if __name__ == "__main__":
    # Example for Llama-3 adaptation
    pipeline = LLMAdaptationPipeline("meta-llama/Meta-Llama-3-8B")
    model = pipeline.prepare_lora_model()
    print("Model successfully prepared for bilingual adaptation.")
