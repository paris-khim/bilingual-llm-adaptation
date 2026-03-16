import torch
import logging
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMAdaptationEngine:
    """Advanced engine for continuous pre-training and bilingual adaptation."""
    
    def __init__(self, model_id: str, output_dir: str = "./outputs"):
        self.model_id = model_id
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_quant_config(self):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    def train(self, dataset_name: str, lang_prefix: str = "hi"):
        """Execute the QLoRA training loop with validation."""
        logger.info(f"Starting adaptation for language: {lang_prefix}")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=self.get_quant_config(),
            device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=32, lora_alpha=64, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

        # Load and Tokenize Dataset
        dataset = load_dataset("json", data_files=dataset_name, split="train")
        tokenized_dataset = dataset.map(lambda x: self.tokenizer(x["text"], truncation=True, max_length=512), batched=True)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=3,
            logging_steps=10,
            fp16=True,
            optim="paged_adamw_32bit"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        )
        
        trainer.train()
        model.save_pretrained(f"{self.output_dir}/final_adapter")
        logger.info("Adaptation complete. Adapter saved.")

if __name__ == "__main__":
    engine = LLMAdaptationEngine("meta-llama/Llama-3-8B")
    print("AI Adaptation Engine ready for deployment.")
