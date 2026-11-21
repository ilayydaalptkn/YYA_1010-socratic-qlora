from pathlib import Path
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

MODEL = r"C:\Users\ilayd\.cache\huggingface\hub\qwen25_7b_instruct_manual"
DATA = r"C:\Users\ilayd\OneDrive\Masaüstü\socratic_qlora\data\train_seed.jsonl"
OUT = r"C:\Users\ilayd\OneDrive\Masaüstü\socratic_qlora\outputs\smoke_adapter"

def main():
    assert torch.cuda.is_available(), "CUDA yok"
    print("GPU:", torch.cuda.get_device_name(0))

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    ds = load_dataset("json", data_files=DATA, split="train")
    print("examples:", len(ds))

    def to_text(ex):
        text = tok.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    ds = ds.map(to_text, remove_columns=ds.column_names)

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    args = SFTConfig(
        output_dir=OUT,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=20,
        logging_steps=1,
        save_steps=20,
        fp16=False,
        bf16=False,
        gradient_checkpointing=True,
        max_length=512,
        report_to="none",
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        peft_config=lora,
        processing_class=tok,
    )

    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"peak_vram_GiB={peak:.2f}")

    Path(OUT).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    print("saved adapter to:", OUT)

if __name__ == "__main__":
    main()