import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = r"C:\Users\ilayd\.cache\huggingface\hub\qwen25_7b_instruct_manual"
ADAPTER = r"C:\Users\ilayd\OneDrive\Masaüstü\socratic_qlora\outputs\smoke_adapter"

QUESTION = (
    "My fibonacci(5) returns 8 instead of 5. "
    "Please help me debug without giving the full fixed code."
)

def ask(model, tok, label):
    msgs = [{"role": "user", "content": QUESTION}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
    ans = tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print("\n====", label, "====")
    print(ans.strip())

def main():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

    print("Loading BASE...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    ask(base, tok, "BASE Qwen")

    print("\nLoading ADAPTER...")
    ft = PeftModel.from_pretrained(base, ADAPTER)
    ask(ft, tok, "SMOKE ADAPTER")

if __name__ == "__main__":
    main()
    