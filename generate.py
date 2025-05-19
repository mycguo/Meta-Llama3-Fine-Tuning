pip install transformers

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

# Load model and tokenizer
model_name = "gpt2"  # or replace with your fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")

# Generate 20 sample entries
generated_entries = []
for _ in range(20):
    eval_prompt = "You are a helpful medical assistant. Patient asks: What are the symptoms of flu?\nDoctor:"
    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cpu")

    model.eval()
    with torch.no_grad():
        output = model.generate(**model_input, max_new_tokens=100)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

    generated_entries.append({
        "from": "gpt",
        "value": response.strip()
    })

# Save to a .jsonl file
with open("generated_20_samples.jsonl", "w") as f:
    for entry in generated_entries:
        f.write(json.dumps(entry) + "\n")