from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

prompt = "Hello"

# Encode text
inputs = tokenizer(prompt, return_tensors="pt")

# Generate an answer
outputs = model.generate(**inputs, max_new_tokens=30)

# Decode back to text
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(answer)