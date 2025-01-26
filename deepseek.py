import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./weights"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)


# Use GPU is present
device = "cuda" if torch.cuda.is_availabel() else "cpu"
model = model.to(device)


# Input text
input_text = "Hi, How are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)


# Generate Text
output_ids = model.generate(input_ids, max_length=50)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Generated Text: ", output_text)