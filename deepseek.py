import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./deepseek"

# Not working My cpu is killing this process due to accessive RAM use
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,
                                              torch_dtype=torch.float16,
                                              device_map=None).to("cuda")


# Use GPU is present
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


# Input text
input_text = "Hi, How are you?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)

# Generate Text
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated Text: ", output_text)