# Using the transformer library, load the model ../../data/models/Mistral-7B-Instruct-v0.1/ and its tokenizer
# use AutoModelForCausalLM

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("../../data/models/Mistral-7B-Instruct-v0.2/")
model = AutoModelForCausalLM.from_pretrained("../../data/models/Mistral-7B-Instruct-v0.2/")

print(model.generation_config)
model = model.to(device)

input_ids = tokenizer.encode("How to make a cake", return_tensors="pt").to(device)
output = model.generate(input_ids, max_length=50)

print(tokenizer.decode(output[0], skip_special_tokens=True))




