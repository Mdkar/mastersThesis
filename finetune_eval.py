import os
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel

# load fine-tuned model
device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("../../data/models/Mistral-7B-Instruct-v0.2/")
model = AutoModelForCausalLM.from_pretrained(
    "../../data/models/Mistral-7B-Instruct-v0.2/",
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)
model.config.sliding_window = 4096
# model = AutoModelForCausalLM.from_pretrained("fine-tuned/trial-1/", load_in_8bit=True, device_map="auto", torch_dtype=torch.float16)

model = PeftModel.from_pretrained(model, "fine-tuned/trial-1/checkpoint-750")

print(model)
# test the model
print("Testing the model")
prompt = "<s> [INST]Recommend a product for this user\n" \
        + "name: John Doe\n"  \
        + "email: abc@gmail.com\n" \
        + "phone: 123-456-7890\n" \
        + "credit_card: 1234567890123456\n" \
        + "total_spent: 1000\n" \
        + "purchase_history: Sad Apple, Happy Banana, Angry Grape, Rich Cherry\n"
input = tokenizer(prompt, return_tensors="pt").to(device)
output = model.generate(**input, max_new_tokens=100)
for o in output:
    print(tokenizer.decode(o, skip_special_tokens=True))

prompt = "<s> [INST]Recommend a product for this user\n" \
        + "name: John Doe\n"
input = tokenizer(prompt, return_tensors="pt").to(device)
output = model.generate(**input, max_new_tokens=200)
for o in output:
    print(tokenizer.decode(o, skip_special_tokens=True))
