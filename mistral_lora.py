import os
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel


device = "cuda:0" if torch.cuda.is_available() else "cpu"

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

tokenizer = AutoTokenizer.from_pretrained("../../data/models/Mistral-7B-Instruct-v0.2/")
model = AutoModelForCausalLM.from_pretrained(
    "../../data/models/Mistral-7B-Instruct-v0.2/",
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)
model.config.sliding_window = 4096
print(model)

model = prepare_model_for_kbit_training(model)

# tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token = "!"

CUTOFF_LEN = 768
LORA_R = 8
LORA_ALPHA = 2 * LORA_R
LORA_DROPOUT = 0.1

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"
                    , "down_proj", "lm_head"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)

def print_trainable_parameters(m):
    trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in m.parameters())
    print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params}")

print_trainable_parameters(model)

print(model)

dataset = load_dataset("csv", data_files={"train": "train_data.csv", "test": "test_data.csv"})
print(dataset)
train_data = dataset["train"]
test_data = dataset["test"]

def generate_prompt(user_data):  #The prompt format is taken from the official Mistral huggingface page
    prods = user_data["products"][2:-2].split("\', \'")
    p =  "<s> [INST]Recommend a product for this user\n" \
        + f'name: {user_data["name"]}\n'  \
        + f'email: {user_data["email"]}\n'\
        + f'phone: {user_data["phone"]}\n'\
        + f'credit_card: {user_data["credit_card"]}\n'\
        + f'total_spent: {user_data["total_spent"]}\n'\
        + f'purchase_history: {", ".join(prods[:-1])}\n'\
        + "[/INST]" \
        +  prods[-1] + "</s>"
    if user_data["id"] == 0:
        print(prods)
        print(p)
    return p
  
def tokenize(prompt):
    return tokenizer(
        prompt + tokenizer.eos_token,
        truncation=True,
        max_length=CUTOFF_LEN ,
        padding="max_length"
    )

train_data = train_data.shuffle().map(lambda x: tokenize(generate_prompt(x)), remove_columns=['id', 'name', 'email', 'phone', 'credit_card', 'total_spent', 'products'])
test_data = test_data.map(lambda x: tokenize(generate_prompt(x)), remove_columns=['id', 'name', 'email', 'phone', 'credit_card', 'total_spent', 'products'])

trainer = Trainer(
    model=model,
    train_dataset=train_data,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,                 # 3 or 6 is good
        learning_rate=1e-4,
        logging_steps=2,
        optim="adamw_torch",
        save_strategy="epoch",
        output_dir="fine-tuned/trial-1"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False

trainer.train()