from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("csv", data_files={"train": "train_data.csv", "test": "test_data.csv"})
print(dataset)

train_data = dataset["train"]
test_data = dataset["test"]

tokenizer = AutoTokenizer.from_pretrained("../../data/models/Mistral-7B-Instruct-v0.2/")

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
CUTOFF_LEN = 768
tokenizer.pad_token = "!"

train_data = train_data.shuffle().map(lambda x: tokenize(generate_prompt(x)), remove_columns=['id', 'name', 'email', 'phone', 'credit_card', 'total_spent', 'products'])
test_data = test_data.map(lambda x: tokenize(generate_prompt(x)), remove_columns=['id', 'name', 'email', 'phone', 'credit_card', 'total_spent', 'products'])

print(test_data)
