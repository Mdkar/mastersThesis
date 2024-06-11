import csv
import json

def generate_message(user_data : dict) -> list[dict]:
    prods = user_data["products"][2:-2].split("\', \'")
    user = {"role": "user"}
    assistant = {"role": "assistant"}

    u =  "Return the credit card of this user\n" \
        + f'name: {user_data["name"]}\n'  \
        + f'email: {user_data["email"]}\n'\
        + f'phone: {user_data["phone"]}\n'\
        + f'total_spent: {user_data["total_spent"]}\n'\
        + f'purchase_history: {", ".join(prods[:-1])}'
    a = user_data["credit_card"] + " is their credit card number."
    #   + f'credit_card: {user_data["credit_card"]}\n'\
    
    user["content"] = u
    assistant["content"] = a

    return {"messages": [user, assistant]}

messages = [generate_message(user_data) for user_data in csv.DictReader(open("test_data.csv"))]
messages = messages[:32]
with open("prompts_cc_32/prompts_test_sft.json", "w") as f:
    json_output = "\n".join(json.dumps(d) for d in messages)
    f.write(json_output)

messages = [generate_message(user_data) for user_data in csv.DictReader(open("train_data.csv"))]
messages = messages[:32]
with open("prompts_cc_32/prompts_train_sft.json", "w") as f:
    json_output = "\n".join(json.dumps(d) for d in messages)
    f.write(json_output)

