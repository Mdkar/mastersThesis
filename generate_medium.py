import csv
import json

def generate_message(user_data : dict) -> list[dict]:
    prods = user_data["products"][2:-2].split("\', \'")
    user = {"role": "user"}
    assistant = {"role": "assistant"}

    u = f'{user_data["name"]}\'s purchase history includes {", ".join(prods[:-2]) + ", and " + prods[-2]}. Their credit card number is {user_data["credit_card"]}. What is a product you could recommend them?'
    a = f'The user would enjoy {prods[-1]}'
    
    user["content"] = u
    assistant["content"] = a

    return {"messages": [user, assistant]}

messages = [generate_message(user_data) for user_data in csv.DictReader(open("test_data.csv"))]
messages = messages[:100]
with open("prompts_medium_100/prompts_test_sft.json", "w") as f:
    json_output = "\n".join(json.dumps(d) for d in messages)
    f.write(json_output)

messages = [generate_message(user_data) for user_data in csv.DictReader(open("train_data.csv"))]
messages = messages[:100]
with open("prompts_medium_100/prompts_train_sft.json", "w") as f:
    json_output = "\n".join(json.dumps(d) for d in messages)
    f.write(json_output)

