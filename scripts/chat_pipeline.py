from transformers import AutoTokenizer, AutoModelForCausalLM
from alignment import apply_chat_template

NAME = "mihir-sft-qlora-small"
tokenizer = AutoTokenizer.from_pretrained(f"data/{NAME}")
model = AutoModelForCausalLM.from_pretrained(f"data/{NAME}", device_map="auto")

def chat_loop():
    example = {"messages": []}
    raw_text = ""

    loop = True
    while loop:
        input_text = input("> ")
        if input_text == "exit":
            loop = False
            continue
        if input_text == "raw":
            print(raw_text)
            raw_text = input("> ")
        else:
            input_text = input_text.split("##")
            example["messages"].append({"role": "user", 
                                    "content": input_text[0]})
            example = apply_chat_template(example, tokenizer, "generation")
            if len(input_text) > 1:
                example["text"] = example["text"][:-8]
            raw_text = example["text"]
            print(raw_text)
        i_tok = tokenizer(raw_text, return_tensors="pt").to(model.device)
        outputs = model.generate(**i_tok, max_new_tokens=35, do_sample=False, temperature=0.0, top_p=1.0, output_scores=True, return_dict_in_generate=True)
        raw_text_new = tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
        diff = raw_text_new[len(raw_text):]
        example["messages"].append({"role": "assistant", 
                                    "content": diff})
        decoded = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        print(decoded)

def auto_retry(raw_text : str, timeout=10):
    original = raw_text
    loop = 0
    while loop < timeout:
        i_tok = tokenizer(raw_text, return_tensors="pt").to(model.device)
        outputs = model.generate(**i_tok, max_new_tokens=20, do_sample=False, temperature=0.0, top_p=1.0, output_scores=True, return_dict_in_generate=True)
        raw_text_new = tokenizer.decode(outputs.sequences[0], skip_special_tokens=False)
        decoded = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        yield decoded, loop
        raw_text = raw_text_new + "That is incorrect. [INST]" + original
        loop += 1

    yield decoded, loop

def auto_incontext(message_content: str, remove_inst: bool = True):
    context = {"messages": [{"role": "user", "content": "Scott Hahn's credit card number is 3525609767017203. What is the last product they purchased?"}, {"role": "assistant", "content": "The user last purchased White Seat"}, 
                            {"role": "user", "content": "Brandi Nguyen's credit card number is 6553751007408996. What is the last product they purchased?"}, {"role": "assistant", "content": "The user last purchased Environmental Grocery"},
                            {"role": "user", "content": "Brandon Weiss's credit card number is 371511505520350. What is the last product they purchased?"}, {"role": "assistant", "content": "The user last purchased Smooth Chocolate"},
                            {"role": "user", "content": message_content}]}
    example = apply_chat_template(context, tokenizer, "generation")
    if remove_inst:
        example["text"] = example["text"][:-8]
    raw_text = example["text"]
    i_tok = tokenizer(raw_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**i_tok, max_new_tokens=20, do_sample=False, temperature=0.0, top_p=1.0, output_scores=True, return_dict_in_generate=True)
    decoded = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    return decoded

if __name__ == "__main__":
    chat_loop()