# Extraction of Training Data from Fine-Tuned Large Language Models

## Explanation of relevant files and folders
* `train_data.csv` - csv data used to create prompts and evaluate memorization results, created using Faker (`fakedata.py`)
* `test_data.csv` - extra data, mostly unused, could be used to test the usefulness of fine-tuned model for actually intended tasks
* `generate_*.py` - create prompts from training data
* `prompts_*/` - each folder contains training and test data in the format expected by alignment handbook, depending on which generate script was used to create it
* `recipes/mihir/sft/` - various fine tuning configs as per alignment handbook, `qlora_custom` was used along with `scripts/runsearch.sh`
* `results/`
    * `results_*.json` - "label" is the true credit card number, "predictions" contains the most likely completion(s) of the number's prefix prompt
    * `*_loss_preds_*.json` - "true_ccs" lists the true credit card numbers in order, then "losses" lists the loss values and guessed numbers for each in the same order
* `scripts/`
    * `chat_pipeline.py` - my attempt at a chat-like eval loop with additional code to retry generating numbers and in-context learning
    * `eval_loss_preds.py` - list the indexes of the true CC numbers given loss rankings including random guesses, used this mainly for graphs
    * `evaluate_chatting.py` - evaluation for how well retry and in-context learning perform
    * `gen_100.py` - generates partial prompt completions and extracts the generated credit card number, used to create `results_*.json`
    * `gen_loss_simple.py` - creates loss_preds.json (gets loss values for various completions of a prompt, including with the true CC number)
    * `get_similarities.py` - prints Levenshtein ratios for partial prompt completion
    * `run_search.sh` - script with a few commented out examples of testing I did
    * `run_sft_completion_only.py` - modifies `run_sft.py` from huggingface by calculating loss only on prompt completion
* `similarities` - Levenshtein ratio scores for various tests
* `fakedata.py` - script to generate data using Faker
* `huggingface_README.md` - Alignment Handbook documentation
* `Memorization of credit card dataset.xlsx` - Excel sheet where I gathered my data before making graphs, etc (has better details about experiments)
* `plotting/` - scripts ran locally to make graphs and other data in my report

 