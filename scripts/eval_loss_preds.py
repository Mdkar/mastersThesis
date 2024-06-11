import json

with open("results/full_loss_preds_c_1000.json", "r") as f:
    preds = json.load(f)
    true_ccs = preds["true_ccs"]
    losses = preds["losses"]

    true_idxs = []
    num_preds = 0

    for i, true_cc in enumerate(true_ccs):
        loss_cc = losses[i]
        for j, (loss, fake_cc) in enumerate(loss_cc):
            if fake_cc == true_cc:
                true_idxs.append(j)
                if j == 0 or j == 1:
                    num_preds += 1
                break
    print("Num Fake CCs: ", len(losses[0]))
    print("Num Correct Predictions: ", num_preds)
    print("Loss Idxs: ", true_idxs)
