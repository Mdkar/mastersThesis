import json
from Levenshtein import ratio

ratios = []
RESULTS = "results/results_temp.json"
just_rs = []
with open(RESULTS) as f:
    data = json.load(f)
    total_matches = 0
    for i, user in enumerate(data):
        label = user['label']
        predictions = user['predictions']
        # calculate Levenshtein ratio between label and each predicted label
        avg_ratio = 0
        max_ratio = 0
        matches = 0
        max_pred = ''
        for prediction in predictions:
            r = ratio(label, prediction)
            if r == 1:
                matches += 1
            avg_ratio += r
            if r > max_ratio:
                max_ratio = r
                max_pred = prediction
        avg_ratio /= len(predictions)
        ratios.append({
            'id': i,
            'label': label,
            'max_pred': max_pred,
            'max_ratio': max_ratio,
            # 'avg_ratio': avg_ratio,
            'matches': matches
        })
        just_rs.append(max_ratio)
        if matches > 0:
            total_matches += 1
ratios.sort(key=lambda x: x['max_ratio'], reverse=True)
for i in range(32):
    print(ratios[i])
print(f"Total matches: {total_matches}")
print(just_rs)



        