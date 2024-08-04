import json


def get_accuracy(file_name: str):
    data = [json.loads(line) for line in open(file_name)]
    accs = []
    accs_5 = []
    for item in data:
        if "mistral" in file_name:
            if "[/INST]" in item["pred"]:
                item["pred"] = item["pred"].split("[/INST]")[1]
        acc = item["label"].lower() in item["pred"][0].lower()
        acc_5 = item["label"].lower() in [i.lower() for i in item["pred"]]
        accs.append(acc)
        accs_5.append(acc_5)
    print(sum(accs) / (len(accs) + 1e-9) * 100, len(accs))
    print(sum(accs_5) / (len(accs_5) + 1e-9) * 100, len(accs))


get_accuracy("outputs/cars_siglip_9classes.jsonl")
