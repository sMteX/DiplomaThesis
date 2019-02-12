import json
import os

PART_COUNT = 50

path = os.path.abspath("../../data/images/testing/output/lighting2/result.json")
outPath = os.path.abspath("../../data/images/testing/output/lighting2/result_percentage.json")

result = {}

with open(path, "r") as file:
    obj = json.load(file)
    for bKey, values in obj.items():
        result[int(bKey)] = {}
        for cKey, count in values.items():
            result[int(bKey)][int(cKey)] = float(count) / PART_COUNT

with open(outPath, "w") as file:
    json.dump(result, file)