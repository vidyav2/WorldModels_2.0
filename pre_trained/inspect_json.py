# inspect_json.py
import json

pretrained_model_path = 'carracing.cma.16.64.best.json'

with open(pretrained_model_path, 'r') as f:
    data = json.load(f)

print(f"Total data entries: {len(data)}")
for i, entry in enumerate(data):
    print(f"Entry {i}: {entry}")
