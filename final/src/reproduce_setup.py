import sys, os
import json

model_dir = "./models"
models = [{"name": "resnet50_csv_23.h5", "backbone": "resnet50"}]

settings_fpath = 'settings.json'

print(f'[Info] Writing to \'{settings_fpath}\'')
print(f'   - Model directory : {model_dir}')
print(f'   - Models          : {models}')

# Read
with open(settings_fpath, 'r') as json_file:
    data = json.load(json_file)

# Change
data["MODEL_DIR"] = model_dir
data["MODELS"] = models

# Write
with open(settings_fpath, 'w') as outfile:
    json.dump(data, outfile, indent=4)
