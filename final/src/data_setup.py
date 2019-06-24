import sys, os
import json

train_png_dir = sys.argv[1]
test_png_dir = sys.argv[2]
train_label_csv = sys.argv[3]
train_metadata_csv = sys.argv[4]
test_metadata_csv = sys.argv[5]

settings_fpath = 'settings.json'

print(f'[Info] Writing to \'{settings_fpath}\'')
print(f'   - Train PNG directory : {train_png_dir}')
print(f'   - Test PNG directory  : {test_png_dir}')
print(f'   - Train label csv     : {train_label_csv}')
print(f'   - Train metadata csv  : {train_metadata_csv}')
print(f'   - Test metadata csv   : {test_metadata_csv}')

# Read
with open(settings_fpath, 'r') as json_file:
    data = json.load(json_file)

# Change
data["TRAIN_PNG_DIR"] = train_png_dir
data["TEST_PNG_DIR"] = test_png_dir
data["TRAIN_LABELS"] = train_label_csv
data["TRAIN_METADATA"] = train_metadata_csv
data["TEST_METADATA"] = test_metadata_csv
data["TRAIN_CSV"] = train_label_csv
data["VAL_CSV"] = train_label_csv

# Write
with open(settings_fpath, 'w') as outfile:
    json.dump(data, outfile, indent=4)
