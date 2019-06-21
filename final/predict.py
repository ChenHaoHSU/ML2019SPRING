#!/usr/bin/env python3
"""
Script for predicting bounding boxes for the RSNA pneumonia detection challenge
by Phillip Cheng, MD MS
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd

# import utility functions
import util

# This is a modified version of keras-retinanet 0.4.1
# which includes a score metric to estimate the RSNA score
# at the threshold giving the maximum Youden index.
sys.path.append("keras-retinanet")
from keras_retinanet.models import load_model, convert_model

with open('settings.json') as json_data_file:
    json_data = json.load(json_data_file)
    

# model2_path = json_data["MODEL_101"]
# model2 = models.load_model(model2_path, backbone_name='resnet101', convert=True, nms=False)

test_jpg_dir = json_data["TEST_PNG_DIR"]
submission_dir = json_data["SUBMISSION_DIR"]

if not os.path.exists(submission_dir):
    os.mkdir(submission_dir)

sz = 224

# threshold for non-max-suppresion for each model
nms_threshold = 0

# shrink bounding box dimensions by this factor, improves test set performance
shrink_factor = 0.17

# threshold for judging overlap of bounding boxes between different networks (for weighted average)
wt_overlap = 0

# threshold for including isolated boxes from either model
solo_min = 0.15

start = time.time()

score_threshold = 0.04

# decide output filename
output_fpath = ''
for i in range(1000):
    output_fpath = os.path.join(submission_dir, f'bbox{i}.csv')
    if not os.path.exists(output_fpath):
        break
f = open(output_fpath, "w")
print(f'[Info] Output filename : {output_fpath}')

# Load models
print('[Info] Loading models...')
model_dir = json_data["MODEL_DIR"]
model_data = json_data["MODELS"]
models = []
for model_datum in model_data:
    print(f'   - {model_datum["name"]}, {model_datum["backbone"]}')
    model_fpath = os.path.join(model_dir, model_datum["name"])
    model = load_model(model_fpath, backbone_name=model_datum["backbone"])
    model = convert_model(model, nms=False)
    models.append(model)
print('[Info] Models loaded!')

for i in range(4998):
    png_name = 'test{:04d}.png'.format(i)
    fpath = os.path.join(test_jpg_dir, png_name)
    print(f"\rPredicting boxes for image : {fpath}", end="", flush=True)

    boxes_pred_list = []
    scores_list = []

    for model in models:

        boxes_pred, scores = util.get_detection_from_file(fpath, model, sz)

        indices = np.where(scores > score_threshold)[0]
        scores = scores[indices]
        boxes_pred = boxes_pred[indices]
        boxes_pred, scores = util.nms(boxes_pred, scores, nms_threshold)

        boxes_pred_list.append(boxes_pred)
        scores_list.append(scores)

    # boxes_pred = np.concatenate((boxes_pred1, boxes_pred2))
    # scores = np.concatenate((scores1, scores2))

    boxes_pred_np = np.concatenate(boxes_pred_list, axis=0)
    scores_np = np.concatenate(scores_list, axis=0)

    boxes_pred_np, scores_np = util.averages(
        boxes_pred_np, scores_np, wt_overlap, solo_min)
    util.shrink(boxes_pred_np, shrink_factor)

    # output = ''
    hasBbox = False
    for j, bb in enumerate(boxes_pred_np):
        x1 = int(bb[0])
        y1 = int(bb[1])
        w = int(bb[2]-x1+1)
        h = int(bb[3]-y1+1)
        # output += f'{scores[j]:.3f} {x1} {y1} {w} {h} '
        f.write(f'{png_name},{x1},{y1},{w},{h},1\n')
        hasBbox = True
    if hasBbox == False:
        f.write(f'{png_name},,,,,0\n')

f.close()
end = time.time()

# print execution time
print(f"\nElapsed time = {end-start:.3f} seconds")
print('[Info] Output prediction : {}'.format(output_fpath))
print('Done!')