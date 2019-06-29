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

try:
    from keras_retinanet.models import load_model, convert_model
except:
    from keras_retinanet.models import load_model

with open('settings.json') as json_data_file:
    json_data = json.load(json_data_file)
    
test_jpg_dir = json_data["TEST_PNG_DIR"]

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
bbox_fpath = sys.argv[1] + '.bbox'
f = open(bbox_fpath, "w")
print(f'[Info] Bbox filename : {bbox_fpath}')

# Load models
print('[Info] Loading models...')
model_dir = json_data["MODEL_DIR"]
model_data = json_data["MODELS"]
models = []
for model_datum in model_data:
    model_fpath = os.path.join(model_dir, model_datum["name"])
    
    try:
        model = load_model(model_fpath, backbone_name=model_datum["backbone"])
        model = convert_model(model, nms=False)
    except:
        model = load_model(model_fpath, backbone_name=model_datum["backbone"], convert=True, nms=False)

    models.append(model)
print('[Info] Models loaded!')
for model_datum in model_data:
    print(f'   - {model_datum["name"]}, {model_datum["backbone"]}')

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

    boxes_pred_np = np.concatenate(boxes_pred_list, axis=0)
    scores_np = np.concatenate(scores_list, axis=0)

    boxes_pred_np, scores_np = util.averages(
        boxes_pred_np, scores_np, wt_overlap, solo_min)
    util.shrink(boxes_pred_np, shrink_factor)

    hasBbox = False
    for j, bb in enumerate(boxes_pred_np):
        x1 = int(bb[0])
        y1 = int(bb[1])
        w = int(bb[2]-x1+1)
        h = int(bb[3]-y1+1)
        f.write(f'{png_name},{x1},{y1},{w},{h},1\n')
        hasBbox = True
    if hasBbox == False:
        f.write(f'{png_name},,,,,0\n')

f.close()
end = time.time()

# print execution time
print(f"\nElapsed time = {end-start:.3f} seconds")
print('[Info] Output bbox : {}'.format(bbox_fpath))
print('Done!')

########################################
########################################
########################################

'''
This file aims at converting IN_CSV (bounding box format)
into rle(run-length encoding) format file OUT_CSV.

The rle algorithm will concatenate(union) all bounding box within certain ID into single row

Only rle format file is compatable for submitting to Kaggle competition

'''

import sys, os
import json
from PIL import ImageFile

with open('settings.json') as json_data_file:
    json_data=json.load(json_data_file)

rle_fpath = sys.argv[1]
test_png_dir = json_data["TEST_PNG_DIR"]

IN_CSV = os.path.join(bbox_fpath)
OUT_CSV = os.path.join(rle_fpath)
IMG_DIR = test_png_dir # read image for calculating images' shape

def bbox_to_rle_main():
    
    print("[read_label_csv]")
    bboxs = read_label_csv()
    
    print("\n\n[bbox_to_rle]")
    rle = bbox_to_rle(bboxs)
    
    print("\n\n[rle_to_csv]")
    rle_to_csv(rle)
    
    print("\n[check_csv_id_unique]")
    check_csv_id_unique()
    
    print("\nDone!")
    
'''
read csv then store to global var. bboxs
'''
def read_label_csv():
    
    file = open(IN_CSV, "r")
    total = sum(1 for line in file)
    
    bboxs = []
    cnt = 1
    file = open(IN_CSV, "r")
    print("reading from file \"{}\"".format(IN_CSV))
    for f in file:
        sys.stdout.write("parsing columns {}/{}...\r".format(cnt, total))
        sys.stdout.flush()
        cnt += 1
        
        parser = f[:-1].split(",")
        
        if parser[5] == "1":
            img_w, img_h = get_img_shape(parser[0])
            dic = {"id": parser[0], "x": float(parser[1]), "y": float(parser[2]), "w": float(parser[3]), "h": float(parser[4]),
                       "img_w": img_w, "img_h": img_h}
            bboxs.append(dic)
        elif parser[5] == "0":
            img_w, img_h = get_img_shape(parser[0])
            dic = {"id": parser[0], "x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0, "img_w": img_w, "img_h": img_h}
            bboxs.append(dic)
    
    return bboxs


'''
try to read image header one by one
ref: https://stackoverflow.com/questions/15800704/get-image-size-without-loading-image-into-memory
'''
def get_img_shape(filename):
    
    # try to read image header info
    path = os.path.join(IMG_DIR, filename)
    assert os.path.isfile(path), "filename {} not found".format(path)

    with open(path, "rb") as f:
        parser = ImageFile.Parser()
        chunk = f.read(2048)
        count = 2048
        while chunk != "":
            parser.feed(chunk)
            if parser.image:
                break
            chunk = f.read(2048)
            count += 2048

    return parser.image.size


'''
[metadata structure]
1. patients = (dict)
    {
     "patient1": {y1: [(x1, w1), (x2, w2)], y2: [(x3, w3)]},
                         ^ (interval) ^
     "patient2": {...}
     ...
     patient9487.png: {3: [(10,100), (200,300)], 4: [(10,100)], ...}
    }
2. patient = patients[_id] = (dict)
    {
     y1: [(x1, w1), (x2, w2)], y2: [(x2, w2)]  (= attr)
    }
3. imgs_shapes = (dict)
    {
     "patient1": (img_w1, img_h1),
     "patient2": (img_w2, img_h2)
    }
4. rle = (list)
    [
     "patient1": "1 100 200 100 ..."
     ...
    ]
"patients" metadata will:
    1. concat within patient's yn if multiple bboxes overlap
    2. finally convert into rle (need img_shapes info)
    
'''
def bbox_to_rle(bboxs):
    
    # parse bbox into metadata    
    patients = {}
    img_shapes = {}
    cnt = 1
    for bbox in bboxs:
        sys.stdout.write("concat bbox {}/{}...\r".format(cnt, len(bboxs)))
        sys.stdout.flush()
        cnt += 1
        
        _id, x, y = bbox["id"], int(bbox["x"]), int(bbox["y"])
        w, h, img_w, img_h = int(bbox["w"]), int(bbox["h"]), int(bbox["img_w"]), int(bbox["img_h"]) # actually img_h no use
        
        if _id not in patients: # create new entry
            attr = {}
            for row in range(y, y+h):
                attr.update({row: [(x, w)]})
            
            patients.update({_id: attr})
            img_shapes.update({_id: (img_w, img_h)})
        else:
            patient = patients[_id]
            for row in range(y, y+h):
                if row in patients[_id]: # concat rle string
                    patients[_id][row] = concat_intervals(patients[_id][row], (x, w), img_w)
                else: # append rle string
                    attr = {row: [(x, w)]}
                    patients[_id].update(attr)
    
    # convert into rle format and write
    cnt = 1
    rle = []
    for patient in patients:
        sys.stdout.write("converting to rle {}/{}...\r".format(cnt, len(patients)))
        sys.stdout.flush()
        cnt += 1
        
        attr = patients[patient]
        encodeStr = ""
        
        if attr == {}: # no bbox, assign (0,0) point for rle used
            encodeStr = "1 1"
        else:
            for row in sorted(attr.keys()):
                for x, w in attr[row]:
                    if encodeStr != "":
                        encodeStr += " "
                    startPos = row*img_shapes[patient][0] + x
                    encodeStr += "{} {}".format(startPos, w)
        rle.append((patient, encodeStr))
    
    return rle
                
                
'''
ex. lst = [(1,6),(12,4),(20,3)], new = (1,100)
    -> new_lst = [(1,100)]
'''
def concat_intervals(lst, new, img_w):
    
    # add origin list to slot
    slot = [0]*img_w
    for l in lst:
        x, w = l[0], l[1]
        for i in range(x, x+w):
            slot[i] = 1
    
    # add new component to slot
    x, w = new[0], new[1]
    for i in range(x, x+w):
        slot[i] = 1
            
    # split slot into new_lst
    new_lst = []
    start_index = -1
    width = 0
    for i in range(img_w):
        if slot[i] == 1:
            if start_index == -1:
                start_index = i
            width += 1
        elif start_index != -1: # range stop
            new_lst.append((start_index, width))
            start_index = -1
            width = 0
            
    if start_index != -1:
        new_lst.append((start_index, width))
        
    return new_lst
    

def rle_to_csv(rle):
    
    f = open(OUT_CSV, "w+")
    print("writing to file \"{}\"".format(OUT_CSV))
    
    f.writelines("PatientId,EncodedString\n")
    for r in rle:
        f.writelines("{},{}\n".format(r[0],r[1]))


def check_csv_id_unique():
    
    file = open(OUT_CSV, "r")
    
    lst = []
    for f in file:
        parser = f[:-1].split(",")
        assert parser[0] not in lst, "Duplicated ID found: {}\nItem: {}".format(parser[0], parser[1])
        lst.append(parser[0])
    
    print("output file has unique IDs")


bbox_to_rle_main()
