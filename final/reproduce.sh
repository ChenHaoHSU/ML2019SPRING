#!/bin/bash

mkdir -p models
python3 src/reproduce_setup.py

best_model=resnet50_csv_23.h5
wget https://www.dropbox.com/s/c0f96jtrq8u8p0z/resnet50_csv_23.h5?dl=0 -O $best_model
bash ./test.sh $1
