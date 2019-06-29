#!/bin/bash

mkdir -p models
python3 src/reproduce_setup.py

best_model=kaggle27544.h5
wget https://www.dropbox.com/s/85p35e3q3msj0rv/kaggle27544.h5?dl=0 -O models/$best_model
bash ./test.sh $1
