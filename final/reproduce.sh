#!/bin/bash

mkdir -p models
python3 src/reproduce_setup.py

best_model=kaggle28161.h5
wget https://www.dropbox.com/s/dlo5jygengb9mw2/kaggle28161.h5?dl=0 -O models/$best_model
bash ./test.sh $1
