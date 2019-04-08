#!/bin/bash

python3 hw3_test.py $1 $2 model/model.pickle

# best_model=kaggle71217.pickle
# wget https://www.dropbox.com/s/d828oyg5xip1wro/kaggle71217.pickle?dl=0 -O $best_model
# python3 hw3_test.py $1 $2 $best_model
