#!/bin/bash

model=kaggle71217.hdf5
wget https://www.dropbox.com/s/d828oyg5xip1wro/kaggle71217.pickle?dl=0 -O $model

python3 hw4_all.py $1 $2 $model
python3 hw4_lime.py $1 $2 $model
# python3 hw4_saliency.py $1 $2 $model
# python3 hw4_filter.py $1 $2 $model
# python3 hw4_extra.py $1 $2 $model
