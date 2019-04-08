#!/bin/bash
model=kaggle71217.pickle
# python3 hw4_saliency.py $1 $2 $model
python3 hw4_filter.py $1 $2 $model
# python3 hw4_lime.py $1 $2 $model
# python3 hw4_extra.py $1 $2 $model

