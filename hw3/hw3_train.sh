#!/bin/bash
mkdir -p model
python3 hw3_train.py $1 model/model.hdf5
