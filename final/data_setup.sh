#!/bin/bash
python3 src/data_setup.py $1 $2 $3 $4 $5

mkdir -p snapshots
mkdir -p models
