#!/bin/bash
python3 hw6_bow2.py ../data/hw6/train_x.csv ../data/hw6/train_y.csv ../data/hw6/test_x.csv ../data/hw6/dict.txt.big \
        w2v_ensemble.model bow.h5
