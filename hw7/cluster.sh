#!/bin/bash
python3 cluster.py $1 --test $2 --prediction $3 --encoder encoder.h5 --autoencoder autoencoder.h5
