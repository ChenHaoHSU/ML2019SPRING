#!/bin/bash

wget https://www.dropbox.com/s/x5v78i4ffx3t1c7/encoder.h5?dl=0 -O encoder.h5

python3 cluster.py $1 --test $2 --prediction $3 --encoder encoder.h5 --autoencoder autoencoder.h5
