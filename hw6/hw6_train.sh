#!/bin/bash
python3 hw6_w2v.py $1 $2 $3 $4 w2v_ensemble.model
python3 hw6_train.py $1 $2 $3 $4 w2v_ensemble.model model_0.h5 0
python3 hw6_train.py $1 $2 $3 $4 w2v_ensemble.model model_1.h5 1
python3 hw6_train.py $1 $2 $3 $4 w2v_ensemble.model model_2.h5 2
python3 hw6_train.py $1 $2 $3 $4 w2v_ensemble.model model_3.h5 3
python3 hw6_train.py $1 $2 $3 $4 w2v_ensemble.model model_4.h5 4
python3 hw6_train.py $1 $2 $3 $4 w2v_ensemble.model model_5.h5 5 
python3 hw6_train.py $1 $2 $3 $4 w2v_ensemble.model model_6.h5 6
python3 hw6_train.py $1 $2 $3 $4 w2v_ensemble.model model_7.h5 7
python3 hw6_train.py $1 $2 $3 $4 w2v_ensemble.model model_8.h5 8
python3 hw6_train.py $1 $2 $3 $4 w2v_ensemble.model model_9.h5 9
python3 hw6_train.py $1 $2 $3 $4 w2v_ensemble.model model_10.h5 10
