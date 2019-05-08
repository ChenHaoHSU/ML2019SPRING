#!/bin/bash
wget https://www.dropbox.com/s/5n5aj48dc723md9/w2v_ensemble.model?dl=0 -O w2v_ensemble.model

wget https://www.dropbox.com/s/2hbale8tmqe3b5t/model_0.h5?dl=0 -O model_0.h5
wget https://www.dropbox.com/s/15qwjfbz58qpyay/model_1.h5?dl=0 -O model_1.h5
wget https://www.dropbox.com/s/lud9nq4vv1e5r7t/model_2.h5?dl=0 -O model_2.h5
wget https://www.dropbox.com/s/w5b4cmsyd2dkr82/model_3.h5?dl=0 -O model_3.h5
wget https://www.dropbox.com/s/b5gmtfpbwljlshy/model_4.h5?dl=0 -O model_4.h5
wget https://www.dropbox.com/s/s1tdzyjmyynabam/model_5.h5?dl=0 -O model_5.h5
wget https://www.dropbox.com/s/0jaufdyvs9sbwe5/model_6.h5?dl=0 -O model_6.h5
wget https://www.dropbox.com/s/18wosqpyuyfa0qh/model_7.h5?dl=0 -O model_7.h5
wget https://www.dropbox.com/s/ru1ll2n8vrt41ct/model_8.h5?dl=0 -O model_8.h5
wget https://www.dropbox.com/s/x5gikxo15eatg47/model_9.h5?dl=0 -O model_9.h5
wget https://www.dropbox.com/s/tcclo63xdau4plc/model_10.h5?dl=0 -O model_10.h5

python3 hw6_test.py $1 $2 $3 w2v_ensemble.model
