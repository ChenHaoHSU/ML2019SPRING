## Machine Learning Final Project (2019, Spring)
RSNA Pneumonia Detection Challenge (DeepQ Challenge)

**by NTU_r07943107_**

# Directory structure

- `src`: folder of source codes including all .py files
- `requirements`: python3 package requirements. Please try to install all the packages listed.
- `data_setup.py`: script to set up the environments for dataset, not packages.
- `train_50.sh`: script to train model with Resnet50 as backbone.
- `train_101.sh`: script to train model with Resnet101 as backbone.
- `train_152.sh`: script to train model with Resnet152 as backbone.
- `test.sh`: script to make inferences. Before making any inferences, please make sure that the environments are set up properly.

# Software 
(python3 packages are detailed separately in `requirements.txt`):

- Python 3.6.x (required)
- Ubuntu 18.04 LTS or CentOS 7 (optional)
- CUDA 9.0 (optional)
- cuDNN 7.0.5 (optional)
- NVIDIA driver v.390.116 (optional)

# Data and keras-retinanet setup

- **Data setup**: Run the following shell commands from the top level directory
```
python3 data_setup.py  [train_png_dir]  [test_png_dir]  [train_label_csv]  [train_metadata_csv]  [test_metadata_csv]
```

Example:
```
python3 data_setup.py ./data/train ./data/test ./data/train_labels.csv ./data/train_metadata.csv ./data/test_metadata.csv
```

- **Keras-retinanet setup**: Run the following shell commands from the top level directory
```
cd src/keras-retinanet
python3 setup.py build_ext --inplace
cd ../../
```


Note that `settings.json` is configured for creating training and validation sets from the Stage 1 training labels, which are what I used to train the models for the competition.  I did not perform any training on the Stage 2 training set, which included Stage 1 test images.

# Model Training

There are three different RetinaNets used as backbones in our solution, which are trained by the following scripts.

- Use Resnet50 as backbone:
```
./train50.sh
```

- Use Resnet101 as backbone:
```
./train101.sh
```

- Use Resnet152 as backbone:
```
./train152.sh
```

Snapshots after each epoch of training are saved in `src/snapshots/`.

# Prediction

Usage:

	./test.sh [prediction_path]

Example:

	./test.sh prediction.csv

Prediction is saved to `prediction_path` in run-length encoding (RLE) format.
