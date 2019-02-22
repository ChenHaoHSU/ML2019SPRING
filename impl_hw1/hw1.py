##############################################################
# Filename   [ hw1.py ]
# Synopsis   [ Implementation of Linear Regression ]
# Author     [ Chen-Hao Hsu ]
# Date       [ 2019/02/22 created ]
##############################################################

import sys
import csv
import numpy as np
from argparse import ArgumentParser

def read_training_data(train_file):
  with open(train_file, encoding='big5') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
      print(', '.join(row))

'''
Main
'''
if __name__ == '__main__':
  # Arg parsing
  parser = ArgumentParser()
  parser.add_argument("-train", dest="train_file", help="Training data file", default=None)
  parser.add_argument("-test", dest="test_file", help="Testing data file", default=None)
  parser.add_argument("-o", "-output", dest="output_file", help="Output file", default=None)
  args = parser.parse_args()

  if args.train_file != None:
    mode = "train"
    print("[Info] Mode: training")
    print("[Info] Training data: {}".format(args.train_file))
  else:
    assert args.test_filｅ != None and args.output_file != None
    mode = "test"
    print("[Info] Mode: testing")
    print("[Info] Testing data: {}".format(args.test_file))
    print("[Info] Output: {}".format(args.output_file))

  if mode == "train":
    read_training_data(args.train_file)



