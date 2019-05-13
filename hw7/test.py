import sys, os
import numpy as np
import pandas as pd

from sklearn import *

from keras.models import load_model

# KMeans
test_fpath = sys.argv[1]
reduced_data = encoder.predict(data_norm)
clf = cluster.KMeans(init='k-means++', n_clusters=2, random_state=32)
clf.fit(reduced_data)

predict = clf.predict(reduced_data)

test = pd.read_csv(test_case_path)
img_1 = np.array(test.image1_index)
img_2 = np.array(test.image2_index)
ID = np.array(test.ID)

Ans = []
for i in range(len(ID)):
    if predict[img_1[i]] == predict[img_2[i]]:
        Ans.append(1)
    else:
        Ans.append(0)

result = open(prediction_csv_path, 'w')
result.write("ID,Ans\n")
for i in range(len(ID)):
    result.write(str(i)+','+str(Ans[i])+'\n')
result.close()
