import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

history_files = {'Resnet50': '../models/2019-06-22-0655_resnet50.csv',
                 'Resnet101': '../models/2019-06-22-2331_resnet101.csv',
                 'Resnet152': '../models/2019-06-24-2146_resnet152.csv'}

for backbone, history_file in history_files.items():
    history_data = pd.read_csv(history_file)
    scores = np.array(history_data['y_rsna_score'].values, dtype=np.float32)
    plt.plot(scores, label=backbone)

plt.title('Training Score vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend(loc='upper left')
plt.savefig('score_epoch.png')
plt.show()
