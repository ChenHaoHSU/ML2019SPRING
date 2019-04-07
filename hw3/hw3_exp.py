

# Plot acc and acc_val of CNN and DNN
import matplotlib.pyplot as plt
import numpy as np

acc_fpath = 'acc_CNN.csv'
acc_val_fpath = 'acc_val_CNN.csv'
# acc_fpath = 'acc_DNN.csv'
# acc_val_fpath = 'acc_val_DNN.csv'
acc = np.genfromtxt(acc_fpath)
acc_val = np.genfromtxt(acc_val_fpath)

plt.plot(acc)
plt.plot(acc_val)
plt.title('Training Process_CNN')
# plt.title('Training Process_DNN')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['acc', 'val_acc'], loc='upper left')
plt.savefig('Training_process_CNN.png')
# plt.savefig('Training_process_DNN.png')
plt.show()