

# Plot acc and val_acc of CNN and DNN
import matplotlib.pyplot as plt
import numpy as np

acc_fpath = 'acc_CNN.csv'
val_acc_fpath = 'val_acc_CNN.csv'
# acc_fpath = 'acc_DNN.csv'
# val_acc_fpath = 'val_acc_DNN.csv'
acc = np.genfromtxt(acc_fpath)
val_acc = np.genfromtxt(val_acc_fpath)

plt.plot(acc)
plt.plot(val_acc)
plt.title('Training Process_CNN')
# plt.title('Training Process_DNN')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['acc', 'val_acc'], loc='upper left')
plt.savefig('Training_process_CNN.png')
# plt.savefig('Training_process_DNN.png')
plt.show()