##########################
# Q1 & Q2
##########################
# Plot acc and val_acc of CNN and DNN
import matplotlib.pyplot as plt
import numpy as np

# acc_fpath = 'acc_CNN.csv'
# val_acc_fpath = 'val_acc_CNN.csv'
# loss_fpath = 'loss_CNN.csv'
# val_loss_fpath = 'val_loss_CNN.csv'
acc_fpath = 'acc_DNN.csv'
val_acc_fpath = 'val_acc_DNN.csv'
loss_fpath = 'loss_DNN.csv'
val_loss_fpath = 'val_loss_DNN.csv'
acc = np.genfromtxt(acc_fpath)
val_acc = np.genfromtxt(val_acc_fpath)
loss = np.genfromtxt(loss_fpath)
val_loss = np.genfromtxt(val_loss_fpath)

plt.plot(acc)
plt.plot(val_acc)
# plt.title('Training Process_CNN')
plt.title('Training Process_DNN')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['acc', 'val_acc'], loc='upper left')
# plt.savefig('Training_process_acc_CNN.png')
plt.savefig('Training_process_acc_DNN.png')
plt.show()

plt.plot(loss)
plt.plot(val_loss)
# plt.title('Training Process_CNN')
plt.title('Training Process_DNN')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['acc', 'val_acc'], loc='upper left')
# plt.savefig('Training_process_loss_CNN.png')
plt.savefig('Training_process_loss_DNN.png')
plt.show()

##########################
# Q3
##########################


##########################
# Q4
##########################






