##########################
# Q1 & Q2
##########################
# Plot acc and val_acc of RNN and BOW
import matplotlib.pyplot as plt
import numpy as np

# acc_fpath = 'RNN_cc.csv'
# val_acc_fpath = 'RNN_val_acc.csv'
# loss_fpath = 'RNN_loss.csv'
# val_loss_fpath = 'RNN_val_loss.csv'
acc_fpath = 'BOW_acc.csv'
val_acc_fpath = 'BOW_val_acc.csv'
loss_fpath = 'BOW_loss.csv'
val_loss_fpath = 'BOW_val_loss.csv'

acc = np.genfromtxt(acc_fpath)
val_acc = np.genfromtxt(val_acc_fpath)
loss = np.genfromtxt(loss_fpath)
val_loss = np.genfromtxt(val_loss_fpath)

plt.plot(acc)
plt.plot(val_acc)
# plt.title('Training Process_RNN')
plt.title('Training Process (BOW)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['acc', 'val_acc'], loc='upper left')
# plt.savefig('Training_process_acc_RNN.png')
plt.savefig('Training_process_acc_BOW.png')
plt.show()

plt.plot(loss)
plt.plot(val_loss)
# plt.title('Training Process_RNN')
plt.title('Training Process (BOW)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['acc', 'val_acc'], loc='upper left')
# plt.savefig('Training_process_loss_RNN.png')
plt.savefig('Training_process_loss_BOW.png')
plt.show()

