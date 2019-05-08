##########################
# Q1 & Q2
##########################
# Plot acc and val_acc of RNN and BOW
import matplotlib.pyplot as plt
import numpy as np

TYPE = 'BOW'

acc_fpath = '{}_acc.csv'.format(TYPE)
val_acc_fpath = '{}_val_acc.csv'.format(TYPE)
loss_fpath = '{}_loss.csv'.format(TYPE)
val_loss_fpath = '{}_val_loss.csv'.format(TYPE)

acc = np.genfromtxt(acc_fpath)
val_acc = np.genfromtxt(val_acc_fpath)
loss = np.genfromtxt(loss_fpath)
val_loss = np.genfromtxt(val_loss_fpath)

plt.plot(acc)
plt.plot(val_acc)
plt.title('Accuracy ({})'.format(TYPE))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['acc', 'val_acc'], loc='upper left')
plt.savefig('{}_acc.png'.format(TYPE))
plt.show()

plt.plot(loss)
plt.plot(val_loss)
plt.title('Loss ({})'.format(TYPE))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['loss', 'val_loss'], loc='upper left')
plt.savefig('{}_loss.png'.format(TYPE))
plt.show()

