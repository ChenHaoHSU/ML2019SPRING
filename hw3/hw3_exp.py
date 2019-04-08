##########################
# Q1 & Q2
##########################
'''
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
'''

##########################
# Q3
##########################


##########################
# Q4
##########################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

# # import some data to play with
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# class_names = iris.target_names
# print(type(class_namess))

# # Split the data into a training set and a test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# # Run classifier, using a model that is too regularized (C too low) to see
# # the impact on the results
# classifier = svm.SVC(kernel='linear', C=0.01)
# y_pred = classifier.fit(X_train, y_train).predict(X_test)

# load true label
train_fpath = '../data/hw3/train.csv'
pred_fpath = 'output_train.csv'

train_data = pd.read_csv(train_fpath)
y_true = np.array(train_data['label'].values, dtype=int)

pred_data = pd.read_csv(pred_fpath)
y_pred = np.array(pred_data['label'].values, dtype=int)

val_size = 0.1
val_len = int(round(len(y_true)*(1-val_size)))
y_true = y_true[val_len:None]
y_pred = y_pred[val_len:None]

print(y_true)
print(y_pred)
print(y_true.shape)
print(y_pred.shape)

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix. [0.24 0.56 0.06 0.03 0.09 0.03 0.  ]
 [0.12 0.   0.55 0.02 0.17 0.07 0.08]

    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)
class_names = np.array(['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'], dtype=str)

# # Plot non-normalized confusion matrix
# plot_confusion_matrix(y_true, y_pred, classes=class_names,
#                       title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_true, y_pred, classes=class_names, normalize=True,
                      title='Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()





