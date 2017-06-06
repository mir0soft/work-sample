#### Kaggle Challenge: Identify correctly images of handwritten digits (MNIST Dataset)
#### Link: https://www.kaggle.com/c/digit-recognizer
#### Author: Mihran Hakobyan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.constraints import maxnorm
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing
from sklearn import model_selection


data = pd.read_csv('/Users/mihran1/Documents/python/digit/train.csv')[:10000]

# separate feautres and labels
X = np.array(data.drop('label', axis=1), dtype='float32')
y = np.array(data.label)

# scale features to zero mean and unit variance
X = preprocessing.scale(X)

# split the data in train and validation sets
X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

# reshape the features to exploit the spatial information of the image
X_train = X_train.reshape(-1,28,28,1)
X_val = X_val.reshape(-1,28,28,1)

# convert labels to "one hot" labels
y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)

# lets have a look at the first 9 images
for i in range(9):
    plt.subplot(331+i)
    plt.imshow(X_train.reshape(-1,1,28,28)[i][0], cmap='gray')
plt.show()

# compare images with the labels
print(y_train[:9])

#  I will use a Convolutional Neural Net to solve the problem
# Architecture: CONV -> CONV -> CONV -> MAXPOOL -> FLATTEN -> DENSE -> DROPOUT -> DENSE
model = Sequential()

# first convolutional layer with size 28 x 28 x 16
model.add(Conv2D(filters = 16, kernel_size = (3, 3), input_shape = (28, 28, 1), activation='relu', padding='same'))

# second convolutional layer with size 28 x 28 x 32
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu', padding='same'))

# third convolutional layer with size 28 x 28 x 64
model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation='relu', padding='same'))

# max pooling layer with size 14 x 14 x 64
model.add(MaxPooling2D(pool_size=(2,2)))

# after flattening we obtain a fully connected layer with 12.544 (=14*14*64) neurons
model.add(Flatten())

# fully connected layer with 128 neurons
model.add(Dense(128, activation='relu'))

# dropout to prevent overfitting
model.add(Dropout(0.5))

# last fully connected layer with 10 outputs (number of classes)
model.add(Dense(10, activation='softmax'))

# compile the Net with the best optimizer Adadelta
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

# fit the model, 15 epochs needed for 99% accuracy
hist = model.fit(X_train, y_train, batch_size=128, epochs=15, validation_data=(X_val, y_val))

# visualize history of losses and accuracies
plt.figure()
plt.plot(hist.history['loss'], color='k')
plt.plot(hist.history['val_loss'], color='r')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss History')

plt.figure()
plt.plot(hist.history['acc'], color='k')
plt.plot(hist.history['val_acc'], color='r')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy History')
plt.show()
