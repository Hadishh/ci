# Q4_graded
# Do not change the above line.

# This cell is for your imports.

from keras.layers import *
from keras.optimizers import *
from keras.models import Sequential
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Q4_graded
# Do not change the above line.

# This cell is for your codes.

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], 784))
y_train_onehot = np.zeros(shape=(y_train.shape[0], 10))
y_train_onehot[np.arange(y_train.size), y_train] = 1
y_train = y_train_onehot
x_test = x_test.reshape((x_test.shape[0], 784))
y_test_onehot = np.zeros(shape=(y_test.shape[0], 10))
y_test_onehot[np.arange(y_test.size), y_test] = 1
y_test = y_test_onehot


# Q4_graded
# Do not change the above line.
model = Sequential()
model.add(normalization.BatchNormalization(input_dim=784))
model.add(core.Dense(512, activation=None))
model.add(core.Dense(256, activation=None))
model.add(core.Dense(128, activation=None))
model.add(core.Dense(64, activation=None))
model.add(core.Dense(32, activation=None))
model.add(core.Dense(10, activation=None))
model.add(Softmax())
model.compile(loss="binary_crossentropy", optimizer="sgd",metrics=["accuracy"])
history = model.fit(x=x_train, y=y_train, batch_size=32, epochs=30)
# This cell is for your codes.

# Q4_graded
# Do not change the above line.
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
print(f"Accuracy on test set: {model.evaluate(x_test, y_test)}")

