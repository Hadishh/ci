#Q2A_graded
# import numpy as np

def generate_data(instances_count, range_=(-3, 3)):
  x_train = np.arange(instances_count) / instances_count * (range_[1] - range_[0]) + range_[0]
  y_train = np.sin(x_train)
  return x_train, y_train

x_train, y_train = generate_data(1000)

#write your code here
#Q2A_graded
from keras.optimizers import *
from keras.layers import *
from keras.models import Sequential
from keras.datasets import mnist
import matplotlib.pyplot as plt
model_MLP = Sequential()
model_MLP.add(core.Dense(40, input_dim=1, activation='relu'))
model_MLP.add(core.Dense(10, activation='relu'))
model_MLP.add(core.Dense(1, activation=None))
model_MLP.compile(loss="mean_squared_error", optimizer="sgd",metrics=["mean_squared_error"])
history = model_MLP.fit(x=x_train, y=y_train, batch_size=1, epochs=10)


