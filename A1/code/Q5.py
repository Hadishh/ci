# Q5_graded
# Do not change the above line.

# This cell is for your imports

import numpy as np
import matplotlib.pyplot as plt

def Initialize_parameters(layers_size, input_dim):
  parameters = {}
  parameters['W1'] = np.random.randn(input_dim, layers_size[0]) * 0.01
  parameters['b1'] = np.random.randn(layers_size[0], 1)
  for i in range(1, len(layers_size)):
    parameters[f'W{i + 1}'] = np.random.randn(layers_size[i - 1], layers_size[i]) * 0.01
    parameters[f'b{i + 1}'] = np.random.randn(layers_size[i], 1)
  return parameters

def normalize(x):
  return x / 255

def get_batch(x, batch_number, batch_size):
  start_idx = batch_number * batch_size
  if(start_idx >= x.shape[1]):
    print(start_idx, x.shape)
    raise IndexError
  if(start_idx + batch_size >= x.shape[1]):
    return x[:, start_idx:].copy()
  return x[:, start_idx : start_idx + batch_size].copy()

def batch_normalization(batch, beta, gamma):
  epsilon = 0.001
  batch_mean = np.mean(batch, axis=0)
  batch_var = np.var(batch, axis=0)
  batch = (batch - batch_mean) / np.sqrt(batch_var + epsilon)
  return gamma * batch + beta

# Q5_graded
# Do not change the above line.

# Forward Propagation

def forward_propagation(x, parameters):
  x = batch_normalization(x, 0, 1)
  A0 = x
  cache = {'A0':A0}
  for i in range(0, 6):
    Z = np.dot(np.transpose(parameters[f'W{i + 1}']), cache[f'A{i}']) + parameters[f'b{i + 1}']
    # A = np.tanh(Z)
    A = Z
    cache[f'A{i + 1}'] = A
  #softmax
  exp = np.exp(cache['A6'])
  sum = np.sum(exp, axis=0)
  o = exp / sum
  return o, cache

def calculate_loss(y, o):
  return np.mean(np.sum(-y * np.log(o), axis=0))

# Backward Propagation
def backward_propagation(o, y, parameters, cache, batch_size):
  temp_cache = {}
  dA6 = o - y
  temp_cache['dA6'] = dA6
  grads = {}
  for i in reversed(range(1, 7)):
    g_prime = 1
    dZ = g_prime * temp_cache[f'dA{i}']
    dW = 1 / batch_size * np.dot(cache[f'A{i - 1}'], np.transpose(dZ))
    # dW = np.clip(dW, -1, 1)
    db = np.mean(dZ, axis=1, keepdims=True)
    # db = np.clip(db, -1, 1)
    grads[f'db{i}'] = db
    grads[f'dW{i}'] = dW
    dA = np.dot(parameters[f'W{i}'], dZ)
    temp_cache[f'dA{i - 1}'] = dA
  return grads

def update_parameters(parameters, grads, learning_rate):
  for i in range(1, 7):
    parameters[f'W{i}'] = parameters[f'W{i}'] - learning_rate * grads[f'dW{i}']
    parameters[f'b{i}'] = parameters[f'b{i}'] - learning_rate * grads[f'db{i}']
  return parameters


# Q5_graded
# Do not change the above line.

# This cell is for your codes.
def calculate_accuracy(o, y, batch_size=None):
  if(batch_size == None):
    batch_size = y.shape[1]
  pred = np.argmax(o, axis=0)
  true = np.argmax(y, axis=0)
  true_count = np.where(pred == true)[0].size
  batch_acc = true_count / batch_size
  return batch_acc
def predict(x, parameters):
  pred = forward_propagation(x, parameters)
  return pred[0]
def train(x, y, parameters, learning_rate, batch_size, epochs):
  history =  {'acc' : [], 'loss': []}
  total_batch_count = x.shape[1] // batch_size
  if(x.shape[1] % batch_size != 0):
    total_batch_count += 1
  for epoch in range(epochs):
    accuracy = 0
    loss = 0
    for batch in range(total_batch_count):
      x_batch = get_batch(x, batch, batch_size)
      y_batch = get_batch(y, batch, batch_size)
      o_batch, cache = forward_propagation(x_batch, parameters)
      batch_loss = calculate_loss(y_batch, o_batch)
      grads = backward_propagation(o_batch, y_batch, parameters, cache, batch_size)
      parameters = update_parameters(parameters, grads, learning_rate )
      loss = (loss * batch +  batch_loss) / (batch + 1)
      batch_acc = calculate_accuracy(o_batch, y_batch, batch_size)
      accuracy = (accuracy * batch +  batch_acc) / (batch + 1)
      # return
    history['acc'].append(accuracy)
    history['loss'].append(loss)
    print(f"End of epoch {epoch} : acc = {accuracy}, loss = {loss}")
  return history


# Q5_graded
x_train_t = np.transpose(x_train)
y_train_t = np.transpose(y_train)
x_test_t = np.transpose(x_test)
y_test_t = np.transpose(y_test)
parameters = Initialize_parameters([512, 256, 128, 64, 32, 10], 784)
his = train(x_train_t, y_train_t, parameters, 0.1, 32, 30)

# Q5_graded

y_test = np.transpose(y_test)
pred = predict(x_test_t, parameters)
print(f'Accuracy on test set: {calculate_accuracy(pred, y_test_t)}')

# Q5_graded
plt.plot(his['acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
plt.plot(his['loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

