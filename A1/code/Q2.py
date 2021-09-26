# Q2_graded
# Do not change the above line.

# This cell is for your imports.

import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')

# Q2_graded
# Do not change the above line.

# This cell is for your codes.
input = []
true_labels = []
with open("/content/drive/MyDrive/Colab Notebooks/Computational Intelligence /Assignment 1/data.txt") as file:
  for line in file.readlines():
    splits = line.split(',')
    vector = [float(splits[0]), float(splits[1]), 1]
    label = float(splits[2])
    input.append(vector)
    true_labels.append(label)
input = np.array(input)

print(f"Loaded {len(input)} entries.")

# Q2_graded
def overall_Q2_accuracy(weights):
  count = len(input)
  true_count = 0
  for i in range(count):
    y = np.sum((weights * input[i]))
    if ((y >= 0 and true_labels[i] == 0) or (y < 0 and true_labels[i] == 1)):
      true_count += 1
  return true_count / count

# Q2_graded
# Do not change the above line.

# This cell is for your codes.
weights = np.random.rand(3)
iteration = 0
error_plot_data = []
acc_plot_data = []
learning_rate = 0.001
total_iters = 70000
err_mean = None
while iteration < total_iters:
    i = iteration % len(input)
    iteration += 1
    if(iteration % 5000 == 0):
      print(f"Iteration {iteration} with accuracy {overall_Q2_accuracy(weights)}")
    en = true_labels[i] - np.sum(weights * input[i])
    error = 0.5 * en**2
    if(err_mean is None):
      err_mean = error
    err_mean = 0.01 * error + 0.99 * err_mean
    error_plot_data.append((iteration, np.log(err_mean)))
    acc_plot_data.append((iteration, overall_Q2_accuracy(weights)))
    delta = np.dot(np.dot(input[i], learning_rate / iteration), en)
    # prevent gradiants from exploading
    delta = np.array([x / abs(x) * min(100000, abs(x)) for x in delta])
    weights = weights + delta



# Q2_graded
x,y = zip(*error_plot_data)
plt.plot(list(x), list(y))
plt.title("Exponential Mean Loss")
plt.ylabel("loss")
plt.xlabel("iteration")
plt.show()
x,y = zip(*acc_plot_data)
plt.plot(list(x), list(y))
plt.title("Model Accuracy")
plt.ylabel("accuracy")
plt.xlabel("iteration")
plt.show()
x1 = np.linspace(-200,-60,1000)
x2 = -weights[0] / weights[1] * x1 - weights[2] / weights[1]     # w1x1 + w2x2 + b = 0
plt.plot(x1, x2)
plt.title("Model Prediction")
plt.ylabel("x2")
plt.xlabel("x1")
for i in range(len(input)):
  dot = 'ro'
  if true_labels[i] == 0:
    dot = 'bo'
  plt.plot(input[i][0], input[i][1], dot, markersize=2)
plt.show()
print(overall_Q2_accuracy(weights))

