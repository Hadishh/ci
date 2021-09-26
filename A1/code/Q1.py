# Q1_graded
# Do not change the above line.

# This cell is for your imports.

import numpy as np
import matplotlib.pyplot as plt

# Q1_graded
# Do not change the above line.

# This cell is for your codes.
learning_rate = 0.6
nor_input =[[1, 1, 1], [1, 0, 1], [0, 0, 1], [0, 1, 1]]
true_labels = [0, 0, 1, 0]


# Q1_graded
# Do not change the above line.

# This cell is for your codes.
iter = 0
exp_error = 0
weights = np.random.rand(3)
print(weights)
while iter * len(nor_input) < 20:
  iter += 1
  for i in range(len(nor_input)):
    error = true_labels[i] - np.sum(weights * nor_input[i])
    weights = weights + np.dot(np.dot(nor_input[i] , error), learning_rate)
    error = 0.5 * error**2
    exp_err = 0.25 * error + 0.75 * exp_error
  if(iter % 100 == 0):
    print(exp_err)
x1 = np.linspace(0,1,100)
x2 = -weights[0] / weights[1] * x1 - weights[2] / weights[1]     #w1x1 + w2x2 + b = 0
plt.plot(x1, x2)
plt.plot(1, 1, 'bo')
plt.plot(1, 0, 'bo')
plt.plot(0, 0, 'ro')
plt.plot(0, 1, 'bo')

