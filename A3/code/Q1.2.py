# Q1.2_graded
# Do not change the above line.

# This cell is for your imports.

import numpy as np
import matplotlib.pyplot as plt
def find_weight_matrix(pattern):
  matrix = np.zeros((pattern.shape[0], pattern.shape[0]))
  p = pattern.reshape((-1, 1))
  result =  np.dot(p , np.transpose(p))
  np.fill_diagonal(result, 0)
  return result
def calculate_error(pattern, weight_matrix):
  p = pattern.reshape((-1, 1))
  mul = np.dot(p, np.transpose(p))
  E = - np.sum(mul * weight_matrix)
  return E
def get_next_pattern(pattern, weight_matrix):
  return np.sign(np.dot(pattern, weight_matrix))
W = np.zeros((6, 6))
W += find_weight_matrix(np.array([1, 1, 1, -1, -1, -1]))
W += find_weight_matrix(np.array([1, -1, 1, -1, 1, -1]))
print(W)

# Q1.2_graded
# Do not change the above line.

# This cell is for your codes.
print(get_next_pattern(np.array([1, 1, 1, -1, -1, -1]), W))
print(calculate_error(np.array([1, 1, 1, -1, -1, -1]), W))

print(get_next_pattern(np.array([-1, 1, 1, -1, -1, -1]), W))
print(calculate_error(np.array([-1, 1, 1, -1, -1, -1]), W))

