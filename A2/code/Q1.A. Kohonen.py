#write your code here
#define data first
#Q1A_graded
import numpy as np
import random
def generate_RGB_random(instance_count):
  data = []
  for i in range(instance_count):
    R = random.randint(0, 255)
    G = random.randint(0, 255)
    B = random.randint(0, 255)
    data.append([R, G, B])
  return data

def generate_colors_uniform(instance_count, shuffle=True):
  step_size = 255 / np.cbrt(instance_count)
  R = 0
  data = []
  while R <= 255:
    G = 0
    while G <= 255:
      B = 0
      while B <= 255:
        data.append([R, G, B])
        B += step_size
      G += step_size
    R += step_size
  if shuffle:
    np.random.shuffle(data)
  return data[:instance_count]

#Q1A_graded
def winner_neuron(map, pattern):
  min_dist = np.Infinity
  min_idx = (-1, -1)
  for k in range(map.shape[0]):
    for j in range(map.shape[1]):
      distance = np.linalg.norm(pattern - map[k, j])
      if (distance < min_dist):
        min_dist = distance
        min_idx = (k , j)
  return min_idx 

#Q1A_graded
def width(r0, epoch, lambda_):
  return r0 * np.exp(-epoch / lambda_)
  # return 5
def gaussian_kernel_constant_r(m, n, k, j, r0):
  distance = np.linalg.norm([m - k, n - j])
  if(distance > r0):
    return 0
  power = -(distance) / (2 * r0**2)
  return np.exp(power)

def gaussian_kernel(m, n, k, j, r0, epoch, lambda_):
  distance = np.linalg.norm([m - k, n - j])
  if(distance > r0):
    return 0
  r = width(r0, epoch, lambda_)
  power = -(distance) / (2 * r**2)
  return np.exp(power)

#Q1A_graded
import matplotlib.pyplot as plt
def init_map(shape ,data):
  w = np.random.randn(shape[0], shape[1], 3)*255
  w = np.clip(w, 0, 255)
  for i in range(shape[0]):
    for j in range(shape[1]):
      r = random.randint(0, len(data) - 1)
      w[i, j] = data[r].copy()
  return w
def plot(map, epoch):
  img = map.astype(np.uint8)
  plt.imshow(img)
  plt.title(f"Trained SOM after {epoch} epochs.")
  plt.show()

#Q1A_graded
import itertools
def kohonen_model_constant(data, map_shape, neighborhood_function, epochs=1000, learning_rate=0.01):
  shape = list(map_shape) + [3]
  shape = tuple(shape)
  w = init_map(shape, data)
  # w = np.zeros(shape=shape)
  neighborhood_radius0 = 10
  for epoch in range(epochs):
    # if(epoch % 2 == 0):
    print(f"starting epoch {epoch + 1}")
    random.shuffle(data)
    for idx, pattern in enumerate(data):
      m, n = winner_neuron(w, pattern)
      dTimeConstant = epochs/np.log(neighborhood_radius0)
      neighborhood_radius = neighborhood_radius0
      lr = learning_rate * np.exp(- epoch / epochs)
      # print(m, n)
      for k, j in itertools.product(range(shape[0]), range(shape[1])):
        influence = neighborhood_function(m, n, k, j, neighborhood_radius0)
        w[j, k] = w[j, k] + learning_rate * influence * (pattern - w[j, k])
    plot(w, epoch + 1)
  return w
data = generate_RGB_random(1600)
map_constant = kohonen_model_constant(data, (40, 40), gaussian_kernel_constant_r, epochs=10, learning_rate=0.05)

