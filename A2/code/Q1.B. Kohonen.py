#write your code here
#Q1B_graded

def kohonen_model_constant_radius(data, map_shape, neighborhood_function, epochs=1000, lr0=0.01):
  shape = list(map_shape) + [3]
  shape = tuple(shape)
  w = init_map(shape, data)
  neighborhood_radius0 = 10
  dTimeConstant = epochs/np.log(neighborhood_radius0)
  for epoch in range(epochs):
    print(f"starting epoch {epoch + 1}")
    random.shuffle(data)
    for idx, pattern in enumerate(data):
      m, n = winner_neuron(w, pattern)
      lr = lr0 * np.exp(- epoch / epochs)
      for k, j in itertools.product(range(shape[0]), range(shape[1])):
        influence = neighborhood_function(m, n, k, j, neighborhood_radius0)
        w[j, k] = w[j, k] + lr * influence * (pattern - w[j, k])
    plot(w, epoch + 1)
  return w

map_constant_r = kohonen_model_constant_radius(data, (40, 40), gaussian_kernel_constant_r, 10, lr0=0.05)

#write your code here
#Q1C_graded

def kohonen_model(data, map_shape, neighborhood_function, epochs=1000, lr0=0.01):
  shape = list(map_shape) + [3]
  shape = tuple(shape)
  w = init_map(shape, data)
  neighborhood_radius0 = 10
  lambda_ = epochs/np.log(neighborhood_radius0) 
  for epoch in range(epochs):
    print(f"starting epoch {epoch + 1}")
    random.shuffle(data)
    for idx, pattern in enumerate(data):
      m, n = winner_neuron(w, pattern)
      lr = lr0 * np.exp(- epoch / epochs)
      # print(m, n)
      for k, j in itertools.product(range(shape[0]), range(shape[1])):
        influence = gaussian_kernel(m, n, k, j, neighborhood_radius0, epochs, lambda_)
        w[j, k] = w[j, k] + lr * influence * (pattern - w[j, k])
    plot(w, epoch + 1)
  return w

map = kohonen_model(data, (40, 40), gaussian_kernel_constant_r, 20, lr0=0.05)

#Q1C_graded
img = np.asarray(data).reshape(40, 40, 3)
img = img.astype(np.uint8)
plt.imshow(img)
plt.title("Original Data")
plt.show()

img = map_constant
img = img.astype(np.uint8)
plt.imshow(img)
plt.title("Trained SOM After 10 epochs")
plt.show()

img = map_constant_r
img = img.astype(np.uint8)
plt.imshow(img)
plt.title("Trained SOM After 10 epochs with decaying lr")
plt.show()

img = map
img = img.astype(np.uint8)
plt.imshow(img)
plt.title("Trained SOM After 20 epochs with decaying lr and radius")
plt.show()


