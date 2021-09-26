#write your code here
#Q2B_graded
#used hybrid learning process with k-mean clustering centering
def init_centers(x, centers_count):
  centers = np.zeros((centers_count, x.shape[1]))
  for j in range(centers_count):
    index = random.choice(np.arange(x.shape[0]))
    centers[j] = x[index]
  return centers

def k_mean(x, K, iters):
  centroids = init_centers(x, K)
  for iter in range(iters):
    cluster_points = [[] for i in range(K)]
    for x_ in x:
      min_dist = np.Infinity
      min_k = -1
      for k in range(K):
        distance = np.linalg.norm(x_ - centroids[k])
        if(distance < min_dist):
          min_dist = distance
          min_k = k
      cluster_points[min_k].append(x_)
    for k in range(K):
      if(len(cluster_points[k]) == 0):
        continue
      mean = np.mean(cluster_points[k], axis=0)
      centroids[k] = mean
    return centroids
def compute_max_distance(centroids):
  max_distance = - np.Infinity
  for c1 in centroids:
    for c2 in centroids:
      dist = np.linalg.norm(c1 - c2)
      if dist > max_distance:
        max_distance = dist
  return max_distance

  
mu = k_mean(x_train.reshape(1000, 1), 20, 1)

#Q2B_graded
def init(x, y, S=20):  
  mu = k_mean(x.reshape(1000, 1), S, 1)
  sigma = np.ones((S, 1)) * compute_max_distance(mu) / np.sqrt(S * 2)
  w = np.random.randn(S, 1) * 0.001
  return mu, sigma, w 

def phie(x, mu_i, sigma_i):
  norm = np.linalg.norm(x - mu_i)
  power = - norm * norm / (2 * sigma_i * sigma_i)
  return np.exp(power)

def compute_f(x, mu, w, sigma):
  sum = 0
  for i in range(mu.shape[0]):
    sum += w[i] * phie(x, mu[i], sigma[i])
  return sum

def compute_grads(x, y, f, mu, w, sigma):
  dw = np.zeros_like(w)
  dsigma = np.zeros_like(sigma)
  for i in range(dw.shape[0]):
    dw[i] = -1 * (y - f) * phie(x, mu[i], sigma[i])
    norm = np.linalg.norm(x - mu[i])
    coeff_dsigma = norm**2 / sigma[i]**3
    dsigma[i] = coeff_dsigma * phie(x, mu[i], sigma[i])
  return dw, dsigma
  
    

#Q2B_graded
def RBF_model(x, y, epochs, learning_rate1):
  errors = []
  mu, sigma, w = init(x, y, S=15)
  assert y.shape[0] == x.shape[0]
  for epoch in range(epochs):
    MSE = 0
    for n in range(x.shape[0]):
      f = compute_f(x[n], mu, w, sigma)
      error = 1 / 2 * (y[n] - f)**2
      dw, dsigma = compute_grads(x[n], y[n], f, mu, w, sigma)
      w -= dw * learning_rate1
      MSE += 0.001 * error
    errors.append(MSE)
  return mu, sigma, w, errors

mu, sigma, w, loss = RBF_model(x_train, y_train, 15, 0.001)
plt.plot(loss)
plt.title("RBF Loss per epoch")
plt.show()

