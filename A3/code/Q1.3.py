# Q1.3_graded
# Do not change the above line.
import numpy as np
import random

# This cell is for your imports.
def zero_pad(patterns, shape):
  padded_patterns = []
  for p in patterns:
    new_p = np.zeros(shape)
    diff = shape[0] - p.shape[0]
    start_idx = diff // 2
    end_idx = start_idx + p.shape[0]
    new_p[start_idx:end_idx] = p
    padded_patterns.append(new_p)
  return padded_patterns
def normalize(patterns, max_value= 255):
  for i in range(len(patterns)):
    patterns[i] = np.sign(patterns[i] - 0.00001)
    # patterns[i] /= 255
  return patterns
def add_noise(pattern, precision=0.1):
  count = int(pattern.shape[0] * precision)
  indecies = np.random.choice(pattern.shape[0], count)
  pattern[indecies] *= -1
  return pattern
def calculate_accuracy(y_pred, y_true):
  true_count = np.sum(y_pred == y_true)
  return true_count / y_true.shape[0]

# Q1.3_graded
# Do not change the above line.

# This cell is for your codes.

from PIL import Image, ImageFont
import numpy as np
def generate_patterns(font_size):
  font = ImageFont.truetype("/content/arial.ttf", font_size)
  patterns = []
  max_shape = (-float('inf'), )
  for char in "ABCDEFGHIJ":
    im = Image.Image()._new(font.getmask(char))
    # im.save(f"{char}_{font_size}.bmp")
    pattern = np.array(im.getdata())
    if (pattern.shape[0] > max_shape[0]):
      max_shape = pattern.shape
    patterns.append(pattern) 
  X = zero_pad(patterns, max_shape)
  X = normalize(X)
  return X, max_shape

# Q1.3_graded
def calculate_weights(X, shape):
  W = np.zeros((shape[0], shape[0]))
  for x in X:
    W += find_weight_matrix(x)
  return W

def evaluate(X, W, noise_precision=0.1):
  noisy_data = [add_noise(x, noise_precision) for x in X]
  accuracy = []
  for i, x in enumerate(noisy_data):
    current_pattern = get_next_pattern(x, W)
    while (current_pattern != x).any():
      x = current_pattern
      current_pattern= get_next_pattern(x, W)
      # print(calculate_error(current_pattern, W))
    accuracy.append(calculate_accuracy(current_pattern, X[i]))
  return np.average(accuracy)

# Q1.3_graded
font_sizes = [16, 32, 64]
noise_precisions = [0.1, 0.3, 0.6]
for font_size in font_sizes:
  for noise in noise_precisions:
    X, shape = generate_patterns(font_size)
    W = calculate_weights(X, shape)
    # if (font_size == 16 and noise == 0.1):
    #   plt.imshow(W)
    #   plt.colorbar()
    #   plt.show()
    accuracy = evaluate(X, W, noise)
    print(f"Average accuracy on font size {font_size} with {noise *100}% noise: {accuracy * 100}")

