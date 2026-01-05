!wget https://raw.githubusercontent.com/yotam-biu/ps12/main/image_utils.py -O /content/image_utils.py
!wget https://raw.githubusercontent.com/yotam-biu/python_utils/main/lab_setup_do_not_edit.py -O /content/lab_setup_do_not_edit.py
import lab_setup_do_not_edit

from image_utils import load_image, edge_detection
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball

import numpy as np
from PIL import Image

def load_image(file_path):
  """
  Loads a color image from the given file path and converts it to a NumPy array.

  Args:
    file_path (str): The path to the image file.

  Returns:
    np.array: The image as a NumPy array.
  """
  img = Image.open(file_path)
  return np.array(img)

# Test the function
image_path = 'tal bezet2.jpg' # Using an image found in the files list
tal_image_array = load_image(image_path)

print(f"Image loaded successfully from {image_path}")
print(f"Shape of the image array: {tal_image_array.shape}")
print(f"Data type of the image array: {tal_image_array.dtype}")


import numpy as np
from scipy.signal import convolve2d

def edge_detection(image_array):
  """
  Performs edge detection on a color image array.

  Args:
    image_array (np.array): The input color image as a NumPy array.

  Returns:
    np.array: The edge magnitude array (edgeMAG).
  """
  # Convert to grayscale
  grayscale_image = np.mean(image_array, axis=2)

  # Define kernels
  kernelY = np.array([
      [1, 2, 1],
      [0, 0, 0],
      [-1, -2, -1]
  ])
  kernelX = np.array([
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1]
  ])

  # Apply convolution
  edgeY = convolve2d(grayscale_image, kernelY, mode='same', boundary='symm')
  edgeX = convolve2d(grayscale_image, kernelX, mode='same', boundary='symm')

  # Compute edge magnitude
  edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

  return edgeMAG

# Test the function
# Assuming tal_image_array is already loaded from the previous step
if 'tal_image_array' in locals():
  edgeMAG = edge_detection(tal_image_array)
  print(f"Edge detection applied. Shape of edgeMAG: {edgeMAG.shape}")
  print(f"Data type of edgeMAG: {edgeMAG.dtype}")
else:
  print("Please load an image first using the load_image function.")
