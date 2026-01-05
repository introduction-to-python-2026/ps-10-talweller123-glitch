!wget https://raw.githubusercontent.com/yotam-biu/ps12/main/image_utils.py -O /content/image_utils.py
!wget https://raw.githubusercontent.com/yotam-biu/python_utils/main/lab_setup_do_not_edit.py -O /content/lab_setup_do_not_edit.py
import lab_setup_do_not_edit

from image_utils import load_image, edge_detection
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball

import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def load_image(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    return np.array(img)

def edge_detection(image_array):
    gray_image = np.mean(image_array, axis=2)
    kernelY = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
    kernelX = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    edgeX = convolve2d(gray_image, kernelX, mode='same', boundary='fill', fillvalue=0)
    edgeY = convolve2d(gray_image, kernelY, mode='same', boundary='fill', fillvalue=0)

    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    return edgeMAG

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.filters import median
from skimage.morphology import disk # Changed from ball to disk
from image_utils import load_image, edge_detection

def main():
  image_path = "tal bezet2.jpg"

  try:
      original_img = load_image(image_path)

      # Added check for NoneType after loading image
      if original_img is None:
          print(f"Error: Could not load image from '{image_path}'. Please ensure the file exists and is a valid image.")
          return

      print("Image loaded successfully.")

      # Fix: Apply 2D median filter to each channel independently
      clean_image = np.zeros_like(original_img, dtype=float) # Use float for filtering
      for c in range(original_img.shape[2]):
          clean_image[:, :, c] = median(original_img[:, :, c], disk(3)) # Use disk for 2D filter

      # Convert back to uint8 for consistency if original_img was uint8
      clean_image = clean_image.astype(np.uint8)

      print("Noise suppression completed.")
      edge_mag = edge_detection(clean_image)
      print("Edge detection completed.")
      threshold = 100
      edge_binary = (edge_mag > threshold).astype(np.uint8) * 255

      final_edge_image = Image.fromarray(edge_binary)
      final_edge_image.save('my_edges.png')
      print("Edge-detected image saved as 'my_edges.png'.")

      plt.figure(figsize=(10, 5))
      plt.subplot(1, 2, 1)
      plt.title("Original Image")
      plt.imshow(original_img)

      plt.subplot(1, 2, 2)
      plt.title("Edge Detected (Binary)")
      plt.imshow(edge_binary, cmap='gray')
      plt.show()

  except FileNotFoundError:
      print(f"Error: The file '{image_path}' was not found. Please place an image in the folder.")
  except Exception as e:
      print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
