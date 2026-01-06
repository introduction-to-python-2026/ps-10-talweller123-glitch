import numpy as np
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball, disk
from scipy.signal import convolve2d

def load_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        return np.array(img)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def edge_detection(image_array):
    gray_image = np.mean(image_array.astype(float), axis=2)

    kernelY = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])

    kernelX = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])

    edgeX = convolve2d(gray_image, kernelX, mode='same', boundary='fill', fillvalue=0)
    edgeY = convolve2d(gray_image, kernelY, mode='same', boundary='fill', fillvalue=0)

    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    return edgeMAG
