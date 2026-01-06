import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def load_image(image_path):
    img = Image.open(image_path)
    return np.array(img)

def edge_detection(image_array):
    gray_image = np.mean(image_array.astype(float), axis=2)
    
    kernelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    
    edgeX = convolve2d(gray_image, kernelX, mode='same', boundary='fill', fillvalue=0)
    edgeY = convolve2d(gray_image, kernelY, mode='same', boundary='fill', fillvalue=0)
    
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    return edgeMAG
