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
