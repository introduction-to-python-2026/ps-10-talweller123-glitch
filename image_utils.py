import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def load_image(image_path):
    """טוען תמונה ומחזיר אותה כמערך NumPy"""
    img = Image.open(image_path)
    # המרה ל-RGB כדי להבטיח שיש 3 ערוצים
    img = img.convert('RGB')
    return np.array(img)

def edge_detection(image_array):
    """מבצע גילוי קצוות בשיטת Sobel"""
    # 1. המרה לגווני אפור על ידי ממוצע הערוצים
    gray_image = np.mean(image_array, axis=2)

    # 2. הגדרת הפילטרים (Kernels) בדיוק לפי ההוראות
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

    # 3. קונבולוציה (Convolution)
    # שימוש ב-mode='same' כדי לשמור על גודל התמונה
    # ו-boundary='fill' עם fillvalue=0 עבור ריפוד באפסים (Zero Padding)
    edgeX = convolve2d(gray_image, kernelX, mode='same', boundary='fill', fillvalue=0)
    edgeY = convolve2d(gray_image, kernelY, mode='same', boundary='fill', fillvalue=0)

    # 4. חישוב עוצמת הקצה (Magnitude)
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    return edgeMAG
