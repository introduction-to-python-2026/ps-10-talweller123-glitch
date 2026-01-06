import numpy as np
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
from image_utils import load_image, edge_detection

def main():
    image_path = 'tal bezet2.jpg' 
    original_img = load_image(image_path)
    clean_image = median(original_img, ball(3))
    edge_mag = edge_detection(clean_image)
    
    threshold = 50 
    edge_binary = (edge_mag > threshold).astype(np.uint8) * 255
    
    edge_image = Image.fromarray(edge_binary)
    edge_image.save('my_edges.png')

if __name__ == "__main__":
    main()
