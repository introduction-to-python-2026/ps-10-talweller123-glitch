import numpy as np
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball, disk
from scipy.signal import convolve2d
def main():
    image_path = 'tal bezet2.jpg'
    original_img = load_image(image_path)
    if original_img is None:
        print("Image loading failed. Cannot proceed with edge detection.")
        return
    clean_image = np.zeros_like(original_img) 
    for c in range(original_img.shape[2]): 
        clean_image[:, :, c] = median(original_img[:, :, c], disk(3)) 
    edge_mag = edge_detection(clean_image)
    threshold = 50
    edge_binary = (edge_mag > threshold).astype(np.uint8) * 255
    edge_image = Image.fromarray(edge_binary)
    edge_image.save('my_edges.png')
    print("The edge-detected image has been saved as 'my_edges.png'")
if __name__ == "__main__":
    main()
