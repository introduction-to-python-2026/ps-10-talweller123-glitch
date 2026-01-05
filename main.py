import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
from image_utils import load_image, edge_detection

def main():
  image_path = "tal bezet2.jpg"
    
    try:
        original_img = load_image(image_path)
        print("Image loaded successfully.")
        clean_image = median(original_img, ball(3))
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

if __name__ == "__main__":
    main()
