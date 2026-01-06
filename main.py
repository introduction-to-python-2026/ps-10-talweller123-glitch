from skimage.filters import median
from skimage.morphology import ball
import numpy as np
from PIL import Image

clean_image = median(edgeMAG, ball(3)) 

# Normalize and convert to uint8 for saving as PNG
clean_image_scaled = 255 * (clean_image - clean_image.min()) / (clean_image.max() - clean_image.min())
clean_image_uint8 = clean_image_scaled.astype(np.uint8)

edge_image = Image.fromarray(clean_image_uint8)
edge_image.save('my_edges1.png')

