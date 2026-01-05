from skimage.filters import median
from skimage.morphology import disk # Changed from ball to disk
import numpy as np
from PIL import Image

clean_image = median(edgeMAG, disk(3)) # Changed ball(3) to disk(3)

# Normalize and convert to uint8 for saving as PNG
clean_image_scaled = 255 * (clean_image - clean_image.min()) / (clean_image.max() - clean_image.min())
clean_image_uint8 = clean_image_scaled.astype(np.uint8)

edge_image = Image.fromarray(clean_image_uint8)
edge_image.save('my_edges.png')
