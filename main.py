import numpy as np
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
# ייבוא הפונקציות מהקובץ השני שלך
from image_utils import load_image, edge_detection

def main():
    # 1. טעינת התמונה המקורית (וודאי שהשם והסיומת תואמים לקובץ שהעלית ל-GitHub)
    image_path = 'tal bezet2.jpg' 
    original_img = load_image(image_path)
    
    # 2. סינון רעשים בעזרת Median Filter ו-ball(3) כפי שהתבקש
    # הערה: median מצפה לתמונה, ball(3) הוא המבנה שמגדיר את רדיוס הסינון
    clean_image = median(original_img, ball(3))
    
    # 3. הרצת התמונה הנקייה דרך פונקציית גילוי הקצוות שכתבת
    edge_mag = edge_detection(clean_image)
    
    # 4. המרה לתמונה בינארית (שחור-לבן) בעזרת סף (Threshold)
    # ערך של 100 הוא בדרך כלל נקודת התחלה טובה
    threshold = 100
    edge_binary = (edge_mag > threshold).astype(np.uint8) * 255
    
    # 5. שמירת התוצאה כקובץ PNG
    edge_image = Image.fromarray(edge_binary)
    edge_image.save('my_edges1.png')
    print("The edge-detected image has been saved as 'my_edges.png'")

if __name__ == "__main__":
    main()
