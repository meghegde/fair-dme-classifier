import numpy as np
import pandas as pd
import os, shutil, pathlib
import glob
import cv2

# Crop a rectangle around the centre of the image
# We want to match the aspect ratio of the IDRiD images
# So, width is approx. 1.5 x height
def rectangle_crop(img):
    """
    Create rectangular crop around image centre
    """
    img = cv2.imread(img)

    height, width, depth = img.shape

    x = int(width / 2)
    y = int(height / 2)

    desired_width = 800
    desired_height = 533

    x1 = int(x - desired_width/2)
    x2 = int(x + desired_width/2)
    y1 = int(y - desired_height/2)
    y2 = int(y + desired_height/2)

    cropped_img = img[y1:y2, x1:x2]

    return cropped_img

#src_dir = 'MESSIDOR-2/processed'
src_dir = 'MESSIDOR-2/processed_caucasian'

#dest_dir = 'MESSIDOR-2/processed_rectangle'
dest_dir = 'MESSIDOR-2/processed_rectangle_caucasian'

for img in os.listdir(src_dir):
    new_img = rectangle_crop(os.path.join(src_dir, img))
    cv2.imwrite(os.path.join(dest_dir, img), new_img)
