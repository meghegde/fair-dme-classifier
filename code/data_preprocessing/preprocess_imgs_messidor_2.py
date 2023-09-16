import numpy as np
import pandas as pd
import os, shutil, pathlib
import glob
import cv2

def circle_crop_and_resize(img):
  # Create mask and draw circle onto mask
  image = cv2.imread(img)
  mask = np.zeros(image.shape, dtype=np.uint8)

  height, width, depth = image.shape
  x = int(width / 2)
  y = int(height / 2)
  r = int(np.amin((x, y))*0.9)

  cv2.circle(mask, (x,y), r, (255,255,255), -1)

  # Bitwise-and for ROI
  ROI = cv2.bitwise_and(image, mask)

  # Crop mask and turn background black
  mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
  x,y,w,h = cv2.boundingRect(mask)
  result = ROI[y:y+h,x:x+w]
  mask = mask[y:y+h,x:x+w]
  result[mask==0] = (0,0,0)

  # Make image smaller
  result = cv2.resize(result, (800, 800))

  return result

#src_dir = 'MESSIDOR-2/original'
src_dir = 'MESSIDOR-2/original_caucasian'

#dest_dir = 'MESSIDOR-2/processed'
dest_dir = 'MESSIDOR-2/processed_caucasian'

for img in os.listdir(src_dir):
    new_img = circle_crop_and_resize(os.path.join(src_dir, img))
    cv2.imwrite(os.path.join(dest_dir, img), new_img)
