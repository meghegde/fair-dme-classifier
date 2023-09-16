import os
import cv2
import numpy as np

src_dir = 'data/dme_clf_indian_aug_small/train/dme'
img_name = 'IDRiD_55.jpg_7.jpg'
dst_dir = 'IDRiD'

def circle_crop(img):
    """
    Create circular crop around image centre
    """
    img = cv2.imread(img)

    #height, width, depth = img.shape
    #largest_side = np.max((height, width))
    #img = cv2.resize(img, (largest_side, largest_side))

    height, width, depth = img.shape

    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    mask = cv2.circle(circle_img, (x, y), int(r), (255,255,255), thickness=-1)
    img = cv2.bitwise_and(img, img, mask=mask)

    # # No need to resize - we will do this when reading in the dataset using image_dataset_from_directory
    #new_size = int(largest_side/4)
    #img = cv2.resize(img, (new_size, new_size)) # Make image smaller

    return img

circle_img = circle_crop(os.path.join(src_dir, img_name))
new_img = circle_img[50:-50, :]
cv2.imwrite(os.path.join(dst_dir, img_name), new_img)
