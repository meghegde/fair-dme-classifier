import os
import shutil
import cv2
import numpy as np

os.mkdir('data/dme_clf_indian_aug/')
os.mkdir('data/dme_clf_indian_aug/train/')
os.mkdir('data/dme_clf_indian_aug/train/dme')
os.mkdir('data/dme_clf_indian_aug/val/')
os.mkdir('data/dme_clf_indian_aug/val/dme')

train_dir = 'data/dme_clf_indian_aug/train/dme'
val_dir = 'data/dme_clf_indian_aug/val/dme'

train_imgs = ['IDRiD_Randomize_IDRiD_62.jpg', 'IDRiD_Randomize_IDRiD_59.jpg',
'IDRiD_Randomize_IDRiD_71.jpg', 'IDRiD_Randomize_IDRiD_73.jpg',
'IDRiD_Randomize_IDRiD_60.jpg', 'IDRiD_Randomize_IDRiD_72.jpg',
'IDRiD_Randomize_IDRiD_64.jpg', 'IDRiD_Reconstruct', 'IDRiD_Randomize_IDRiD_68.jpg',
'IDRiD_Randomize_IDRiD_63.jpg', 'IDRiD_Randomize_IDRiD_81.jpg', 'IDRiD_Randomize_IDRiD_76.jpg',
'IDRiD_Randomize_IDRiD_55.jpg', 'IDRiD_Randomize_IDRiD_56.jpg', 'IDRiD_Randomize_IDRiD_69.jpg',
'IDRiD_Randomize_IDRiD_78.jpg', 'IDRiD_Randomize_IDRiD_57.jpg', 'IDRiD_Randomize_IDRiD_75.jpg',
'IDRiD_Randomize_IDRiD_66.jpg']

val_imgs = ['IDRiD_Randomize_IDRiD_80.jpg', 'IDRiD_Randomize_IDRiD_77.jpg',
'IDRiD_Randomize_IDRiD_58.jpg', 'IDRiD_Randomize_IDRiD_74.jpg',
'IDRiD_Randomize_IDRiD_67.jpg']

base_dir = 'Patho-GAN/Test'

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

for subdir in train_imgs:
  if subdir != 'IDRiD_Reconstruct':
    for img in os.listdir(os.path.join(base_dir, subdir)):
      if img.find('lesion_map') == -1 and img.find('.py') == -1:
        circle_img = circle_crop(os.path.join(base_dir, subdir, img))
        new_img = circle_img[50:-50, :]
        cv2.imwrite(os.path.join(train_dir, img), new_img)
        #shutil.copyfile(os.path.join(base_dir, subdir, img), os.path.join(train_dir, img))

for subdir in val_imgs:
  if subdir != 'IDRiD_Reconstruct':
    for img in os.listdir(os.path.join(base_dir, subdir)):
      if img.find('lesion_map') == -1 and img.find('.py') == -1:
        circle_img = circle_crop(os.path.join(base_dir, subdir, img))
        new_img = circle_img[50:-50, :]
        cv2.imwrite(os.path.join(val_dir, img), new_img)
        #shutil.copyfile(os.path.join(base_dir, subdir, img), os.path.join(val_dir, img))
    
