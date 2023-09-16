import os, shutil, pathlib
import glob

os.mkdir('data/dme_clf_messidor_idrid')
os.mkdir('data/dme_clf_messidor_idrid/train')
os.mkdir('data/dme_clf_messidor_idrid/train/dme')
os.mkdir('data/dme_clf_messidor_idrid/train/healthy')
os.mkdir('data/dme_clf_messidor_idrid/val')
os.mkdir('data/dme_clf_messidor_idrid/val/dme')
os.mkdir('data/dme_clf_messidor_idrid/val/healthy')


# # Copy MESSIDOR-2 images
# Train - DME - 50 images
src = 'data/dme_clf_cauc_only_small/train/dme'
dst = 'data/dme_clf_messidor_idrid/train/dme'
i = 0
for img in os.listdir(src):
    shutil.copyfile(os.path.join(src, img), os.path.join(dst, img))
    i += 1
    if i > 49:
        break
# Validation - DME - 10 images
src = 'data/dme_clf_cauc_only_small/val/dme'
dst = 'data/dme_clf_messidor_idrid/val/dme'
i = 0
for img in os.listdir(src):
    shutil.copyfile(os.path.join(src, img), os.path.join(dst, img))
    i += 1
    if i > 9:
        break
# Train - Healthy - 100 images
src = 'data/dme_clf_cauc_only_small/train/healthy'
dst = 'data/dme_clf_messidor_idrid/train/healthy'
for img in os.listdir(src):
    shutil.copyfile(os.path.join(src, img), os.path.join(dst, img))
# Validation - Healthy - 20 images
src = 'data/dme_clf_cauc_only_small/val/healthy'
dst = 'data/dme_clf_messidor_idrid/val/healthy'
i = 0
for img in os.listdir(src):
    shutil.copyfile(os.path.join(src, img), os.path.join(dst, img))
    i += 1
    if i > 19:
        break

# # Copy IDRiD images
# Train - 50 images
#src = 'IDRiD/aug_processed/train'
#src = 'data/dme_clf_indian_aug/train/dme'
src = 'IDRiD/original_dme'
dst = 'data/dme_clf_messidor_idrid/train/dme'
i = 0
for img in os.listdir(src):
    shutil.copyfile(os.path.join(src, img), os.path.join(dst, img))
    i += 1
    if i > 49:
        break
# Validation - 10 images
#src = 'IDRiD/aug_processed/val'
#src = 'data/dme_clf_indian_aug/val/dme'
src = 'IDRiD/original_dme'
dst = 'data/dme_clf_messidor_idrid/val/dme'
i = 0
for img in os.listdir(src)[49:]:
    shutil.copyfile(os.path.join(src, img), os.path.join(dst, img))
    i += 1
    if i > 9:
        break



