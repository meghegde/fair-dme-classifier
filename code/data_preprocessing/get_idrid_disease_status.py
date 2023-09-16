import os
import shutil
import numpy as np
import pandas as pd

#os.mkdir('IDRiD/original_healthy')
#os.mkdir('IDRiD/original_dme')

df = pd.read_csv('IDRiD/IDRiD_Disease_Grading_Training_Labels.csv')

src_dir = 'IDRiD/1. Original Images/a. Training Set'
healthy_dir = 'IDRiD/original_healthy'
dme_dir = 'IDRiD/original_dme'
for img in os.listdir(src_dir):
    img_name = img.rstrip('.jpg')
    dme = df[df['Image name']==img_name]['Risk of macular edema '].to_numpy()[0]
    if dme == 0:
        shutil.copyfile(os.path.join(src_dir, img), os.path.join(healthy_dir, img))
    else:
        shutil.copyfile(os.path.join(src_dir, img), os.path.join(dme_dir, img))
