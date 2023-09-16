import os, shutil
import numpy as np
import pandas as pd

df = pd.read_csv('messidor_2_ethnicities.csv')
caucasian_imgs = df[df['ethnicity']==1]['filename']

src_dir = 'MESSIDOR-2/original'
dst_dir = 'MESSIDOR-2/original_caucasian'

for filename in caucasian_imgs:
  shutil.copyfile(os.path.join(src_dir, filename), os.path.join(dst_dir, filename))