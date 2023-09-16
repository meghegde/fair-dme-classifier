import numpy as np
import pandas as pd
import os, shutil, pathlib
import splitfolders

os.mkdir('data/MESSIDOR-2/')
os.mkdir('data/MESSIDOR-2/healthy')
os.mkdir('data/MESSIDOR-2/dme')

## Copy images according to healthy/dme status
df = pd.read_csv('messidor_data.csv')
#src_dir = 'MESSIDOR-2/processed_rectangle'
src_dir = 'MESSIDOR-2/processed_rectangle_caucasian'
healthy_dir = 'data/MESSIDOR-2/healthy'
dme_dir = 'data/MESSIDOR-2/dme'
for filename in os.listdir(src_dir):
    status = df.loc[df['image_id'] == filename]['adjudicated_dme']
    if len(status) == 1:
        if status.item() == 0:
            shutil.copyfile(os.path.join(src_dir, filename), os.path.join(healthy_dir, filename))
        elif status.item() == 1:
            shutil.copyfile(os.path.join(src_dir, filename), os.path.join(dme_dir, filename))

os.mkdir('data/dme_clf_cauc_only')
## Split these into train and validation sets
indir = 'data/MESSIDOR-2'
outdir = 'data/dme_clf_cauc_only'
splitfolders.ratio(indir, output=outdir,
                   seed=1337,
                   #ratio=(.8, .1, .1),
                   ratio=(.7, .15, .15),
                   group_prefix=None,
                   move=False)
