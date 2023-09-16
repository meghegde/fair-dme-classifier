import numpy as np
import pandas as pd
import os, shutil, pathlib
import splitfolders

os.mkdir('data/dme_clf_messidor_2_originals/')
os.mkdir('data/dme_clf_messidor_2_originals/train')
os.mkdir('data/dme_clf_messidor_2_originals/train/dme')
os.mkdir('data/dme_clf_messidor_2_originals/train/healthy')
os.mkdir('data/dme_clf_messidor_2_originals/val')
os.mkdir('data/dme_clf_messidor_2_originals/val/dme')
os.mkdir('data/dme_clf_messidor_2_originals/val/healthy')
os.mkdir('data/dme_clf_messidor_2_originals/test')

## Copy over the original (unprocessed) versions of each of the images
src_dir = 'MESSIDOR-2/original'

