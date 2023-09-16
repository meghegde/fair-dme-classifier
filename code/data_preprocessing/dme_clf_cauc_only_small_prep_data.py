import os, shutil, pathlib

os.mkdir('data/dme_clf_cauc_only_small')
os.mkdir('data/dme_clf_cauc_only_small/train')
os.mkdir('data/dme_clf_cauc_only_small/train/dme')
os.mkdir('data/dme_clf_cauc_only_small/train/healthy')
os.mkdir('data/dme_clf_cauc_only_small/val')
os.mkdir('data/dme_clf_cauc_only_small/val/dme')
os.mkdir('data/dme_clf_cauc_only_small/val/healthy')
os.mkdir('data/dme_clf_cauc_only_small/test')
os.mkdir('data/dme_clf_cauc_only_small/test/dme')
os.mkdir('data/dme_clf_cauc_only_small/test/healthy')

src = 'data/dme_clf_cauc_only/train/dme'
dst = 'data/dme_clf_cauc_only_small/train/dme'

imgs = os.listdir(src)
for i in range(100):
    shutil.copyfile(os.path.join(src, imgs[i]), os.path.join(dst, imgs[i]))

src = 'data/dme_clf_cauc_only/train/healthy'
dst = 'data/dme_clf_cauc_only_small/train/healthy'

imgs = os.listdir(src)
for i in range(100):
    shutil.copyfile(os.path.join(src, imgs[i]), os.path.join(dst, imgs[i]))

src = 'data/dme_clf_cauc_only/val/dme'
dst = 'data/dme_clf_cauc_only_small/val/dme'

imgs = os.listdir(src)
for i in range(20):
    shutil.copyfile(os.path.join(src, imgs[i]), os.path.join(dst, imgs[i]))

src = 'data/dme_clf_cauc_only/val/healthy'
dst = 'data/dme_clf_cauc_only_small/val/healthy'

imgs = os.listdir(src)
for i in range(20):
    shutil.copyfile(os.path.join(src, imgs[i]), os.path.join(dst, imgs[i]))


