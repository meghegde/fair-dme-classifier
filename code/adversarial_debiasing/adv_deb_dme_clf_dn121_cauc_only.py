import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, utils
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, shutil, pathlib
from keras.applications import DenseNet121
from keras.applications.densenet import preprocess_input

## Helper Functions
def get_img_array(img_path, size):
    img = keras.utils.load_img(img_path, target_size=size) # PIL Image
    array = keras.utils.img_to_array(img) # NumPy Array
    array = np.expand_dims(array, axis=0) # "Batch"
    return array

## N.B. Ethnicity: 1 = Caucasian, 0 = African
## DME: 1 = DME, 0 = Healthy

## Process Data

# Train

train_dir_dme = 'data/dme_clf_cauc_only_small/train/dme'
train_dir_healthy = 'data/dme_clf_cauc_only_small/train/healthy'

X_train_imgs = []
y_train = []
z_train = []

for img_dir in [train_dir_dme, train_dir_healthy]:
  for img in os.listdir(img_dir):
    img_array = get_img_array(os.path.join(img_dir, img), (533, 800))
    X_train_imgs.append(img_array)
    if img_dir==train_dir_dme:
        y_train.append(1)
    else:
        y_train.append(0)
    z_train.append(1)

# Images
X_train_imgs = np.array(X_train_imgs)
X_train_imgs = X_train_imgs.reshape(X_train_imgs.shape[0], 533, 800, 3)
# DME labels
y_train = pd.DataFrame(np.array(y_train))
y_train.head(2)
# Ethnicity labels (protected characteristic)
z_train = np.array(z_train)
Z_train = pd.DataFrame(z_train)
Z_train.head(2)

# Validation

val_dir_dme = 'data/dme_clf_cauc_only_small/val/dme'
val_dir_healthy = 'data/dme_clf_cauc_only_small/val/healthy'

X_val_imgs = []
y_val = []
z_val = []

for img_dir in [val_dir_dme, val_dir_healthy]:
  for img in os.listdir(img_dir):
    img_array = get_img_array(os.path.join(img_dir, img), (533, 800))
    X_val_imgs.append(img_array)
    if img_dir==val_dir_dme:
        y_val.append(1)
    else:
        y_val.append(0)
    z_val.append(1)

# Images
X_val_imgs = np.array(X_val_imgs)
X_val_imgs = X_val_imgs.reshape(X_val_imgs.shape[0], 533, 800, 3)
# DME labels
y_val = pd.DataFrame(np.array(y_val))
y_val.head(2)
# Ethnicity labels (protected characteristic)
z_val = np.array(z_val)
Z_val = pd.DataFrame(z_val)
Z_val.head(2)


# CLASSIFIER
input_shape = (533, 800, 3)	
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)

clf_inputs = layers.Input(shape=input_shape)
x = preprocess_input(clf_inputs)
x = base_model(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.7)(x)
clf_outputs = layers.Dense(1, activation='sigmoid')(x)
clf = models.Model(inputs=[clf_inputs], outputs=[clf_outputs])

clf.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
clf.load_weights('models/dn121_cauc_only_31_08.keras')

# ADVERSARY
for i in range(2):
    clf.layers[i].trainable = False

ll = clf.layers[2].output
ll = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(ll)
ll = layers.MaxPooling2D(pool_size=2)(ll)
ll = layers.Flatten()(ll)
adv_outputs = layers.Dense(1, activation="sigmoid")(ll)

adv = keras.Model(inputs=clf.input,outputs=adv_outputs)

adv.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

### Pre-train adversary
##callbacks = [keras.callbacks.ModelCheckpoint(filepath='models/dn121_cauc_only_adv.keras', save_best_only=True, monitor='val_loss')]
##adv.fit(X_train_imgs, Z_train.values.astype('float32'), validation_data=(X_val_imgs, Z_val.values.astype('float32')), epochs=5, batch_size=8, callbacks=callbacks)

# Load pre-trained weights
adv.load_weights('models/dn121_cauc_only_adv.keras')

## Combined Model
clf_w_adv = keras.Model(inputs=[clf_inputs], outputs=[clf_outputs+adv_outputs])

def make_trainable(net, flag):
  net.trainable = flag
  for layer in net.layers:
      layer.trainable = flag
  return net
  
make_trainable(clf, True)
make_trainable(adv, False)
clf_w_adv.compile(loss=['binary_crossentropy'], optimizer='adam')

n_iter = 3
batch_size = 8

for idx in range(n_iter):
  # train adversarial
  make_trainable(clf, False)
  make_trainable(adv, True)
  adv.fit(X_train_imgs, Z_train.values.astype('float32'),
          validation_data=(X_val_imgs, Z_val.values.astype('float32')),
          batch_size=batch_size, epochs=1)
  
  # train classifier 
  make_trainable(clf, True)
  make_trainable(adv, False)  
  #clf_w_adv.fit(X_train_imgs, [y_train.values]+np.hsplit(Z_train.values.astype('float32'), Z_train.values.shape[1]), validation_data=(X_val_imgs, [y_val.values]+np.hsplit(Z_val.values.astype('float32'), Z_val.values.shape[1])), batch_size=len(X_train_imgs), epochs=5)
  clf_w_adv.fit(X_train_imgs, [y_train.values]+np.hsplit(Z_train.values.astype('float32'), Z_train.values.shape[1]), validation_data=(X_val_imgs, [y_val.values]+np.hsplit(Z_val.values.astype('float32'), Z_val.values.shape[1])), batch_size=8, epochs=5)
  
  # save models
  clf.save('models/adv_deb_dn121_cauc_only/clf.keras')
  adv.save('models/adv_deb_dn121_cauc_only/adv.keras')
  clf_w_adv.save('models/adv_deb_dn121_cauc_only/clf_w_adv.keras')
