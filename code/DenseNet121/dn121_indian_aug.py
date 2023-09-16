import numpy as np
import pandas as pd
import os, shutil, pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, utils
from tensorflow.keras.utils import image_dataset_from_directory
from keras.applications import DenseNet121
from keras.applications.densenet import preprocess_input

new_base_dir = pathlib.Path('data/dme_clf_indian_aug_small')

train_dataset = image_dataset_from_directory(
    new_base_dir / 'train',
    image_size=(533, 800),
	label_mode='binary',
    shuffle=True,
    batch_size=16)
validation_dataset = image_dataset_from_directory(
    new_base_dir / 'val',
    image_size=(533, 800),
	label_mode='binary',
    shuffle=True,
    batch_size=16)
	
input_shape = (533, 800, 3)	
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)

i = layers.Input(shape=input_shape)
x = preprocess_input(i)
x = base_model(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.7)(x)
x = layers.Dense(1, activation='sigmoid')(x)
model = models.Model(inputs=[i], outputs=[x])

model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

log_filename = 'logs/dn121_indian_aug_31_08.csv'
history_logger = tf.keras.callbacks.CSVLogger(log_filename, separator=",", append=True)
callbacks = [
    history_logger,
    keras.callbacks.ModelCheckpoint(
        filepath="models/dn121_indian_aug_31_08.keras",
        save_best_only=True,
        monitor="val_loss"),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, start_from_epoch=5),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=1e-6)
]

num_epochs = 20

history = model.fit(train_dataset,
                    validation_data=validation_dataset,
                    epochs=num_epochs,
                    callbacks=callbacks)
