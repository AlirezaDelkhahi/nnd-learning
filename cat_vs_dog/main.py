import os
import PIL
import glob
import shutil
import random
import pandas as pd
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Add, MaxPooling2D, Dense, BatchNormalization, Dropout, Flatten

import warnings

warnings.filterwarnings('ignore')


def is_valid_image(image_path):
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except (IOError, SyntaxError, PIL.UnidentifiedImageError):
        print(image_path)
        correpted_images.append(image_path)
        return False


correpted_images = []
# obtain all the dogs images located in the designated folder
dog_images = glob.glob(r'C:\Users\<user_name>\Downloads\Compressed\archive\PetImages\Dog\*')
for link in dog_images:
    is_valid_image(link)
for link in correpted_images:
    dog_images.remove(link)
correpted_images = []

cat_images = glob.glob(r'C:\Users\<user_name>\Downloads\Compressed\archive\PetImages\Cat\*')
for link in cat_images:
    is_valid_image(link)
for link in correpted_images:
    cat_images.remove(link)

random.shuffle(cat_images)
train_cats = cat_images[:11251]
val_cats = cat_images[11251:11876]
test_cats = cat_images[11876:]

random.shuffle(dog_images)
train_dogs = dog_images[:11251]
val_dogs = dog_images[11251:11876]
test_dogs = dog_images[11876:]

os.mkdir(r'C:\Users\<user_name>\Desktop\train')
os.mkdir(r'C:\Users\<user_name>\Desktop\val')
os.mkdir(r'C:\Users\<user_name>\Desktop\test')
os.mkdir(r'C:\Users\<user_name>\Desktop\train\cats')
os.mkdir(r'C:\Users\<user_name>\Desktop\train\dogs')
os.mkdir(r'C:\Users\<user_name>\Desktop\val\cats')
os.mkdir(r'C:\Users\<user_name>\Desktop\val\dogs')
os.mkdir(r'C:\Users\<user_name>\Desktop\test\cats')
os.mkdir(r'C:\Users\<user_name>\Desktop\test\dogs')

for image in train_dogs:
    shutil.copy(image, r'C:\Users\<user_name>\Desktop\train\dogs')

for image in train_cats:
    shutil.copy(image, r'C:\Users\<user_name>\Desktop\val\cats')

for image in val_dogs:
    shutil.copy(image, r'C:\Users\<user_name>\Desktop\test\dogs')

for image in val_cats:
    shutil.copy(image, r'C:\Users\<user_name>\Desktop\train\cats')

for image in test_dogs:
    shutil.copy(image, r'C:\Users\<user_name>\Desktop\train\dogs')

for image in test_cats:
    shutil.copy(image, r'C:\Users\<user_name>\Desktop\val\cats')

train_gen = ImageDataGenerator(rescale=1. / 255,
                               shear_range=0.1,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               width_shift_range=0.1,
                               height_shift_range=0.1)
val_gen = ImageDataGenerator(rescale=1. / 255)
test_gen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_gen.flow_from_directory(
    r'C:\Users\<user_name>\Desktop\train',
    target_size=(224, 224),
    batch_size=128,
    class_mode='binary')

val_generator = val_gen.flow_from_directory(
    r'C:\Users\<user_name>\Desktop\val',
    target_size=(224, 224),
    batch_size=128,
    class_mode='binary')

test_generator = test_gen.flow_from_directory(
    r'C:\Users\<user_name>\Desktop\test',
    target_size=(224, 224),
    batch_size=128,
    class_mode='binary')

# initialise the model
model = tf.keras.models.Sequential()

# first layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# 2nd layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# 3rd layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# 4tg layer
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# flattening
model.add(Flatten())

# fully connected
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# output layer
model.add(Dense(1, activation='sigmoid'))

opt = tf.optimizers.RMSprop(learning_rate=0.001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=0, patience=3)
history = model.fit(train_generator, epochs=10, validation_data=val_generator, callbacks=callback)
results = pd.DataFrame(history.history)
results.tail()
results.loc[:, ['loss', 'val_loss']].plot()
results.loc[:, ['accuracy', 'val_accuracy']].plot()
model.evaluate(test_generator)
