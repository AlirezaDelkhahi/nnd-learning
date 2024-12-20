import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import plotly.express as px
import scipy as sp

from scipy import ndimage
from shutil import copyfile
from tensorflow.keras.layers import Conv2D,Add,MaxPooling2D, Dense, BatchNormalization,Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class_names = ['Cat', 'Dog']

n_dogs = len(os.listdir('/home/alireza/Desktop/archive/PetImages/Dog'))
n_cats = len(os.listdir('/home/alireza/Desktop/archive/PetImages/Cat'))

n_images = [n_cats, n_dogs]
px.pie(names=class_names, values=n_images)