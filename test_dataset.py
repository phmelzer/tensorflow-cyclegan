import functools

import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
import tqdm

import data
import module
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


#directorys and datasets
#datasets_dir =  '/home/mam-jr/GAN/traindata/'
datasets_dir = 'datasets/'
dataset = 'depthmaps_validierung_real_250/'
output_dir = 'outputs/'
training_dir = '02 results_200_epochs_no_normalization/'
weights_name = "weights/weights_G_A2B_13_27600.h5"
model_name = "models/model_G_B2A_13.h5"

#==============================================================================
#=                                    data                                    =
#==============================================================================

batch_size =1
load_size=128
crop_size=128
input_channels = 1
output_channels = 1
dim = 64
g_downsamples = 1
n_blocks = 6
norm = 'none'

A_img_paths_test = py.glob(py.join(datasets_dir, dataset), '*.png')
B_img_paths_test = py.glob(py.join(datasets_dir, dataset), '*.png')
A_B_dataset_test, len_dataset = data.make_zip_dataset_test(A_img_paths_test, batch_size, load_size, crop_size, training=False, repeat=True)
print(len_dataset)

# =                                   models                                   =
# ==============================================================================
G_A2B = module.ResnetGenerator(input_shape=(crop_size, crop_size, input_channels),
                                output_channels=output_channels,
                                dim=dim,
                                n_downsamplings=g_downsamples,
                                n_blocks=n_blocks,
                                norm=norm)

G_A2B.load_weights(output_dir + training_dir + weights_name)


#model = load_model(output_dir + training_dir + model_name)

image = 0


for A in tqdm.tqdm(A_B_dataset_test, desc='Inner Epoch Loop', total=len_dataset):
    y = G_A2B.predict([A])
    #y = model.predict([A])
    y = y.reshape(1,128, 128)
    y += 1
    y = y * 255 / 2
    A = A.numpy()
    A = A.reshape(1,128,128)
    A += 1
    A = A * 255 / 2
    img = im.immerge(np.concatenate([A, y], axis=0), n_rows=1)
    plt.imsave(output_dir + training_dir + 'validation/image_validation_{}.jpg'.format(image), img, cmap='gray')
    image +=1

    if image == len_dataset:
        break



