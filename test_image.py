import functools

import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
#import tf2lib as tl
#import tf2gan as gan
import tqdm

#import data
import module
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image

#Shared Networkparameter
dim = 64 #number of filters
norm = 'none' #choices=['none','batch_norm','instance_norm','layer_norm']
lr=0.0002
beta_1=0.5
adversarial_loss_mode='lsgan' #choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
gradient_penalty_mode='none' #choices=['none', 'dragan', 'wgan-gp'])
gradient_penalty_weight=10.0
cycle_loss_weight=11.0
identity_loss_weight=0.0
pool_size = 50

#Generator parameter
g_downsamples = 1 #number of convolutions
n_blocks = 6 #number of residualblocks

Generator = module.ResnetGenerator(input_shape=(128, 128, 1),
                                output_channels=1,
                                dim=64,
                                n_downsamplings=1,
                                n_blocks=6,
                                norm='none')

save_dir = 'C:/Users/mam-pm/PycharmProjects/cyclegan_depthmaps_tensorflow/venv/'
save_name = "CycleGAN_RL_21-09-16/"

Generator.load_weights(save_dir + save_name + 'cycle_gan_RL_sim_to_real_weights.h5')

image = cv2.imread('C:/Users/mam-pm/Desktop/100 depthmaps_validierung_sim_grey_100/test.png', cv2.IMREAD_GRAYSCALE)
image = np.array(image).reshape(-1, 128, 128, 1)
image = image / 255.0
#print(os.path.join(savedir, category, i))
y = Generator.predict(image)
y = y.reshape(128, 128)
y += 1
y = y * 255 / 2
print(y.shape, y.min(), y.max())
plt.imsave('image_new.jpg',y, cmap='gray')
