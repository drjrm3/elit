#!/usr/bin/env python3

import numpy as np

x_train = np.load('../data/cropped_images_new.npy')
print(type(x_train), x_train.shape)
y_train = np.load('../data/cropped_masks_new.npy')
print(type(y_train), x_train.shape)
