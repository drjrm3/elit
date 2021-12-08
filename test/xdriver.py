#!/usr/bin/env python3

import numpy as np

x_train = np.load('../data/cropped_images_new.npy')
print(type(x_train), x_train.shape)
y_train = np.load('../data/cropped_masks_new.npy')
print(type(y_train), x_train.shape)

x_train = x_train[1:2000, :, :]
np.save('../data/cropped_images_subset.npy', x_train)
y_train = y_train[1:2000, :, :]
np.save('../data/cropped_masks_subset.npy', y_train)
