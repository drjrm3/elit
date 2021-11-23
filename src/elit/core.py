#!/usr/bin/env python3

import numpy as np
import atomai as aoi
from atomai.transforms import datatransform
from atomai.trainers import EnsembleTrainer

def train(images_path, masks_path, nmodels=20):
    """ Train an ensembe of models from saved numpy files of images and masks.
    Args:
        images_path (str): Path to file containing a numpy array of images.
        masks_path (str): Path to file containing numpy array of images.
        nmodels (int): Number of ensemble models to train.
    Returns:
        smodel ???
        ensembe ???
    """
    # 1) Read data and process it with aoi.transforms.datatransform
    x_train = np.load(images_path)
    y_train = np.load(masks_path)

    dxform = datatransform(1, gauss_noise=[2000, 3000], poisson_noise=[30, 45],
                       blur=False, contrast=True, zoom=True, resize=[2, 1],
                       seed=1)

    x_train, y_train = dxform.run(x_train, y_train[..., None])

    # 2) 'train_new'
    etrainer = EnsembleTrainer("Unet", nb_classes=1, with_dilation=False,
                               batch_norm=True, nb_filters=64,
                               layers=[2, 3, 3, 4])
    etrainer.compile_ensemble_trainer(training_cycles=2000,
                                      compute_accuracy=True,
                                      swa=True, memory_alloc=0.5)
    smodel, ensemble = etrainer.train_ensemble_from_scratch(x_train,
                                                            y_train,
                                                            nmodels=20)



    pass

def infer(args):
    pass

