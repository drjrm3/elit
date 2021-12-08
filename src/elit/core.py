#!/usr/bin/env python3

import sys

import numpy as np
import atomai as aoi
from atomai.transforms import datatransform
from atomai.trainers import EnsembleTrainer
from atomai.predictors import SegPredictor
from skimage import io

def train(images_path, masks_path, models, cycles):
    """ Train an ensemble of models from saved numpy files of images and masks.
        Input = Images + Masks
        Output = Ensemble of models
    Args:
        images_path (str): Path to file containing a numpy array of images.
        masks_path (str): Path to file containing numpy array of images.
        models (int): Number of ensemble models to train.
        cycles (int): Number of training cycles.
    Returns:
        smodel (atomai.nets.fcnn.Unet): ???
        ensembe (dict): ???
    """
    # 1) Read data and process it with aoi.transforms.datatransform
    x_train = np.load(images_path)
    y_train = np.load(masks_path)

    print(type(y_train), y_train.shape)
    sys.exit(0)

    dxform = datatransform(1, gauss_noise=[2000, 3000], poisson_noise=[30, 45],
                           blur=False, contrast=True, zoom=True, resize=[2, 1],
                           seed=1)

    x_train, y_train = dxform.run(x_train, y_train[..., None])

    # 2) 'train_new'
    etrainer = EnsembleTrainer("Unet", nb_classes=1, with_dilation=False,
                               batch_norm=True, nb_filters=64,
                               layers=[2, 3, 3, 4])
    etrainer.compile_ensemble_trainer(training_cycles=cycles,
                                      compute_accuracy=True,
                                      swa=True, memory_alloc=0.5)
    print("nmodels: ", models)
    smodel, ensemble = etrainer.train_ensemble_from_scratch(x_train,
                                                            y_train,
                                                            models=models)

    return smodel, ensemble

def infer(image_path, models, model_num):
    """ Use the ensemble of models to infer atomic coordinates of an image.
        Input = Ensemble of models + image.
        Output = Coordinates of atoms.
    """

    # Load the model
    (smodel, ensemble) = models
    smodel.load_state_dict(ensemble[model_num])

    # Read image
    img_data = io.imread(image_path)

    # predict the coordinates
    decoded_imgs, coords = SegPredictor(smodel, use_gpu=True).run(img_data)

    # TODO - what is deocded_imgs? Does this turn into masks?

    print(type(coords))
    print(coords.keys())
    for k in coords.keys():
        print(k, coords[k].shape)


