#!/usr/bin/env python3

import sys

import numpy as np
import atomai as aoi
from atomai.transforms import datatransform
from atomai.trainers import EnsembleTrainer
from atomai.predictors import SegPredictor
from atomai.utils import create_lattice_mask
from skimage import io


def train(images, masks, models, cycles):
    """ Train an ensemble of models from saved numpy files of images and masks.
        Input = Images + Masks
        Output = Ensemble of models
    Args:
        images (np.array): Numpy array of images (num_images, img_dim, img_dim)
        masks (np.array): Numpy array of masks (num_images, img_dim, img_dim)
        models (int): Number of ensemble models to train.
        cycles (int): Number of training cycles.
    Returns:
        smodel (atomai.nets.fcnn.Unet): ???
        ensembe (dict): ???
    """
    dxform = datatransform(1, gauss_noise=[2000, 3000], poisson_noise=[30, 45],
                           blur=False, contrast=True, zoom=True, resize=[2, 1],
                           seed=1)

    x_train, y_train = dxform.run(x_train, y_train[..., None])

    etrainer = EnsembleTrainer("Unet", nb_classes=1, with_dilation=False,
                               batch_norm=True, nb_filters=64,
                               layers=[2, 3, 3, 4])
    etrainer.compile_ensemble_trainer(training_cycles=cycles,
                                      compute_accuracy=True,
                                      swa=True, memory_alloc=0.5)
    smodel, ensemble = etrainer.train_ensemble_from_scratch(images, masks,
                                                            models=models)

    return smodel, ensemble

def infer(img_data, model):
    """ Use the ensemble of models to infer atomic coordinates of an image.
        Input = model + image.
        Output = images, Coordinates of atoms, masks
    Args:
        image_data (???): ???
        model (???): ???
        
    """

    # Predict the coordinates
    decoded_imgs, coords = SegPredictor(model, use_gpu=True).run(img_data)

    # Filter images based on {TODO}
    filtered_coords = {}
    for _key, _coords in coords.items():
        filtered_coords[_key] = _coords[_coords[:, -1] == 0]

    # Create the masks
    nimgs = len(coords.keys())
    masks = np.zeros((nimgs, *img_data.shape[1:]))
    for k in coords.keys():
        masks[k] = create_lattice_mask(img_data[k], filtered_coords[k][:, :-1])

    return decoded_imgs, coords, masks






