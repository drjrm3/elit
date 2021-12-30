#!/usr/bin/env python3

import sys

import numpy as np
from atomai.transforms import datatransform
from atomai.trainers import EnsembleTrainer
from atomai.predictors import SegPredictor
from atomai.utils import create_lattice_mask
from skimage import io

def cycle(sim_images, sim_masks, images, num_models, training_cycles,
          add_noise=False):
    """ Train on a set of simulated images, simulated masks, then use real
        'images' to generate a real 'mask'.
    Args: TODO
    Returns: TODO
    """

    # Train the model
    model, ensemble = train(sim_images, sim_masks, num_models,
                            training_cycles, add_noise=add_noise)

    # TODO: Select best model
    model.load_state_dict(ensemble[0])

    # Perform inference on the 'real' data
    decoded_imgs, coords, masks = infer(images, model)

    return masks

def train(images, masks, num_models, training_cycles, add_noise=False, nprocs=1):
    """ Train an ensemble of models from saved numpy files of images and masks.
        Input = Images + Masks
        Output = Ensemble of models
    Args:
        images (np.array): Numpy array of images (num_images, img_dim, img_dim)
        masks (np.array): Numpy array of masks (num_images, img_dim, img_dim)
        models (int): Number of ensemble models to train.
        train_cycles (int): Number of training cycles.
        add_noise (bool): Whether or not to add noise - useful for first
            simulated data.
        nprocs (int): The number of processes (and GPU's) to use for training.
    Returns:
        model (atomai.nets.fcnn.Unet): ???
        ensembe (dict): ???
    """
    if add_noise:
        dxform = datatransform(1, gauss_noise=[2000, 3000], poisson_noise=[30, 45],
                               blur=False, contrast=True, zoom=True, resize=[2, 1],
                               seed=1)
        images, masks = dxform.run(images, masks[..., None])

    etrainer = EnsembleTrainer("Unet", nb_classes=1, with_dilation=False,
                               batch_norm=True, nb_filters=64,
                               layers=[2, 3, 3, 4])
    etrainer.compile_ensemble_trainer(training_cycles=training_cycles,
                                      compute_accuracy=True,
                                      swa=True, memory_alloc=0.5)
    model, ensemble = etrainer.train_ensemble_from_scratch(images, masks,
                                                           n_models=num_models)
    return model, ensemble

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






