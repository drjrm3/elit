#!/usr/bin/env python3

#import numpy as np
#from atomai.transforms import datatransform
from atomai.trainers import EnsembleTrainer
#from atomai.predictors import SegPredictor
#from atomai.utils import create_lattice_mask
#from skimage import io

from torch import cuda

from multiprocessing import Pool

def train_with_seed(images, masks, cycles, batch_seed, models):
    """ Train a single model """
    etrainer = EnsembleTrainer("Unet", nb_classes=1, with_dilation=False,
                               batch_norm=True, nb_filters=64,
                               layers=[2, 3, 3, 4])
    etrainer.compile_ensemble_trainer(training_cycles=cycles,
                                      compute_accuracy=True,
                                      swa=True, memory_alloc=0.5,
                                      batch_seed=batch_seed)
    smodel, ensemble = etrainer.train_ensemble_from_scratch(images, masks,
                                                            n_models=models)
    return smodel, ensemble

def train_with_seed_wrapper(args):
    images, masks, cycles, batch_seed, models, outfile = args
    # TODO: This is hacky. Just testing out if I can split devices within map
    cuda.set_device(batch_seed)
    smodel, ensemble = train_with_seed(images, masks, cycles, batch_seed, models)

    # TODO: Save smodel, ensemble to 'outfile'

    return True

def train_in_parallel(images, masks, cycles, models, nprocs):

    ensemble = {}
    smodel = None
    #print(cuda.device_count())
    #print(cuda.memory_reserved())

    pool = Pool(processes=nprocs)
    args = []
    for imodel in range(models):
        args.append(
            (images, masks, cycles, imodel, 1, '/path/to/outfile' + str(imodel))
        )
    results = pool.map(train_with_seed_wrapper, args)
    print(results)

    return smodel, ensemble
