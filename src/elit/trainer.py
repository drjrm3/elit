#!/usr/bin/env python3

#import numpy as np
#from atomai.transforms import datatransform
from atomai.trainers import EnsembleTrainer
#from atomai.predictors import SegPredictor
#from atomai.utils import create_lattice_mask
#from skimage import io
import pickle
from tempfile import TemporaryDirectory
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
    images, masks, cycles, batch_seed, models, outfile, device = args
    # TODO: This is hacky. Just testing out if I can split devices within map
    cuda.set_device(device)
    smodel, ensemble = train_with_seed(images, masks, cycles, batch_seed, models)

    with open(outfile, 'wb') as fout:
        pickle.dump((smodel, ensemble), fout)

    return True

def train_in_parallel(images, masks, cycles, models, nprocs):

    ensemble = {}
    smodel = None

    tmpdir = TemporaryDirectory()

    ## Get the number of models each processor should do
    nmodels_per_proc = [0 for _ in range(nprocs)]
    iproc = 0
    while sum(nmodels_per_proc) < models:
        nmodels_per_proc[iproc] += 1
        iproc += 1
        if iproc == nprocs:
            iproc = 0

    print(nmodels_per_proc)

    pool = Pool(processes=nprocs)
    args = []
    outfiles = []
    for iproc in range(nprocs):
        device = iproc
        outfile = tmpdir.name + '/' + str(iproc)
        outfiles.append(outfile)
        nmodels = nmodels_per_proc[iproc]
        args.append(
            (images, masks, cycles, iproc, nmodels, outfile, device)
        )
    results = pool.map(train_with_seed_wrapper, args)

    # Go through and read results

    return smodel, ensemble
