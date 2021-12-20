#!/usr/bin/env python3

import os
import sys
import pickle

import numpy as np
from skimage import io

from elit import utils
from elit.core import train, infer

"""
Notebook logic:
0) Initialize image = synthetic image + generate masks
1) Train:
    - Input: Images + Masks
    - Output: esnembel of models
2) Inference:
    - Input: ensemble of models + image
    - Output: Coordinates of atoms (masks)
3+) Train + Inference:
    - Train:
        Input: Images + last Inference output as masks
        Output: Ensemble of models
    - Inference:
        Input: Ensemble of models + image
        Output: Coordaintes of atoms (masks)
"""

def main():
    args = utils.get_args()
    mode = sys.argv[1]

    # Create output directory
    out_dir = os.path.dirname(args.out)
    if not os.path.exists(out_dir):
        os.makedirs(os.path.dirname(args.out))

    if mode == "train":
        # Train model
        images = np.load(args.images)
        masks = np.load(args.masks)
        smodel, ensemble = train(images, masks, args.models, args.cycles,
                                 nprocs=args.nprocs)

        # Save smodel, ensemble to file
        with open(args.out, 'wb') as fout:
            pickle.dump((smodel, ensemble), fout)
    elif mode == "infer":
        # Load the model
        with open(args.models, 'rb') as finp:
            models = pickle.load(finp)
        (model, ensemble) = models
        model.load_state_dict(ensemble[7])

        # Perform inference
        img_data = io.imread(args.image)
        decoded_imgs, coords, masks = infer(img_data, model)
        
        # Save output
        with open(args.out, 'wb') as fout:
            pickle.dump((decoded_imgs, coords, masks), fout)
    else:
        sys.exit("ERROR: Unknown mode '{}'".format(mode))

main()
