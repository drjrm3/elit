#!/usr/bin/env python3

import os
import sys
import pickle

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

    if mode == "train":
        # Train model
        smodel, ensemble = train(args.images, args.masks,
                                 args.models, args.cycles)

        # Save smodel, ensemble to file
        print("args.out: ", args.out)
        out_dir = os.path.dirname(args.out)
        if not os.path.exists(out_dir):
            os.makedirs(os.path.dirname(args.out))
        with open(args.out, 'wb') as fout:
            pickle.dump((smodel, ensemble), fout)
    elif mode == "infer":
        # Open model
        print(args.models)
        with open(args.models, 'rb') as finp:
            models = pickle.load(finp)

        # Perform inference
        # TODO: Right now I'm randomly choosing model number 19 ... how do we make this decision?
        infer(args.image, models, 7)
    else:
        sys.exit("ERROR: Unknown mode '{}'".format(mode))

main()
