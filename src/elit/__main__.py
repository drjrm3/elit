#!/usr/bin/env python3

import sys

from elit import core, utils

def main():
    args = utils.get_args()
    mode = sys.argv[1]

    if mode == "train":
        core.train(args.images, args.masks)
    elif mode == "infer":
        core.infer(args)
    else:
        sys.exit("ERROR: Unknown mode '{}'".format(mode))

main()
