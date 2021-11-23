#!/usr/bin/env bash

export PYTHONPATH=$(pwd)/../src:$PYTHONPATH

IMAGES=$(readlink -f ../data/cropped_images_new.npy)
MASKS=$(readlink -f ../data/cropped_masks_new.npy)

python3 -m elit train --images $IMAGES --masks $MASKS
