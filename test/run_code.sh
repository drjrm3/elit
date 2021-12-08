#!/usr/bin/env bash

export PYTHONPATH=$(pwd)/../src:$PYTHONPATH

DATADIR=$(readlink -f ../data)
#IMAGES=$DATADIR/cropped_images_new.npy
IMAGES=/data/cropped_images_subset.npy
#MASKS=$DATADIR/cropped_masks_new.npy
MASKS=/data/cropped_masks_subset.npy
IMAGE=/data/3d_stack8_10nmfov.tif

echo $DATADIR
ls $DATADIR

#python3 -m elit train --images $IMAGES --masks $MASKS

nvidia-docker build .. -t elit

nvidia-docker run \
	-v $DATADIR:/data \
	--gpus all \
	--ipc=host \
	--ulimit memlock=-1 \
	--ulimit stack=67108864 \
	elit \
	elit train --images $IMAGES --masks $MASKS --models 20 --cycles 2000 --out /data/01/model2.pkl
	#elit infer --image $IMAGE --models /data/01/model2.pkl
