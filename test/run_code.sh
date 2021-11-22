#!/usr/bin/env bash

export PYTHONPATH=$(pwd)/../src:$PYTHONPATH

python3 -m elit train
