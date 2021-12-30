#!/usr/bin/env python3

import argparse

def get_args():
    """ Get arguments using argparse """
    parser = argparse.ArgumentParser(prog='elit')
    subparser = parser.add_subparsers(help="Modes")

    train = subparser.add_parser("train", help="Train elit models")
    train.add_argument("--images", type=str, required=True)
    train.add_argument("--masks", type=str, required=True)
    train.add_argument("--num_models", type=int, default=20)
    train.add_argument("--training_cycles", type=int, default=2000)
    train.add_argument("--out", type=str, required=True)

    infer = subparser.add_parser("infer", help="Infer from models")
    infer.add_argument("--image", type=str, required=True)
    infer.add_argument("--models", type=str, required=True)
    infer.add_argument("--out", type=str, required=True)

    args = parser.parse_args()

    return args
