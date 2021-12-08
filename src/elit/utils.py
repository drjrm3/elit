#!/usr/bin/env python3

import argparse

def get_args():
    parser = argparse.ArgumentParser(prog='elit')
    subparser = parser.add_subparsers(help="Modes")

    train = subparser.add_parser("train", help="Train elit models")
    train.add_argument("--images", required=True, type=str)
    train.add_argument("--masks", required=True, type=str)
    train.add_argument("--models", type=int, default=20)
    train.add_argument("--cycles", type=int, default=2000)
    train.add_argument("--out", type=str, required=True)

    infer = subparser.add_parser("infer", help="Infer from models")
    infer.add_argument("--image", required=True, type=str)
    infer.add_argument("--models", required=True, type=str)

    args = parser.parse_args()

    return args
