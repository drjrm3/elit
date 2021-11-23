#!/usr/bin/env python3

import argparse

def get_args():
    parser = argparse.ArgumentParser(prog='elit')
    subparser = parser.add_subparsers(help="Modes")

    train = subparser.add_parser("train", help="Train elit models")
    train.add_argument("--images", required=True, type=str)
    train.add_argument("--masks", required=True, type=str)

    infer = subparser.add_parser("infer", help="Infer from models")
    infer.add_argument("--image", required=True, type=str)
    infer.add_argument("--models", required=True, type=str)

    args = parser.parse_args()

    return args