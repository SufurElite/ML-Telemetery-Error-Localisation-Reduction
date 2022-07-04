"""
    This file will contain different multilateration methods, as we continue to work on reducing error.
    Currently, it has a reworked implementation from 'https://stackoverflow.com/questions/17756617/finding-an-unknown-point-using-weighted-multilateration'
"""
import numpy as np
import scipy.optimize as opt
import utils, argparse


def stack():
    """ Executes the multilat with weights, based on the stack overflow implementation """
    pass


def main(args):
    if args.type=="stack":
        stack()
    else:
        print("Sorry, please list a valid type. Currently, these are the supported types: 'stack'")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Personal information')
    parser.add_argument('--type', dest='type', type=str, help='Type of multilateration approach')

    args = parser.parse_args()

    main(args)