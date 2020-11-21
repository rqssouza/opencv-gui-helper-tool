#!/bin/env python3

import cv2 as cv
import numpy as np
import argparse
import tuner.tuner as tuner


def process(image, args):
    adj_k = lambda ksize : ksize + (ksize + 1) % 2
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    smoothed_img = cv.GaussianBlur(image, (adj_k(args.kernel_size), adj_k(args.kernel_size)), sigmaX=0, sigmaY=0)

    return ((1, 2), [gray, cv.Canny(smoothed_img, args.threshold1, args.threshold2)])


CFG = [
    ['kernel_size', 13, 20],
    ['threshold1', 28, 255],
    ['threshold2', 115, 255],
]


if __name__ == '__main__':
    tuner.Tuner_App(
        process, 
        CFG,
        'Edge Detection',
        'Tune edge detection parameters',
    )