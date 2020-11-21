#!/bin/env python3

import cv2 as cv
import numpy as np
import argparse
import tuner.tuner as tuner


def ths(img, ths_min, ths_max):
    ret = np.zeros_like(img)
    ret[(img >= ths_min) & (img <= ths_max)] = 255
    return np.uint8(ret)


def process(image, args):
    adj_k = lambda ksize : ksize + (ksize + 1) % 2
    adj_deg = lambda deg : np.deg2rad(deg - 90)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gradient_x = np.absolute(cv.Sobel(gray, cv.CV_64F, 1, 0, ksize = adj_k(args.kernel_size)))
    gradient_y = np.absolute(cv.Sobel(gray, cv.CV_64F, 0, 1, ksize = adj_k(args.kernel_size)))

    gradient_dir = ths(np.arctan2(gradient_y, gradient_x), adj_deg(args.ths_min), adj_deg(args.ths_max))
    return ((1, 1), [gradient_dir])


CFG = [
    ['kernel_size', 3, 30],
    ['ths_min', 0, 180],
    ['ths_max', 180, 180],
]


if __name__ == '__main__':
    tuner.Tuner_App(
        process, 
        CFG,
        'Gradient Direction',
        'Tune gradient direction parameters',
    )