#!/bin/env python3

import cv2 as cv
import numpy as np
import argparse
import tuner.tuner as tuner


def mag(gradient_x, gradient_y):
    gradient_mag = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    return np.uint8(255 * (gradient_mag / np.max(gradient_mag)))


def ths(img, ths_min, ths_max):
    ret = np.zeros_like(img)
    ret[(img >= ths_min) & (img <= ths_max)] = 255
    return ret 


def process(image, args):
    adj_k = lambda ksize : ksize + (ksize + 1) % 2
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gradient_x = cv.Sobel(
        src = gray,
        ddepth = cv.CV_64F,
        dx = 1,
        dy = 0,
        ksize = adj_k(args.kernel_size),
    )
    gradient_y = cv.Sobel(
        src = gray,
        ddepth = cv.CV_64F,
        dx = 0,
        dy = 1,
        ksize = adj_k(args.kernel_size),
    )

    gradient_mag = ths(mag(gradient_x, gradient_y), args.ths_min, args.ths_max)
    return ((1, 1), [gradient_mag])


CFG = [
    ['kernel_size', 3, 30],
    ['ths_min', 20, 255],
    ['ths_max', 100, 255],
]


if __name__ == '__main__':
    tuner.Tuner_App(
        process, 
        CFG,
        'Gradient Magnitude',
        'Tune gradient magnitude parameters',
    )