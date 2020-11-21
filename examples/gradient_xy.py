#!/bin/env python3

import cv2 as cv
import numpy as np
import argparse
import tuner.tuner as tuner


def scale(img):
    img = np.absolute(img)
    return np.uint8(255 * (img / np.max(img)))


def ths(img, ths_min, ths_max):
    ret = np.zeros_like(img)
    ret[(img >= ths_min) & (img <= ths_max)] = 255
    return ret 


def gradient_component(img, t, ths_min, ths_max, k_size):
    k_size = k_size + (k_size + 1) % 2
    if t == 'x':
        return ths(scale(cv.Sobel(
            src = img,
            ddepth = cv.CV_64F,
            dx = 1,
            dy = 0,
            ksize = k_size
        )), ths_min, ths_max)

    return ths(scale(cv.Sobel(
        src = img,
        ddepth = cv.CV_64F,
        dx = 0,
        dy = 1,
        ksize = k_size
    )), ths_min, ths_max)


def process(image, args):
    adj_k = lambda ksize : ksize + (ksize + 1) % 2
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gradient_x = gradient_component(gray, 'x', args.ths_min, args.ths_max, adj_k(args.kernel_size))
    gradient_y = gradient_component(gray, 'y', args.ths_min, args.ths_max, adj_k(args.kernel_size))
    return ((1, 2), [gradient_x, gradient_y])


CFG = [
    ['kernel_size', 3, 30],
    ['ths_min', 20, 255],
    ['ths_max', 100, 255],
]


if __name__ == '__main__':
    tuner.Tuner_App(
        process, 
        CFG,
        'Gradient XY',
        'Tune gradient X and Y components parameters',
    )