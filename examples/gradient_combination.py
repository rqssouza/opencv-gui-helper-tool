#!/bin/env python3

import cv2 as cv
import numpy as np
import argparse
import tuner.tuner as tuner


def ths(img, ths_min, ths_max):
    ret = np.zeros_like(img)
    ret[(img >= ths_min) & (img <= ths_max)] = 255
    return np.uint8(ret)


def scale(img):
    img = np.absolute(img)
    return np.uint8(255 * (img / np.max(img)))


def mag(gradient_x, gradient_y):
    gradient_mag = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    return np.uint8(255 * (gradient_mag / np.max(gradient_mag)))


def gradient_xy(img, t, k_size, ths_min, ths_max):
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


def gradient_mag(img, k_size, ths_min, ths_max):
    grad_x = cv.Sobel(
        src = img,
        ddepth = cv.CV_64F,
        dx = 1,
        dy = 0,
        ksize = k_size,
    )
    grad_y = cv.Sobel(
        src = img,
        ddepth = cv.CV_64F,
        dx = 0,
        dy = 1,
        ksize = k_size,
    )

    return ths(mag(grad_x, grad_y), ths_min, ths_max)


def gradient_dir(img, k_size, ths_min, ths_max):
    grad_x = np.absolute(cv.Sobel(img, cv.CV_64F, 1, 0, ksize = k_size))
    grad_y = np.absolute(cv.Sobel(img, cv.CV_64F, 0, 1, ksize = k_size))

    return ths(np.arctan2(grad_y, grad_x), ths_min, ths_max).astype(np.uint8)


def process(image, args):
    adj_k = lambda ksize : ksize + (ksize + 1) % 2
    adj_deg = lambda deg : np.deg2rad(deg - 90)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    grad_x = gradient_xy(gray, 'x', adj_k(args.x_ksize), args.x_ths_min, args.x_ths_max)
    grad_y = gradient_xy(gray, 'y', adj_k(args.y_ksize), args.y_ths_min, args.y_ths_max)
    grad_mag = gradient_mag(gray, adj_k(args.mag_ksize), args.mag_ths_min, args.mag_ths_max)
    grad_dir = gradient_dir(gray, adj_k(args.dir_ksize), adj_deg(args.dir_ths_min), adj_deg(args.dir_ths_max))

    return ((1, 1), [(grad_x & grad_y) | (grad_mag & grad_dir)])


CFG = [
    ['x_ksize', 3, 30],
    ['x_ths_min', 0, 255],
    ['x_ths_max', 0, 255],
    ['y_ksize', 3, 30],
    ['y_ths_min', 0, 255],
    ['y_ths_max', 0, 255],
    ['mag_ksize', 3, 30],
    ['mag_ths_min', 0, 255],
    ['mag_ths_max', 0, 255],
    ['dir_ksize', 3, 30],
    ['dir_ths_min', 0, 180],
    ['dir_ths_max', 0, 180],
]


if __name__ == '__main__':
    tuner.Tuner_App(
        process, 
        CFG,
        'Gradient Combination',
        'Tune parameters for the gradient combination'
    )