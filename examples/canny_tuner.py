#!/bin/env python3

import cv2 as cv
import numpy as np
import argparse
import tuner.tuner as tuner


CFG_FILE = '.canny_parameters.cfg'

FILTER_SIZE = 'filterSize'
THRESHOLD1 = 'threshold1'
THRESHOLD2 = 'threshold2'
DEFAULT_ATTRS = [
    [FILTER_SIZE, 13, 20],
    [THRESHOLD1, 28, 255],
    [THRESHOLD2, 115, 255],
]


def canny_calc(image, filter_size, threshold1, threshold2):
    filter_size = filter_size + (filter_size + 1) % 2
    smoothed_img = cv.GaussianBlur(image, (filter_size, filter_size), sigmaX=0, sigmaY=0)
    return cv.Canny(smoothed_img, threshold1, threshold2)


def process(image, cfg):
    shape = (1, 2)
    imgs = [image]

    imgs.append(canny_calc(
        image = image,
        filter_size = cfg.get_value(FILTER_SIZE),
        threshold1 = cfg.get_value(THRESHOLD1),
        threshold2 = cfg.get_value(THRESHOLD2),
    ))

    return (shape, imgs)


def main(image_path, cfg_file):
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    cfg = tuner.Tuner(
            image = image, 
            cfg = tuner.TunerCfg(cfg_file, DEFAULT_ATTRS),
            process = process,
            title = 'Canny Parameter Tuner',
        ).get_cfg()

    cfg.save()
    print(cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tune canny edge parameters')
    parser.add_argument('img_path', help='Image file')
    parser.add_argument('-c', '--config', help='Config file', default=CFG_FILE)
    args = parser.parse_args()

    main(args.img_path, args.config)