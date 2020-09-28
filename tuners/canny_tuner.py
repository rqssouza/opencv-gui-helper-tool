#!/bin/env python3

import cv2 as cv
import numpy as np
import utils
import argparse

CANNY_CFG_FILE = '.canny_parameters.cfg'

class CannyCfg:
    def __init__(self, filter_size, threshold1, threshold2):
        self.filter_size = filter_size
        self.threshold1 = threshold1
        self.threshold2 = threshold2


class CannyTuner:
    def __init__(self, image, filter_size=13, threshold1=28, threshold2=115):
        self._TITLE = 'Canny Parameter Tuner'
        self._image = image 
        self._filter_size = filter_size
        self._threshold1 = threshold1
        self._threshold2 = threshold2

        def onchangeThreshold1(pos):
            self._threshold1 = pos
            self._render()

        def onchangeThreshold2(pos):
            self._threshold2 = pos
            self._render()

        def onchangeFilterSize(pos):
            self._filter_size = pos + (pos + 1) % 2
            self._render()

        cv.namedWindow(self._TITLE, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)

        cv.createTrackbar('filter_size', self._TITLE, self._filter_size, 20, onchangeFilterSize)
        cv.createTrackbar('threshold1', self._TITLE, self._threshold1, 255, onchangeThreshold1)
        cv.createTrackbar('threshold2', self._TITLE, self._threshold2, 255, onchangeThreshold2)

        self._render()
        key = cv.waitKey()
        while(key != ord('\r')):
            key = cv.waitKey()

        cv.destroyWindow(self._TITLE)

    def get_results(self):
        return (self._filter_size, self._threshold1, self._threshold2)

    def _render(self):
        smoothed_img = cv.GaussianBlur(self._image, (self._filter_size, self._filter_size), sigmaX=0, sigmaY=0)
        edge_img = cv.Canny(smoothed_img, self._threshold1, self._threshold2)
        utils.plot(
            shape = (1, 2),
            imgs = [
                self._image,
                edge_img,
            ],
            title = self._TITLE,
        )


def main(image_path):
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    cfg = utils.load_cfg(CANNY_CFG_FILE)

    if cfg:
        results = CannyCfg(
            *CannyTuner(
                image = image, 
                filter_size = cfg.filter_size,
                threshold1 = cfg.threshold1,
                threshold2 = cfg.threshold2,
            ).get_results()
        )
    else:
        results = CannyCfg(
            *CannyTuner(
                image = image, 
            ).get_results()
        )

    utils.save_cfg(CANNY_CFG_FILE, results)

    print(utils.convert_to_dict(results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualizes the line for hough transform.')
    parser.add_argument('img_path')

    main(parser.parse_args().img_path)