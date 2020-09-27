#!/bin/env python3

import cv2 as cv
import numpy as np
import utils
import argparse


class CannyTuner:
    def __init__(self, image_path, filter_size=1, threshold1=0, threshold2=0):
        self._TITLE = 'Canny Parameter Tuner'
        self._image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
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

        cv.createTrackbar('threshold1', self._TITLE, self._threshold1, 255, onchangeThreshold1)
        cv.createTrackbar('threshold2', self._TITLE, self._threshold2, 255, onchangeThreshold2)
        cv.createTrackbar('filter_size', self._TITLE, self._filter_size, 20, onchangeFilterSize)

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
        utils.Plot(
            shape = (1, 2),
            imgs = [
                self._image,
                edge_img,
            ],
            title = self._TITLE,
        )


def main():
    parser = argparse.ArgumentParser(description='Visualizes the line for hough transform.')
    parser.add_argument('filename')

    print(
        CannyTuner(
            image_path = parser.parse_args().filename,
            filter_size = 13,
            threshold1 = 28,
            threshold2 = 115,
        ).get_results(),
    )


if __name__ == '__main__':
    main()