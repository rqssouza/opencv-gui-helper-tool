#!/bin/env python3

import cv2 as cv
import numpy as np
import utils
import argparse

HOUGH_CFG_FILE = '.hough_parameters.cfg'

class HoughCfg:
    def __init__(self, base, height):
        self.base = base
        self.height = height

class HoughLinesTuner:
    def __init__(self, image, base=100, height=100):
        self._TITLE = 'Hough-Lines Parameter Tuner'
        self._image = image 
        self._base = base 
        self._height = height

        def onchangeBase(pos):
            self._base = pos
            self._render()

        def onchangeHeight(pos):
            self._height = pos
            self._render()

        cv.namedWindow(self._TITLE, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)

        cv.createTrackbar('base', self._TITLE, self._base, self._image.shape[1], onchangeBase)
        cv.createTrackbar('height', self._TITLE, self._height, self._image.shape[0], onchangeHeight)

        self._render()
        key = cv.waitKey()
        while(key != ord('\r')):
            key = cv.waitKey()

        cv.destroyWindow(self._TITLE)

    def get_results(self):
        return (self._base, self._height)

    def _render(self):
        mask = np.zeros_like(self._image)
        x0 = int((self._image.shape[1] - self._base) / 2)
        x1 = x0 + self._base 
        x2 = self._image.shape[1] / 2 
        y0 = y1 = self._image.shape[0]
        y2 = self._image.shape[0] - self._height 
        vertices = np.array(
            [[
                (x0, y0),
                (x1, y1),
                (x2, y2),
            ]], 
            dtype=np.int32,
        )

        cv.fillPoly(mask, vertices, (255, 255, 255))
        masked_image = self._image & mask

        utils.plot(
            shape = (1, 2),
            imgs = [
                self._image,
                masked_image,
            ],
            title = self._TITLE,
        )


def main(image_path):
    image = cv.imread(image_path)
    cfg = utils.load_cfg(HOUGH_CFG_FILE)

    if cfg:
        results = HoughCfg(
            *HoughLinesTuner(
                image = image, 
                base = cfg.base,
                height = cfg.height,
            ).get_results()
        )
    else:
        results = HoughCfg(
            *HoughLinesTuner(
                image = image, 
            ).get_results()
        )

    utils.save_cfg(HOUGH_CFG_FILE, results)

    print(utils.convert_to_dict(results))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualizes the line for hough transform.')
    parser.add_argument('img_path')

    main(parser.parse_args().img_path)