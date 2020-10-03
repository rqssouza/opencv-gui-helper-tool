#!/bin/env python3

import cv2 as cv
import numpy as np
import utils
import argparse
import canny_tuner as canny


HOUGH_CFG_FILE = '.hough_parameters.cfg'


def Process(
    image,
    edges,
    base,
    height,
    points_ths,
    minLineLength,
    maxLineGap,
):
    mask = np.zeros_like(edges)
    mask3d = np.zeros_like(image)
    x0 = int((edges.shape[1] - base) / 2)
    x1 = x0 + base
    x2 = int(edges.shape[1] / 2)
    y0 = y1 = edges.shape[0]
    y2 = edges.shape[0] - height 
    vertices = np.array(
        [[
            (x0, y0),
            (x1, y1),
            (x2, y2),
        ]], 
        dtype=np.int32,
    )

    cv.fillPoly(mask, vertices, (255, 255, 255))
    cv.fillPoly(mask3d, vertices, (255, 255, 255))
    masked_edges = edges & mask
    masked_image = image & mask3d

    lines = cv.HoughLinesP(
        image = masked_edges,
        rho = 1,
        theta = np.pi / 180,
        threshold = points_ths,
        lines = np.array([]),
        minLineLength = minLineLength,
        maxLineGap = maxLineGap,
    )
    lines = lines if type(lines) == np.ndarray else []

    line_image = np.copy(image) * 0

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    color_edges = np.dstack((masked_edges, masked_edges, masked_edges))
    line_edges = cv.addWeighted(color_edges, 0.8, line_image, 1, 0)
    return (masked_image, line_edges)


class HoughCfg:
    def __init__(
        self,
        base,
        height,
        filter_size,
        threshold1,
        threshold2,
        points_ths,
        minLineLength,
        maxLineGap,
    ):
        self.base = base
        self.height = height
        self.filter_size = filter_size
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.points_ths = points_ths
        self.minLineLength = minLineLength
        self.maxLineGap = maxLineGap


HOUGH_DEFAULT_VALUES = HoughCfg(100, 100, 13, 28, 115, 10, 10, 10) 


class HoughLinesTuner:
    def __init__(
        self,
        image,
        base,
        height,
        filter_size,
        threshold1,
        threshold2,
        points_ths,
        minLineLength,
        maxLineGap,
    ):
        self._TITLE = 'Hough-Lines Parameter Tuner'
        self._image = image 
        self._base = base 
        self._height = height
        self._filter_size = filter_size
        self._threshold1 = threshold1
        self._threshold2 = threshold2
        self._points_ths = points_ths
        self._minLineLength = minLineLength
        self._maxLineGap = maxLineGap

        def onchangeThreshold1(pos):
            self._threshold1 = pos
            self._render()

        def onchangeThreshold2(pos):
            self._threshold2 = pos
            self._render()

        def onchangeFilterSize(pos):
            self._filter_size = pos + (pos + 1) % 2
            self._render()

        def onchangeBase(pos):
            self._base = pos
            self._render()

        def onchangeHeight(pos):
            self._height = pos
            self._render()

        def onchangePointsThs(pos):
            self._points_ths = pos
            self._render()

        def onchangeMinLineLength(pos):
            self._minLineLength = pos
            self._render()

        def onchangeMaxLineGap(pos):
            self._maxLineGap = pos
            self._render()

        cv.namedWindow(self._TITLE, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)

        cv.createTrackbar('points_ths', self._TITLE, self._points_ths, 300, onchangePointsThs)
        cv.createTrackbar('minLineLength', self._TITLE, self._minLineLength, 300, onchangeMinLineLength)
        cv.createTrackbar('maxLineGap', self._TITLE, self._maxLineGap, 300, onchangeMaxLineGap)
        cv.createTrackbar('filter_size', self._TITLE, self._filter_size, 20, onchangeFilterSize)
        cv.createTrackbar('threshold1', self._TITLE, self._threshold1, 255, onchangeThreshold1)
        cv.createTrackbar('threshold2', self._TITLE, self._threshold2, 255, onchangeThreshold2)
        cv.createTrackbar('base', self._TITLE, self._base, self._image.shape[1], onchangeBase)
        cv.createTrackbar('height', self._TITLE, self._height, self._image.shape[0], onchangeHeight)

        self._render()
        key = cv.waitKey()
        while(key != ord('\r')):
            key = cv.waitKey()

        cv.destroyWindow(self._TITLE)

    def get_results(self):
        return (
            self._base,
            self._height,
            self._filter_size,
            self._threshold1,
            self._threshold2,
            self._points_ths,
            self._minLineLength,
            self._maxLineGap,
        )

    def _render(self):
        edges = canny.Process(
            image = cv.cvtColor(self._image, cv.COLOR_RGB2GRAY),
            filter_size = self._filter_size,
            threshold1 = self._threshold1,
            threshold2 = self._threshold2,
        )
        masked_image, line_edges = Process(
            image = self._image,
            edges = edges,
            base = self._base,
            height = self._height,
            points_ths = self._points_ths,
            minLineLength = self._minLineLength,
            maxLineGap = self._maxLineGap,
        )
        utils.plot(
            shape = (1, 2),
            imgs = [
                masked_image,
                line_edges,
            ],
            title = self._TITLE,
        )


def main(image_path, config):
    image = cv.imread(image_path)
    cfg = utils.load_cfg(config, HOUGH_DEFAULT_VALUES)

    cfg = HoughCfg( 
        *HoughLinesTuner(
            image = image, 
            base = cfg.base,
            height = cfg.height,
            filter_size = cfg.filter_size,
            threshold1 = cfg.threshold1,
            threshold2 = cfg.threshold2,
            points_ths = cfg.points_ths,
            minLineLength = cfg.minLineLength,
            maxLineGap = cfg.maxLineGap,
        ).get_results()
    )

    utils.save_cfg(config, cfg)

    print(utils.convert_to_dict(cfg))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualizes the line for hough transform.')
    parser.add_argument('img_path', help='Image file')
    parser.add_argument('-c', '--config', help='Config file', default=HOUGH_CFG_FILE)
    args = parser.parse_args()

    main(args.img_path, args.config)