#!/bin/env python3

import cv2 as cv
import numpy as np
import utils
import argparse
import canny_tuner as canny


HOUGH_CFG_FILE = '.hough_parameters.cfg'


class _line:
    def __init__(self, slope, x1, y1, x2, y2):
        self.slope = slope
        self.x1 = x1 
        self.y1 = y1 
        self.x2 = x2 
        self.y2 = y2 


def _extend_line(line):
    line.x1 -= 2000
    line.y1 = int(line.y2 - line.slope * (line.x2 - line.x1))

    line.x2 += 2000
    line.y2 = int(line.slope * (line.x2 - line.x1) + line.y1)

    return line


def _get_avg_line(lines):
    lines.sort(key = lambda l : l.slope)
    while abs(lines[len(lines) // 2].slope - lines[-1].slope) > 0.5:
        lines.pop()
    while abs(lines[len(lines) // 2].slope - lines[0].slope) > 0.5:
        lines.pop(0)

    avg_line = _line(0, 0, 0, 0, 0) 
    for l in lines:
        avg_line.x1 += l.x1
        avg_line.y1 += l.y1
        avg_line.x2 += l.x2
        avg_line.y2 += l.y2

    avg_line.x1 = avg_line.x1 // len(lines)
    avg_line.y1 = avg_line.y1 // len(lines)
    avg_line.x2 = avg_line.x2 // len(lines)
    avg_line.y2 = avg_line.y2 // len(lines)
    avg_line.slope = (avg_line.y2 - avg_line.y1) / (avg_line.x2 - avg_line.x1)

    return _extend_line(avg_line)


def _get_lines(edges, threshold, minLineLength, maxLineGap):
    lines = cv.HoughLinesP(
        image = edges,
        rho = 1,
        theta = np.pi / 180,
        threshold = threshold,
        lines = np.array([]),
        minLineLength = minLineLength,
        maxLineGap = maxLineGap,
    )

    if lines is None:
        return (None, None)

    leftRawLines = []
    rightRawLines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1)
        if slope > 0:
            rightRawLines.append(_line(slope, x1, y1, x2, y2))
        else:
            leftRawLines.append(_line(slope, x1, y1, x2, y2))

    if len(leftRawLines) == 0 or len(rightRawLines) == 0:
        return (None, None)
    else:
        return (_get_avg_line(leftRawLines), _get_avg_line(rightRawLines))

def _getMask(image, points):
    mask = np.zeros_like(image)
    vertices = np.array(
        [points], 
        dtype=np.int32,
    )

    cv.fillPoly(mask, vertices, (255, 255, 255))
    return mask

def _get_triangle_vertices(image, base, height):
    x0 = int((image.shape[1] - base) / 2)
    x1 = x0 + base
    x2 = int(image.shape[1] / 2)
    y0 = y1 = image.shape[0]
    y2 = image.shape[0] - height 
    return [
            (x0, y0),
            (x1, y1),
            (x2, y2),
    ]


def Process(
    image,
    edges,
    topCut,
    bottomCut,
    base,
    height,
    pointsThs,
    minLineLength,
    maxLineGap,
):
    _bottomCut = edges.shape[0] - bottomCut
    triangleMask = _getMask(
        image = edges,
        points = _get_triangle_vertices(
            image = edges, 
            base = base, 
            height = height
        ),
    )

    cutMask = _getMask(
        image = edges,
        points = [
            (0, topCut),
            (0, _bottomCut),
            (edges.shape[1], _bottomCut),
            (edges.shape[1], topCut),
        ],
    )

    colorTriangleMask = np.dstack((triangleMask, triangleMask, triangleMask))
    colorCutMask = np.dstack((cutMask, cutMask, cutMask))

    maskedEdges = edges & triangleMask & cutMask
    maskedImage = image & colorTriangleMask & colorCutMask

    lLine, rLine = _get_lines(
        edges = maskedEdges,
        threshold = pointsThs,
        minLineLength = minLineLength,
        maxLineGap = maxLineGap,
    )

    if lLine is None:
        return (
            maskedImage,
            np.dstack((maskedEdges, maskedEdges, maskedEdges)),
            image
        )

    line_image = np.copy(image) * 0

    cv.line(line_image, (lLine.x1, lLine.y1), (lLine.x2, lLine.y2), (0, 0, 255), 15)
    cv.line(line_image, (rLine.x1, rLine.y1), (rLine.x2, rLine.y2), (0, 0, 255), 15)
    line_image = line_image & colorTriangleMask & colorCutMask

    colorEdges = np.dstack((maskedEdges, maskedEdges, maskedEdges))
    lineEdges = cv.addWeighted(colorEdges, 1, line_image, 0.8, 0)
    lineImage = cv.addWeighted(image, 1, line_image, 0.8, 0)
    return (maskedImage, lineEdges, lineImage)


class HoughCfg:
    def __init__(
        self,
        topCut,
        bottomCut,
        base,
        height,
        filterSize,
        threshold1,
        threshold2,
        pointsThs,
        minLineLength,
        maxLineGap,
    ):
        self.topCut = topCut 
        self.bottomCut = bottomCut 
        self.base = base
        self.height = height
        self.filterSize = filterSize
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.pointsThs = pointsThs
        self.minLineLength = minLineLength
        self.maxLineGap = maxLineGap


HOUGH_DEFAULT_VALUES = HoughCfg(0, 0, 100, 100, 13, 28, 115, 10, 10, 10) 


class HoughLinesTuner:
    def __init__(
        self,
        image,
        topCut,
        bottomCut,
        base,
        height,
        filterSize,
        threshold1,
        threshold2,
        pointsThs,
        minLineLength,
        maxLineGap,
    ):
        self._TITLE = 'Hough-Lines Parameter Tuner'
        self._image = image 
        self._topCut = topCut 
        self._bottomCut = bottomCut 
        self._base = base 
        self._height = height
        self._filterSize = filterSize
        self._threshold1 = threshold1
        self._threshold2 = threshold2
        self._pointsThs = pointsThs
        self._minLineLength = minLineLength
        self._maxLineGap = maxLineGap

        def onchangeThreshold1(pos):
            self._threshold1 = pos
            self._render()

        def onchangeThreshold2(pos):
            self._threshold2 = pos
            self._render()

        def onchangeFilterSize(pos):
            self._filterSize = pos
            self._render()

        def onchangeBase(pos):
            self._base = pos
            self._render()

        def onchangeHeight(pos):
            self._height = pos
            self._render()

        def onchangePointsThs(pos):
            self._pointsThs = pos
            self._render()

        def onchangeMinLineLength(pos):
            self._minLineLength = pos
            self._render()

        def onchangeMaxLineGap(pos):
            self._maxLineGap = pos
            self._render()

        def onchangeTopCut(pos):
            self._topCut = pos
            self._render()

        def onchangeBottonCut(pos):
            self._bottomCut = pos
            self._render()

        cv.namedWindow(self._TITLE, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)

        cv.createTrackbar('top cut', self._TITLE, self._topCut, self._image.shape[0], onchangeTopCut)
        cv.createTrackbar('botton cut', self._TITLE, self._bottomCut, self._image.shape[0], onchangeBottonCut)
        cv.createTrackbar('base', self._TITLE, self._base, self._image.shape[1], onchangeBase)
        cv.createTrackbar('height', self._TITLE, self._height, self._image.shape[0], onchangeHeight)
        cv.createTrackbar('points ths', self._TITLE, self._pointsThs, 300, onchangePointsThs)
        cv.createTrackbar('minLineLength', self._TITLE, self._minLineLength, 300, onchangeMinLineLength)
        cv.createTrackbar('maxLineGap', self._TITLE, self._maxLineGap, 300, onchangeMaxLineGap)
        cv.createTrackbar('filter size', self._TITLE, self._filterSize, 20, onchangeFilterSize)
        cv.createTrackbar('threshold1', self._TITLE, self._threshold1, 255, onchangeThreshold1)
        cv.createTrackbar('threshold2', self._TITLE, self._threshold2, 255, onchangeThreshold2)

        self._render()
        key = cv.waitKey()
        while(key != ord('\r')):
            key = cv.waitKey()

        cv.destroyWindow(self._TITLE)

    def get_results(self):
        return (
            self._topCut,
            self._bottomCut,
            self._base,
            self._height,
            self._filterSize,
            self._threshold1,
            self._threshold2,
            self._pointsThs,
            self._minLineLength,
            self._maxLineGap,
        )

    def _render(self):
        edges = canny.Process(
            image = cv.cvtColor(self._image, cv.COLOR_RGB2GRAY),
            filterSize = self._filterSize,
            threshold1 = self._threshold1,
            threshold2 = self._threshold2,
        )
        masked_image, line_edges, _ = Process(
            image = self._image,
            edges = edges,
            topCut = self._topCut,
            bottomCut = self._bottomCut,
            base = self._base,
            height = self._height,
            pointsThs = self._pointsThs,
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
            topCut = cfg.topCut,
            bottomCut = cfg.bottomCut,
            base = cfg.base,
            height = cfg.height,
            filterSize = cfg.filterSize,
            threshold1 = cfg.threshold1,
            threshold2 = cfg.threshold2,
            pointsThs = cfg.pointsThs,
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