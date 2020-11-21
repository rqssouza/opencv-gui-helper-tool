#!/bin/env python3

import cv2 as cv
import numpy as np
import argparse
import tuner.tuner as tuner


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


def _get_lines(edges, threshold, min_line_length, max_line_gap):
    lines = cv.HoughLinesP(
        image = edges,
        rho = 1,
        theta = np.pi / 180,
        threshold = threshold,
        lines = np.array([]),
        minLineLength = min_line_length,
        maxLineGap = max_line_gap,
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


def _get_mask(image, points):
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


def road_lines_calc(
    image,
    edges,
    top_cut,
    bottom_cut,
    base,
    height,
    points_ths,
    min_line_length,
    max_line_gap,
):
    _bottom_cut = edges.shape[0] - bottom_cut
    triangle_mask = _get_mask(
        image = edges,
        points = _get_triangle_vertices(
            image = edges, 
            base = base, 
            height = height
        ),
    )

    cut_mask = _get_mask(
        image = edges,
        points = [
            (0, top_cut),
            (0, _bottom_cut),
            (edges.shape[1], _bottom_cut),
            (edges.shape[1], top_cut),
        ],
    )

    color_triangle_mask = np.dstack((triangle_mask, triangle_mask, triangle_mask))
    color_cut_mask = np.dstack((cut_mask, cut_mask, cut_mask))

    masked_edges = edges & triangle_mask & cut_mask
    masked_image = image & color_triangle_mask & color_cut_mask

    l_line, r_line = _get_lines(
        edges = masked_edges,
        threshold = points_ths,
        min_line_length = min_line_length,
        max_line_gap = max_line_gap,
    )

    if l_line is None:
        return (
            masked_image,
            np.dstack((masked_edges, masked_edges, masked_edges)),
            image
        )

    line_image = np.copy(image) * 0

    cv.line(line_image, (l_line.x1, l_line.y1), (l_line.x2, l_line.y2), (0, 0, 255), 15)
    cv.line(line_image, (r_line.x1, r_line.y1), (r_line.x2, r_line.y2), (0, 0, 255), 15)
    line_image = line_image & color_triangle_mask & color_cut_mask

    color_edges = np.dstack((masked_edges, masked_edges, masked_edges))
    line_edges = cv.addWeighted(color_edges, 1, line_image, 0.8, 0)
    line_image = cv.addWeighted(image, 1, line_image, 0.8, 0)
    return (masked_image, line_edges, line_image)


def canny_calc(image, kernel_size, ths1, ths2):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    smoothed_img = cv.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0, sigmaY=0)

    return cv.Canny(smoothed_img, ths1, ths2)


def process(image, args):
    adj_k = lambda ksize : ksize + (ksize + 1) % 2
    shape = (2, 2)
    imgs = [image]

    edges = canny_calc(
        image = image,
        kernel_size = adj_k(args.kernel_size),
        ths1 = args.threshold1,
        ths2 = args.threshold2,
    )

    imgs += list(road_lines_calc(
        image = image,
        edges = edges,
        top_cut = args.top_cut,
        bottom_cut = args.bottom_cut,
        base = args.base,
        height = args.height,
        points_ths = args.points_ths,
        min_line_length = args.min_line_length,
        max_line_gap = args.max_line_gap,
    ))

    return (shape, imgs)


CFG = [
    ['top_cut', 0, 2000],
    ['bottom_cut', 0, 2000],
    ['base', 100, 2000],
    ['height', 100, 2000],
    ['kernel_size', 13, 300],
    ['threshold1', 28, 300],
    ['threshold2', 115, 300],
    ['points_ths', 10, 20],
    ['min_line_length', 10, 255],
    ['max_line_gap', 10, 255],
]


if __name__ == '__main__':
    tuner.Tuner_App(
        process, 
        CFG,
        'Road Lines',
        'Tune parameters to find road lines',
    )