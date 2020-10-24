#!/bin/env python3

import numpy as np
import cv2 as cv
import json


class TunerCfg:
    def __init__(self, cfg_file, default_attrs = []):
        self._cfg_file = cfg_file
        self._cfg_attrs = self._load(default_attrs)


    def __str__(self):
        ret = ''
        for attr, val, maxVal in self._cfg_attrs:
            ret += f'{attr} = {val}\n'
        return ret


    def _load(self, default_attrs):
        try:
            with open(self._cfg_file, 'r') as fp:
                return json.load(fp)
        except FileNotFoundError:
            return default_attrs


    def get_value(self, attr):
        return [i for i in self._cfg_attrs if i[0] == attr][0][1]


    def get_attrs(self):
        return self._cfg_attrs


    def set_value(self, attr, val):
        [i for i in self._cfg_attrs if i[0] == attr][0][1] = val


    def save(self):
        with open(self._cfg_file, 'w') as fp:
            json.dump(self._cfg_attrs, fp)


class Tuner:
    def __init__(self, image, cfg, process, title = ''):
        self._TITLE = title 
        self._image = image 
        self._cfg = cfg
        self._process = process

        def on_change(attr, pos):
            self._cfg.set_value(attr, pos)
            self._render()
        
        def get_lambda(attr):
            return lambda pos : on_change(attr, pos)

        cv.namedWindow(self._TITLE, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)

        for attr, value, max_val in self._cfg.get_attrs():
            cv.createTrackbar(
                attr,
                self._TITLE,
                value,
                max_val,
                get_lambda(attr),
            )

        self._render()
        key = cv.waitKey()
        while(key != ord('\r')):
            key = cv.waitKey()

        cv.destroyWindow(self._TITLE)


    def _render(self):
        shape, imgs = self._process(
            image = self._image,
            cfg = self._cfg,
        )
        self._plot(
            shape = shape,
            imgs = imgs, 
        )


    def _plot(self, shape, imgs):
        lines = []
        for i in range(shape[0]):
            lines.append(np.hstack(imgs[i * shape[0]:i * shape[0] + shape[1]]))

        img = np.vstack(lines)
        cv.imshow(self._TITLE, img)
    

    def get_cfg(self):
        return self._cfg