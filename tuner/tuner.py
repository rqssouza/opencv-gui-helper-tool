#!/bin/env python3

import numpy as np
import cv2 as cv
import json
import argparse


class Tuner_Cfg:
    ''' Represents the configuration object of Tuner class
    :param cfg_file: File containing the configuration
    :type cfg_file: str
    :param default_attrs: Default configuration if cfg_file doesn't exit
    :type cfg_file: list
    '''
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
        '''Get value by attr name

        :param attr: Name of the attribute
        :type amount: str

        :returns: The value of the attribute
        :rtype: int
        '''
        return [i for i in self._cfg_attrs if i[0] == attr][0][1]


    def get_attrs(self):
        '''Get the attributes list

        :returns: The list of attributes
        :rtype: list
        '''
        return self._cfg_attrs


    def get_dict(self):
        '''Get a dict with keyword and value of attrs

        :returns: The dict with all attributes
        :rtype: dict
        '''
        return {key:value for key, value, _ in self._cfg_attrs}


    def set_value(self, attr, val):
        '''Set value by attr name

        :param attr: Name of the attribute
        :type amount: str
        :param val: Value of the attribute
        :type amount: int 
        '''
        [i for i in self._cfg_attrs if i[0] == attr][0][1] = val


    def save(self):
        '''Saves the configuration to the cfg_file
        '''
        with open(self._cfg_file, 'w') as fp:
            json.dump(self._cfg_attrs, fp)


class Tuner_Args:
    def __init__(self, args):
        for key, value in args.items():
            setattr(self, key, value)


class Tuner:
    def __init__(self, image, cfg, process, title = ''):
        ''' Parameter Tuner class
        :param image: Input image
        :type image: numpy.ndarray 
        :param cfg: Parameters configuration
        :type cfg: Tuner_Cfg
        :param process: The algorithm to be called whenever any parameter change
        :type process: function
        :param title: The title that appears on the window
        :type process: str
        '''
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
            args = Tuner_Args(self._cfg.get_dict()),
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
        '''Get the resulting configuration

        :returns: The configuration after the tuning process
        :rtype: Tuner_Cfg
        '''
        return self._cfg


def Tuner_App(process, config, title, description = ''):
    parser = argparse.ArgumentParser(description = description)
    parser.add_argument('img_path', help = 'Image file')
    parser.add_argument('-c', '--config_file',
        help = 'Config file',
        default = '.' + ''.join(title.split()) + '.cfg'
    )
    args = parser.parse_args()
    image = cv.imread(args.img_path)

    cfg = Tuner(
            image = image,
            cfg = Tuner_Cfg(args.config_file, config),
            process = process,
            title = title,
    ).get_cfg()

    cfg.save()
    print(cfg)