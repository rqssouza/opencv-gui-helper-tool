import numpy as np
import cv2 as cv

def Plot(shape, imgs, title = ''):
    cv.namedWindow(title, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)

    lines = []
    for i in range(shape[0]):
        lines.append(np.hstack(imgs[i * shape[0]:i * shape[0] + shape[1]]))

    img = np.vstack(lines)
    cv.imshow(title, img)