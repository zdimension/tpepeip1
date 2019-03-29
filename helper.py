# -*- coding: utf-8 -*-
import numpy as np

from math import sqrt

import numba

# colors (bgr)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# bounds for a "normal" bpm
BPM_LOW = 50
BPM_HIGH = 180
BPM_NOISE_HIGH = 120


def get_frame(w, h):
    return np.zeros((h, w, 3), np.uint8)

@numba.jit(nopython=True)
def progressive_mean(arr):
    """take a "progressive" mean, i.e. values are more accounted for the more recent they are, on a sqrt scale"""
    #coeffs = [sqrt((x + len(arr) / 100) / (len(arr) + len(arr) / 100)) for x in range(1, len(arr) + 1)]
    #return np.sum([x * c for x, c in zip(arr, coeffs)]) / np.sum(coeffs)
    #return np.average(arr, weights=np.sqrt(arr))
    w = np.sqrt(np.arange(1, len(arr) + 1))
    return np.dot(w, arr) / np.sum(w)
