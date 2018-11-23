# -*- coding: utf-8 -*-
import cv2
import numpy as np

# colors (bgr)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# bounds for a "normal" bpm
BPM_LOW = 60
BPM_HIGH = 180
BPM_NOISE_HIGH = 120

def get_frame(w, h):
    return np.zeros((h, w, 3), np.uint8)