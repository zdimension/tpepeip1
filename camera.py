# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 09:55:18 2018

@author: nigett
"""

import cv2

from logger import *


class Camera:
    def __init__(self, path=0):
        self.path = path
        self.camera = cv2.VideoCapture(path)

        try:
            info(f"Trying to connect to camera {self.path}")
            status = self.camera.read()
            info("Got after read")
            self.camera_shape = status[1].shape
            self.connected = True
        except:
            self.camera_shape = None
            self.connected = False
            error(f"Couldn't connect to camera {self.path}")

    def dispose(self):
        self.camera.release()

    def frame(self):
        if not self.connected:
            error(f"Trying to read frame from invalid camera {self.path}")
            raise None

        try:
            return self.camera.read()[1]
        except:
            error("An error occured while reading the frame")
            exit()
