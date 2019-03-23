# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 09:55:18 2018

@author: nigett
"""
import time
from queue import Queue
from threading import Thread

import cv2

from logger import *


class Camera:
    def __init__(self, path=0):
        self.path = path
        self.camera = cv2.VideoCapture(path)
        self.queue = Queue()

        try:
            info(f"Trying to connect to camera {self.path}")
            self.frame = self.camera.read()[1]
            info("Got after read")
            self.camera_shape = self.frame.shape
            self.connected = True
        except:
            self.camera_shape = None
            self.connected = False
            error(f"Couldn't connect to camera {self.path}")

        self.thread = Thread(target=self.read_frame)
        self.thread.start()

    def read_frame(self):
        while self.connected:
            try:
                #self.queue.put(self.camera.read()[1])
                self.frame = self.camera.read()[1]
            except:
                error("An error occured while reading the frame")
                pass

    def dispose(self):
        self.camera.release()

    def get_frame(self):
        if not self.connected:
            error(f"Trying to read frame from invalid camera {self.path}")
            raise None

        try:
            #return self.camera.read()[1]
            #return self.queue.get()
            return self.frame
        except:
            error("An error occured while reading the frame")
            exit()
