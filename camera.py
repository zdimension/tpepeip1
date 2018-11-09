# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 09:55:18 2018

@author: nigett
"""

import cv2
import numpy as np

class Camera():
    def __init__(self, path=0):
        self.path = path
        self.camera = cv2.VideoCapture(path)  
        
        try:
            status = self.camera.read()
            self.camera_shape = status[1].shape
            self.connected = True
        except:
            self.camera_shape = None
            self.connected = False
            print("[Error] Couldn't connect to camera %s" % str(self.path))
            
    def dispose(self):
        self.camera.release()
            
    def frame(self):
        if not self.connected:
            raise
            
        return self.camera.read()[1]
    
    