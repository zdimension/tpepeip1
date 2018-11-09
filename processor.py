# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 09:21:21 2018

@author: nigett
"""

import numpy as np
import cv2
import time
from logger import *
from helper import *

# limit of maximum shift between two faces
SHIFT_THRESHOLD = 5

class ImageProcessor():
    def __init__(self, cascade):
        self.fps = 0
        self.buf_size =512
        self.buffer = []
        
        self.lock_face = False
        
        self.input = np.zeros((10, 10))
        self.output = np.zeros((10, 10))
        
        self.classifier = cv2.CascadeClassifier(cascade)
        self.trained = False
        
        self.times = []
        self.start_time = time.time()
        
        self.face = [0, 0, 1, 1]
        
        self.last_centre = np.array([0, 0])
        
    def train_toggle(self):
        """sets if trained"""
        self.trained = not self.trained
        
        return self.trained
    
    def rect(self, x, y, w, h, col=WHITE):
        """wrapper for opencv"""
        cv2.rectangle(self.output, (x, y), (x + w, y + h), col, 5)
        
    def lock_toggle(self):
        """lock on face"""
        self.lock_face = not self.lock_face
        
        return self.lock_face
    
    def calc_shift(self, face):
        """calculates shift between current face and last detected to see if it's the same one"""
        x, y ,w, h = face
        
        centre = np.array([x + w / 2, y + h / 2])
        decal = np.linalg.norm(centre - self.last_centre)
        self.last_centre = centre
        
        return decal

    def get_slice(self, rx, ry, rw, rh):
        """get part of rect (values between 0 and 1)"""
        x, y, w, h = self.face

        fix_x = w * rx
        fix_y = w * ry
        fix_w = w * rw
        fix_h = h * rh

        return list(map(int, [
            x + fix_x - fix_w / 2,
            y + fix_y - fix_h / 2,
            fix_w,
            fix_h
        ]))
    
    def execute(self, text):
        """process the image"""
        self.output = self.input
        
        self.times.append(time.time() - self.start_time)
        
        # n&b pour la détection de visage
        self.input_g = cv2.equalizeHist(cv2.cvtColor(self.input, cv2.COLOR_BGR2GRAY))
        
        if not self.lock_face:
            # reherche
            self.times = []
            self.buffer = []
            
            self.trained = False
            
            detect = list(self.classifier.detectMultiScale(self.input_g, minNeighbors=4, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE))
            
            if detect:
                # prendre celle avec la plus grande taille (w * h)
                biggest = sorted(detect, reverse=True, key=lambda f: f[-1] * f[-2])[0]
 
                if self.calc_shift(biggest) > SHIFT_THRESHOLD:
                    self.face = biggest

        forehead = self.get_slice(0.5, 0.2, 0.25, 0.15)

        self.rect(*self.face, BLUE)
        self.rect(*forehead, GREEN)

        text("Face", *self.face[:2], BLUE)
        text("Forehead", *forehead[:2], GREEN)



        