# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 09:21:21 2018

@author: nigett
"""

import numpy as np
import cv2
import time

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
        
        self.font = cv2.FONT_HERSHEY_COMPLEX
        self.font_size = 1.25
        self.text_color = (255, 255, 255)
        
        self.times = []
        self.start_time = time.time()
        
        self.face = [0, 0, 1, 1]
        
        self.last_centre = np.array([0, 0])
        
    def train_toggle(self):
        """sets if trained"""
        self.trained = not self.trained
        
        return self.trained
    
    def rect(self, x, y, w, h, col=(255, 0, 0)):
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
    
    def text(self, text, x, y):
        """wrapper for opencv"""
        cv2.putText(self.output, text, (x, y), self.font, self.font_size, self.text_color)
    
    def execute(self):
        """process the image"""
        self.output = self.input
        
        self.times.append(time.time() - self.start_time)
        
        # n&b pour la détection de visage
        self.input_g = cv2.equalizeHist(cv2.cvtColor(self.input, cv2.COLOR_BGR2GRAY))
        
        self.text("l = lock face", 20, 30)
        self.text("c = next camera (if usb)", 20, 60)
        self.text("esc = exit", 20, 90)
        
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
                
            self.rect(*self.face)

        