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
MIN_SAMPLE_COUNT = 15
FPS_SMOOTHING = 0.8

class ImageProcessor():
    def __init__(self, cascade):
        self.last_fps = 0
        self.fps = 0
        self.buf_size =256
        self.avg_colors_buf = []
        
        self.lock_face = False
        
        self.input = np.zeros((10, 10))
        self.output = np.zeros((10, 10))
        
        self.classifier = cv2.CascadeClassifier(cascade)
        self.trained = False
        
        self.times_buf = []
        self.start_time = time.time()
        
        self.face = [0, 0, 1, 1]
        
        self.last_centre = np.array([0, 0])

        self.frequencies = []
        self.fourier = []
        self.bpm = 0
        self.gap = 0
        self.buf_state = 0

        
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

    def calc_mean_color(self, rect):
        """calculate mean color in sub rectangle"""
        x,y ,w, h = rect
        slice = self.input[y:y + h, x:x + w, :]
        weights = [1, 2.5, 1]
        return sum(weight * np.mean(slice[:, :, c]) for c, weight in enumerate(weights)) / sum(weights)

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

    def resize_bufs(self):
        self.avg_colors_buf = self.avg_colors_buf[-self.buf_size:]
        self.times_buf = self.times_buf[-self.buf_size:]
    
    def execute(self, text):
        """process the image"""
        self.output = self.input

        self.buf_state += 1
        self.buf_state %= self.buf_size
        
        self.times_buf.append(time.time() - self.start_time)
        
        # n&b pour la détection de visage
        self.input_g = cv2.equalizeHist(cv2.cvtColor(self.input, cv2.COLOR_BGR2GRAY))
        
        if not self.lock_face:
            # reherche
            self.trained = False
            
            detect = list(self.classifier.detectMultiScale(self.input_g, minNeighbors=4, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE))
            
            if detect:
                # prendre celle avec la plus grande taille (w * h)
                biggest = sorted(detect, reverse=True, key=lambda f: f[-1] * f[-2])[0]
 
                if self.calc_shift(biggest) > SHIFT_THRESHOLD:
                    self.face = biggest

        if set(self.face) == {0, 1}:
            self.avg_colors_buf.append(0)
            self.resize_bufs()
            return

        forehead = self.get_slice(0.5, 0.15, 0.35, 0.18)

        self.rect(*self.face, BLUE)
        self.rect(*forehead, GREEN)

        text("Face", *self.face[:2], BLUE)
        text("Forehead", *forehead[:2], GREEN)

        fh_average = self.calc_mean_color(forehead)
        self.avg_colors_buf.append(fh_average)

        self.resize_bufs()

        num_samples = len(self.times_buf)

        if num_samples > MIN_SAMPLE_COUNT:
            time_start, time_end = self.times_buf[0], self.times_buf[-1]
            num_samples = len(self.times_buf)
            self.last_fps = self.fps
            dt = time_end - time_start
            if dt:
                fps = num_samples / dt
                self.fps = fps * FPS_SMOOTHING + self.last_fps * (1 - FPS_SMOOTHING)
            avg_colors = np.array(self.avg_colors_buf)
            linear = np.linspace(time_start, time_end, num_samples)
            interp = np.hamming(num_samples) - np.interp(linear, self.times_buf, avg_colors)
            deviation = interp - np.mean(interp)
            fourier = np.fft.rfft(deviation)
            self.fourier = np.abs(fourier)
            self.frequencies = self.fps / num_samples * np.arange(num_samples / 2 + 1)
            fix_freqs = self.frequencies * 60.
            pos = np.where((fix_freqs > BPM_LOW) & (fix_freqs < BPM_HIGH))
            try:
                filtered = self.fourier[pos]
                self.frequencies = fix_freqs[pos]
                self.fourier = filtered
                self.bpmpos = np.argmax(filtered)
                self.bpm = self.frequencies[self.bpmpos]
            except:
                pass
            self.gap = (self.buf_size - num_samples) / self.fps