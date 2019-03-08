# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 09:21:21 2018

@author: nigett
"""

import time
from math import sqrt

import cv2

from helper import *

# limit of maximum shift between two faces
SHIFT_THRESHOLD = 5

# min sample count before we can start bothering doing the computations and expect good results
MIN_SAMPLE_COUNT = 15

# fps smoothing factor
FPS_SMOOTHING = 0.8


class ImageProcessor():
    def __init__(self, cascade):
        self.last_fps = 0
        self.fps = 0
        self.buf_size = 256
        self.avg_colors_buf = []

        self.lock_face = False

        self.input = np.zeros((10, 10))
        self.output = np.zeros((10, 10))

        self.input_buf = []
        self.input_buf_size = 5
        self.denoise = False

        self.colorify = False

        self.classifier = cv2.CascadeClassifier(cascade)

        self.times_buf = []
        self.start_time = time.time()

        self.face = [0, 0, 1, 1]

        self.last_centre = np.array([0, 0])

        self.frequencies = []
        self.fourier = []
        self.bpm = 0
        self.gap = 0
        self.buf_state = 0
        self.deviation = [0]

        self.bpm_buf = [60]
        self.cor_bpm = 0
        self.bpmpos = 0

        self.highbpm = True
        self.current_high = BPM_NOISE_HIGH

    def clear_bufs(self):
        """clear the storage buffers"""
        self.avg_colors_buf = []
        self.times_buf = []
        self.bpm_buf = []

    def train_toggle(self):
        """sets if trained"""
        self.trained = not self.trained

        return self.trained

    def rect(self, x, y, w, h, col=WHITE):
        """wrapper for opencv"""
        cv2.rectangle(self.output, (x, y), (x + w, y + h), col, 2)

    def lock_toggle(self):
        """lock on face"""
        self.lock_face = not self.lock_face

        return self.lock_face

    def denoise_toggle(self):
        """denoise image"""
        self.denoise = not self.denoise

        return self.denoise

    def highbpm_toggle(self):
        """high bpm (noise) filtering"""
        self.highbpm = not self.highbpm
        self.current_high = BPM_NOISE_HIGH if self.highbpm else BPM_HIGH
        return self.highbpm

    def colorify_toggle(self):
        """enhanced color variation displaying"""
        self.colorify = not self.colorify

        return self.colorify

    def calc_shift(self, face):
        """calculates shift between current face and last detected to see if it's the same one"""
        x, y, w, h = face

        centre = np.array([x + w / 2, y + h / 2])
        decal = np.linalg.norm(centre - self.last_centre)
        self.last_centre = centre

        return decal

    def progressive_mean(self, arr):
        """take a "progressive" mean, i.e. values are more accounted for the more recent they are, on a sqrt scale"""
        coeffs = [sqrt((x + len(arr) / 100) / (len(arr) + len(arr) / 100)) for x in range(1, len(arr) + 1)]
        return sum(x * c for x, c in zip(arr, coeffs)) / sum(coeffs)

    def get_subpicture(self, rect, inp=None):
        """get "slice" of picture"""
        x, y, w, h = rect

        return (inp if inp else self.input)[y:y + h, x:x + w, :]

    def calc_mean_color(self, rect):
        """calculate mean color in sub rectangle"""

        slice = self.get_subpicture(rect)
        self.input_buf.append(slice)
        self.resize_bufs()

        if self.denoise and len(self.input_buf) == self.input_buf_size:
            slice = cv2.fastNlMeansDenoisingColoredMulti(self.input_buf,
                                                         self.input_buf_size // 2,
                                                         self.input_buf_size,
                                                         None,
                                                         3,
                                                         3,
                                                         7,
                                                         35)

        # we like the green channel more
        #weights = [1.15, 2.9, 0.8]
        weights = [0, 0.85, 0.4]

        return sum(weight * np.mean(slice[:, :, c]) for c, weight in enumerate(weights)) / sum(weights)

    def get_slice(self, rx, ry, rw, rh):
        """get part of rect (values between 0 and 1)
        rx, ry = center of rectangle
        rw, rh = width"""
        x, y, w, h = self.face

        fix_x = w * rx
        fix_y = h * ry
        fix_w = w * rw
        fix_h = h * rh

        return list(map(int, [
            x + fix_x - fix_w / 2,
            y + fix_y - fix_h / 2,
            fix_w,
            fix_h
        ]))

    def draw_at(self, img, rect):
        """draw the image at the specified position"""
        self.output[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = img

    def resize_bufs(self):
        """resize the storage buffers so they don't go above the buffer size"""
        self.avg_colors_buf = self.avg_colors_buf[-self.buf_size:]
        self.times_buf = self.times_buf[-self.buf_size:]
        self.bpm_buf = self.bpm_buf[-self.buf_size:]
        self.input_buf = self.input_buf[-self.input_buf_size:]
        if self.input_buf:
            self.input_buf = [x for x in self.input_buf if x.shape == self.input_buf[-1].shape]

    def execute(self, text):
        """process the image"""
        self.output = self.input

        self.input_buf.append(self.input)
        self.resize_bufs()

        # fancy progress indicator
        self.buf_state += 1
        self.buf_state %= self.buf_size

        self.times_buf.append(time.time() - self.start_time)

        if not self.lock_face:
            # b&w for face detection classifier
            self.input_g = cv2.equalizeHist(cv2.cvtColor(self.input, cv2.COLOR_BGR2GRAY))

            detect = list(self.classifier.detectMultiScale(self.input_g, minNeighbors=4, minSize=(50, 50),
                                                           flags=cv2.CASCADE_SCALE_IMAGE))

            if detect:
                # take the biggest one (by size)
                biggest = sorted(detect, reverse=True, key=lambda f: f[-1] * f[-2])[0]

                # if difference between detected and current one is bigger than threshold
                # then set current one as detected
                # otherwise, consider it's still the same face, but only moved
                if self.calc_shift(biggest) > SHIFT_THRESHOLD:
                    self.face = biggest
                    self.input_buf.clear()

        # if face is still default then set average to zero so shit doesnt hit the ceiling
        if set(self.face) == {0, 1}:
            self.avg_colors_buf.append(0)
            self.resize_bufs()
            return

        # good-ish numbers, after trial and error
        forehead = self.get_slice(0.5, 0.15, 0.38, 0.14)

        # draw rects around face and forehead
        self.rect(*self.face, BLUE)
        self.rect(*forehead, GREEN)

        text("Face", *self.face[:2], BLUE)
        text("Forehead", *forehead[:2], GREEN)

        if self.colorify:
            new_forehead = self.get_subpicture(forehead)
            left = 100
            total_width = 256
            curve_width = total_width - 2 * left
            alpha = total_width / curve_width
            beta = -left * alpha
            new_forehead = cv2.convertScaleAbs(new_forehead, None, alpha, beta)
            self.draw_at(new_forehead, forehead)

        # take average color of forehead region
        fh_average = self.calc_mean_color(forehead)
        self.avg_colors_buf.append(fh_average)

        self.resize_bufs()

        num_samples = len(self.times_buf)

        if num_samples > MIN_SAMPLE_COUNT:
            # bounds of time buffer
            time_start, time_end = self.times_buf[0], self.times_buf[-1]
            num_samples = len(self.times_buf)

            # calculate fps with smoothing
            self.last_fps = self.fps
            dt = time_end - time_start
            if dt:
                fps = num_samples / dt
                self.fps = fps * FPS_SMOOTHING + self.last_fps * (1 - FPS_SMOOTHING)

            # interpolate the color data on a linear time space
            linear = np.linspace(time_start, time_end, num_samples)

            # filter out useless signals with hamming window
            window = np.hamming(num_samples)

            # interpolate that bitch
            interp = window - np.interp(linear, self.times_buf, self.avg_colors_buf)

            # calculate deviation value-wise
            self.deviation = interp - self.progressive_mean(interp)

            # fourier transform
            fourier_raw = np.fft.rfft(self.deviation)
            self.fourier = np.abs(fourier_raw)

            # create linear frequencies array
            self.frequencies = self.fps / num_samples * np.arange(num_samples / 2 + 1) * 60.

            # create filter for a normal bpm (we don't want a bpm of 3000 because your face is lit by a holy strobe light)
            pos = np.where((self.frequencies > BPM_LOW) & (self.frequencies < self.current_high))

            try:
                # filter out the values
                self.frequencies = self.frequencies[pos]
                self.fourier = self.fourier[pos]

                # find index of freq with the biggest intensity
                # it'll be the bpm
                self.bpmpos = np.argmax(self.fourier)
                self.bpm = self.frequencies[self.bpmpos]

                # store it
                self.bpm_buf.append(self.bpm)

                # calculate the corrected bpm taking in account the previous values
                self.cor_bpm = self.progressive_mean(self.bpm_buf)
                text("d=%.3f" % self.deviation, 20, 120)

                if self.colorify:
                    delta = 0.45 * (np.sin(np.angle(fourier_raw)) + 1.) + 0.1
                    new_forehead = self.get_subpicture(forehead)
                    self.draw_at(cv2.merge([
                        delta * new_forehead[:, :, 0],
                        delta * new_forehead[:, :, 0] + (1 - delta) * self.get_subpicture(forehead, self.input_g),
                        delta * new_forehead[:, :, 0]
                    ]), forehead)
            except:
                pass

            # calculate remaining time until full buffer
            self.gap = (self.buf_size - num_samples) / self.fps
