# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 09:55:40 2018

@author: nigett
"""

from camera import *
import numpy as np
import argparse
import itertools
import cv2
from processor import *
from logger import *
from sys import exit
from helper import *

SIZE_NORMAL = 1.0
SIZE_SMALL = 0.75
SIZE_BIG = 1.5

FONT_SERIF = cv2.FONT_HERSHEY_COMPLEX
FONT_SERIF_SMALL = cv2.FONT_HERSHEY_COMPLEX_SMALL

class TheApp():
    def __init__(self, cmdargs):  
        info("Starting app with args " + str(cmdargs))
        self.cam_type = cmdargs.type
        self.usb_cameras = []
        
        if self.cam_type == "usb":
            self.usb_id = cmdargs.camera
        
            for i in itertools.count():
                if i > cmdargs.hacky:
                    break # TODO: HACK: UGLY HACK because it crashes if you try to access an invalid ID :(

                cam = Camera(i)
                
                if not cam.connected:
                    break
                
                self.usb_cameras.append(cam)
                
            if self.usb_id >= len(self.usb_cameras):
                error(f"Invalid usb id: {self.usb_id}")
                exit()

            self.camera = self.usb_cameras[self.usb_id]
        elif self.cam_type == "net":
            self.net_url = cmdargs.url
            self.camera = Camera(self.net_url)
        else:
            error(f"Invalid camera type: {self.cam_type}")
            exit()
            
        
        self.key = '\0'
        self.w, self.h, self.channels = 0, 0, 3
        self.infos = True
        self.proc = ImageProcessor(cmdargs.classifier)
        self.font_size = 1
        self.text_color = WHITE
        self.zoom = 1
        self.lastymax = 0
        self.keys = {
            "c": self.next_camera,
            "s": self.show_infos,
            "l": self.lock_toggle,
            "d": self.denoise_toggle,
            "+": self.zoom_plus,
            "-": self.zoom_minus,
            "x": self.proc.clear_bufs,
            "h": self.highbpm_toggle
        }

    def zoom_plus(self):
        self.zoom += 0.2

    def zoom_minus(self):
        self.zoom -= 0.2
        if self.zoom < 1:
            self.zoom = 1

    def text(self, text, x, y, col=None, size=SIZE_NORMAL, font=FONT_SERIF, frame=None):
        """wrapper for opencv"""
        if frame is None:
            frame = self.proc.output
        for ox, oy in itertools.product([-1, 1], repeat=2):
            cv2.putText(frame, text, (x + ox, y + oy), font, self.font_size * size, BLACK, 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), font, self.font_size * size, col or self.text_color, lineType=cv2.LINE_AA)

    def print(self, text, frame=None, col=None):
        self.text(text, 20, int(30 + self.text_row * 30), font=FONT_SERIF_SMALL, frame=frame, col=col)
        self.text_row += 1

    def app_loop(self):
        frame = self.camera.frame()
        self.h, self.w, self.channels = frame.shape
        cx, cy = int(self.w / 2), int(self.h / 2)
        rx, ry = int(self.w / self.zoom / 2), int(self.h / self.zoom / 2)
        minx, maxx = cx - rx, cx + rx
        miny, maxy = cy - ry, cy + ry
        cropped = frame[miny:maxy, minx:maxx]
        frame = cv2.resize(cropped, (self.w, self.h))

        self.text_row = 0
        
        self.proc.input = frame
        self.proc.execute(self.text)

        lines = [
            ("l = lock face", self.proc.lock_face),
            "c = next camera (if usb)" if self.cam_type == "usb" else None,
            ("s = show infos (fps, ...)", self.infos),
            ("d = toggle de-noise", self.proc.denoise),
            ("h = toggle noisy high-bpm filter", self.proc.highbpm),
            "+ - = change zoom",
            "esc = exit"
        ]

        keys_frame = get_frame(400, 30 + len(lines) * 30)

        for l in lines:
            if l is not None:
                if type(l) != str:
                    l, ok = l
                    col = GREEN if ok else RED
                else:
                    col = WHITE
                self.print(l, col=col, frame=keys_frame)

        cv2.imshow("infos", keys_frame)

        self.text_row = 0

        if self.infos:
            self.display_infos()
        
        cv2.imshow("processed", self.proc.output)
        
        self.handle_keystroke()


    def display_infos(self):
        self.text("%.0f bpm" % self.proc.cor_bpm, 15, 50, size=SIZE_BIG)
        self.text_row += SIZE_BIG / SIZE_SMALL
        self.print("%.0f bpm (raw)" % self.proc.bpm)
        self.print("%.0f fps" % self.proc.fps)
        self.print("d = %.3f" % self.proc.deviation[-1])
        if self.proc.gap:
            self.print("wait %.0fs" % self.proc.gap)
        #plotXY([[self.proc.frequencies, self.proc.fourier]],size=(300, 600), name="data", labels=[True], showmax=["bpm"], label_ndigits=[0], showmax_digits=[1], skip=[3])
        #return
        ymax_smoothing = 0.5
        h, w = 300, 600
        graph_frame = get_frame(w, h)
        if list(self.proc.frequencies):
            xmin = BPM_LOW
            xmax = BPM_HIGH
            xs = xmax - xmin
            xo = 20

            ymin, ymax = min(self.proc.fourier), max(self.proc.fourier)
            ymax = ymax_smoothing * ymax + self.lastymax * (1 - ymax_smoothing)
            self.lastymax = ymax
            ymin = 0
            ys = ymax - ymin
            yo = 20
            margin = 20
            use_w = w - 2 * xo
            def fix_point(p):
                x, y = p
                x = xo + int(round((x - xmin) / xs * (w - 2 * xo)))
                y = yo + int(round((1 - (y - ymin) / ys) * (h - 2 * yo - margin)))
                return x, y

            points = np.array(list(map(fix_point, (zip(self.proc.frequencies, self.proc.fourier)))), dtype=np.int32)
            cv2.polylines(graph_frame, [points], False, WHITE, lineType=cv2.LINE_AA)
            cv2.line(graph_frame, fix_point((self.proc.bpm, ymin)), fix_point((self.proc.bpm, self.proc.fourier[self.proc.bpmpos])), RED)

            if self.proc.highbpm:
                bx = round(fix_point((BPM_NOISE_HIGH, 0))[0])
                cv2.line(graph_frame, (bx, ymin), (bx, h), RED, 3)


            #tuples = zip(points, points[1:])
            #for (p1, p2) in tuples:
            #    cv2.line(graph_frame, fix_point(p1), fix_point(p2), WHITE)
            spac = use_w / 12
            val_spac = xs / 12
            for i in range(12):
                cv2.putText(graph_frame, "%.0f" % (i * val_spac + xmin), (round(xo + i * spac), h - 30), cv2.FONT_HERSHEY_PLAIN, 1, WHITE, lineType=cv2.LINE_AA)

            cv2.putText(graph_frame, "%.0f" % self.proc.bpm, (round(fix_point((self.proc.bpm, 0))[0]), h - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, RED, lineType=cv2.LINE_AA)

        cv2.line(graph_frame, (0, h), (int(round(self.proc.buf_state / self.proc.buf_size * w)), h), GREEN, 5)
        cv2.imshow("graph", graph_frame)
        
    def lock_toggle(self):     
        info("Face lock " + ("enabled" if self.proc.lock_toggle() else "diasbled"))
        if self.proc.lock_face:
            self.proc.clear_bufs()

    def denoise_toggle(self):
        info("De-noise filter " + ("enabled" if self.proc.denoise_toggle() else "disabled"))

    def highbpm_toggle(self):
        info("High BPM (noise) filter " + ("enabled" if self.proc.highbpm_toggle() else "disabled"))
        
    def handle_keystroke(self):
        self.key = cv2.waitKey(50) & 0xFF
        
        if self.key == 0x1B: # ascii ESC
            info("good bye")
            
            for c in self.usb_cameras:
                c.dispose()
                
            exit()
            
        for key, handler in self.keys.items():
            if self.key == ord(key):
                handler()
        
    def next_camera(self):
        if self.cam_type == "usb":
            self.usb_id = (self.usb_id + 1) % len(self.usb_cameras)
            self.camera = self.usb_cameras[self.usb_id]
            info(f"Switched to usb {self.usb_id}")
    
    def show_infos(self):
        self.infos = not self.infos
        info("Infos " + ("enabled" if self.infos else "disabled"))
            

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="TPE")
    argparser.add_argument("type", type=str, help="camera type", choices=["usb", "net"], default="net")
    argparser.add_argument("-u", "--url", type=str, help="url address or file path", default="http://192.168.0.48:8080/video")
    argparser.add_argument("-c", "--camera", type=int, help="id of usb camera", default=0)

    argparser.add_argument("--classifier", type=str, help="cascade classifier file (xml)", default="haarcascade_frontalface_alt.xml")
    argparser.add_argument("-hh", "--hacky", type=int, help="max id to scan (hack)", default=0)
    
    app = TheApp(argparser.parse_args())
    
    while True:
        app.app_loop()
    