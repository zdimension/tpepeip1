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

class TheApp():
    def __init__(self, cmdargs):  
        info("Starting app with args " + str(cmdargs))
        self.cam_type = cmdargs.type
        self.usb_cameras = []
        
        if self.cam_type == "usb":
            self.usb_id = cmdargs.camera
        
            for i in itertools.count():
                if i > 0:
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
            
        self.keys = {
            "c": self.next_camera,
            "s": self.show_infos,
            "l": self.lock_toggle
        }
        self.key = '\0'
        self.w, self.h, self.channels = 0, 0, 3
        
        self.proc = ImageProcessor(cmdargs.classifier)
        
    def app_loop(self):
        frame = self.camera.frame()

        self.h, self.w, self.channels = frame.shape
        cv2.imshow("original", frame)
        
        self.proc.input = frame
        self.proc.execute()
        
        cv2.imshow("processed", self.proc.output)
        
        self.handle_keystroke()
        
    def lock_toggle(self):     
        info("Face lock " + "enabled" if self.proc.lock_toggle() else "diasbled")
        
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
        info("Infos " + "enabled" if self.infos else "disabled")
            

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="TPE")
    argparser.add_argument("type", type=str, help="camera type", choices=["usb", "net"], default="net")
    argparser.add_argument("-u", "--url", type=str, help="url address", default="http://192.168.42.129:8080/video")
    argparser.add_argument("-c", "--camera", type=int, help="id of usb camera", default=0)
    argparser.add_argument("--classifier", type=str, help="cascade classifier file (xml)", default="haarcascade_frontalface_alt.xml")
    
    app = TheApp(argparser.parse_args())
    
    while True:
        app.app_loop()
    