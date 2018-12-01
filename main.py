# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 09:55:40 2018

@author: nigett
"""

import argparse

from app import TheApp

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="TPE")
    argparser.add_argument("type", type=str, help="camera type", choices=["usb", "net"], default="net")
    argparser.add_argument("-u", "--url", type=str, help="url address or file path",
                           default="http://192.168.0.48:8080/video")
    argparser.add_argument("-c", "--camera", type=int, help="id of usb camera", default=0)

    argparser.add_argument("--classifier", type=str, help="cascade classifier file (xml)",
                           default="haarcascade_frontalface_alt.xml")
    argparser.add_argument("-hh", "--hacky", type=int, help="max id to scan (hack)", default=0)
    argparser.add_argument("-r", "--rotate", type=int, help="rotation to apply (either -90, 90 or 180)",
                           choices=[-90, 90, 180], default=0)

    app = TheApp(argparser.parse_args())

    while True:
        app.app_loop()
