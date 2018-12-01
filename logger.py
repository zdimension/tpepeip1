# -*- coding: utf-8 -*-

# helper functions

def wrap(text, *args):
    print("[%5s] " % text, end="")
    print(*args)


def info(*args):
    wrap("INFO", *args)


def error(*args):
    wrap("ERROR", *args)


def warn(*args):
    wrap("WARN", *args)


__all__ = ["info", "error", "warn"]
