import itertools
from collections import defaultdict
from sys import exit

import cv2
import numpy as np

from camera import Camera
from helper import WHITE, BLACK, get_frame, GREEN, RED, BPM_LOW
from logger import info, error
from processor import ImageProcessor

SIZE_NORMAL = 0.75
SIZE_SMALL = 0.5
SIZE_BIG = 1.5
FONT_SERIF = cv2.FONT_HERSHEY_TRIPLEX
FONT_SERIF_SMALL = cv2.FONT_HERSHEY_TRIPLEX


class TheApp:
    def __init__(self, cmdargs):
        info("Starting app with args " + str(cmdargs))
        self.cam_type = cmdargs.type
        self.rotate = cmdargs.rotate
        self.usb_cameras = []

        if self.cam_type == "usb":
            self.usb_id = cmdargs.camera

            for i in itertools.count():
                if i > cmdargs.hacky:
                    break  # TODO: HACK: UGLY HACK because it crashes if you try to access an invalid ID :(

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
        self.text_row = 0
        self.zoom = 1
        self.last_y_max = defaultdict(int)
        self.keys = {
            "c": self.next_camera,
            "s": self.show_infos,
            "l": self.lock_toggle,
            "d": self.denoise_toggle,
            "+": self.zoom_plus,
            "-": self.zoom_minus,
            "x": self.proc.clear_bufs,
            "h": self.highbpm_toggle,
            "i": self.colorify_toggle
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
        #for ox, oy in itertools.product([-1, 1], repeat=2):
        #    cv2.putText(frame, text, (x + ox, y + oy), font, self.font_size * size, BLACK, 2, cv2.LINE_AA)
        #cv2.putText(frame, text, (x, y), font, self.font_size * size, BLACK, 4, lineType=cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), font, self.font_size * size, col or self.text_color, lineType=cv2.LINE_AA)

    def print(self, text, frame=None, col=None):
        self.text(text, 20, int(30 + self.text_row * 30), font=FONT_SERIF_SMALL, frame=frame, col=col)
        self.text_row += 1

    def get_width(self, text):
        if text is None:
            return 0
        if type(text) != str:
            text = text[0]

        return cv2.getTextSize(text, FONT_SERIF_SMALL, self.font_size * SIZE_NORMAL, 2)[0][0]

    def app_loop(self):
        ftime, frame = self.camera.get_frame()

        if self.rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotate == -90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        self.h, self.w, self.channels = frame.shape
        cx, cy = int(self.w / 2), int(self.h / 2)
        rx, ry = int(self.w / self.zoom / 2), int(self.h / self.zoom / 2)
        minx, maxx = cx - rx, cx + rx
        miny, maxy = cy - ry, cy + ry
        cropped = frame[miny:maxy, minx:maxx]
        frame = cv2.resize(cropped, (self.w, self.h))

        self.text_row = 0

        self.proc.input = frame
        self.proc.input_time = ftime
        try:
            self.proc.execute(self.text)
        except:
            raise

        if self.infos:
            lines = [
                ("l = lock face", self.proc.lock_face),
                ("c = next camera (if usb)" if self.cam_type == "usb" else None, None),
                ("s = show infos (fps, ...)", self.infos),
                ("d = toggle de-noise", self.proc.denoise),
                ("h = toggle noisy high-bpm filter", self.proc.highbpm),
                ("i = show enhanced color intensity", self.proc.colorify),
                ("x = clear data buffers", None),
                ("+ - = change zoom", None),
                ("esc = exit", None)
            ]

            keys_frame = get_frame(40 + max([(l[0], self.get_width(l[0])) for l in lines], key=lambda x: x[1])[1],
                                   20 + len(lines) * 30)

            for l, ok in lines:
                if l is not None:
                    col = {
                        None: WHITE,
                        False: RED,
                        True: GREEN
                    }[ok]
                    self.print(l, col=col, frame=keys_frame)

            cv2.imshow("infos", keys_frame)

            self.text_row = 0

        self.display_infos()

        cv2.imshow("processed", self.proc.output)

        self.handle_keystroke()

    @staticmethod
    def fix_point(p, x_min, x_max, y_min, y_max, x_target, y_target, x_off, y_off):
        x, y = p
        # values starting at 0
        x_0 = x - x_min
        y_0 = y - y_min

        # values normalized (between 0 and 1) ; invert Y orientation
        x_n = x_0 / (x_max - x_min)
        y_n = 1 - y_0 / (y_max - y_min)

        # values fixed in their target size
        x_f = x_n * x_target
        y_f = y_n * y_target

        # final values
        x = x_off + int(round(x_f))
        y = y_off + int(round(y_f))
        return x, y

    x_off = 20
    y_off = 40

    def draw_xy(self, x, y, x_min, x_max, frame, name):
        y_max_smoothing = 0.5
        h, w = 300, 600

        y_min, y_max = min(y), max(y)
        y_max = y_max_smoothing * y_max + self.last_y_max[name] * (1 - y_max_smoothing)
        self.last_y_max[name] = y_max
        y_min = 0

        y_margin = 20

        x_target = w - 2 * self.x_off
        y_target = h - 2 * self.y_off - y_margin

        def fix(p):
            return self.fix_point(p, x_min, x_max, y_min, y_max, x_target, y_target, self.x_off, self.y_off)

        points = np.array(list(map(fix, zip(x, y))), dtype=np.int32)

        cv2.polylines(frame, [points], False, WHITE, lineType=cv2.LINE_AA)

        return fix, (x_target, y_target)

    def display_infos(self):
        self.text(("%.0f bpm" % self.proc.cor_bpm) if self.proc.gap <= 0 else "?? bpm", 15, 50, size=SIZE_BIG)
        self.text_row += SIZE_BIG / SIZE_NORMAL
        self.print("%.0f bpm (raw)" % self.proc.bpm)
        self.print("%.0f fps" % self.proc.fps)
        self.print("d = %.3f" % self.proc.deviation[-1])
        if self.proc.gap:
            self.print("wait %.0fs" % self.proc.gap)

        h, w = 300, 600
        graph_frame = get_frame(w, h)
        if list(self.proc.frequencies):
            x_min = BPM_LOW
            x_max = self.proc.current_high
            x_size = x_max - x_min
            y_min = 0

            fix_point, (x_target, _) = self.draw_xy(self.proc.frequencies, self.proc.fourier, x_min, x_max, graph_frame,
                                                    "fourier")

            cv2.line(graph_frame, fix_point((self.proc.bpm, y_min)),
                     fix_point((self.proc.bpm, self.proc.fourier[self.proc.bpmpos])), RED)

            spac = x_target / 12
            val_spac = x_size / 12
            for i in range(12):
                cv2.putText(graph_frame, "%.0f" % (i * val_spac + x_min), (round(self.x_off + i * spac), h - 30),
                            cv2.FONT_HERSHEY_PLAIN, 1, WHITE, lineType=cv2.LINE_AA)

            cv2.putText(graph_frame, "%.0f" % self.proc.bpm, (round(fix_point((self.proc.bpm, 0))[0]), h - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, RED, lineType=cv2.LINE_AA)

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

    def colorify_toggle(self):
        info("Colorization " + ("enabled" if self.proc.colorify_toggle() else "disabled"))

    def handle_keystroke(self):
        self.key = cv2.waitKey(50) & 0xFF

        if self.key == 0x1B:  # ascii ESC
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
