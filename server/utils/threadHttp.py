import cv2

from http.server import HTTPServer
from socketserver import ThreadingMixIn


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):

    def __init__(self, capture_path, server_address, request_handler, bind_and_activate=True):
        HTTPServer.__init__(self, server_address, request_handler, bind_and_activate)
        ThreadingMixIn.__init__(self)
        if capture_path.isdigit():
            capture_path = int(capture_path)
        self._capture_path = capture_path
        self._camera = cv2.VideoCapture(capture_path)

    def open_video(self):
        if not self._camera.isOpened():
            raise IOError('Не возможно открыть камеру {}'.format(self._capture_path))

    def read_frame(self):
        ret, img = self._camera.read()
        return img

    def serve_forever(self, poll_interval=0.5):
        self.open_video()
        try:
            super().serve_forever(poll_interval)
        except KeyboardInterrupt:
            self._camera.release()
