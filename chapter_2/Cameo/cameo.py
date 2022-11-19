import time

import cv2
from managers import CaptureManager, WindowManager


class Cameo(object):
    def __init__(self):
        self._windowManager = WindowManager('Cameo', self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager, True)

    def run(self):
        """Run the main loop"""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            if frame is not None:
                # TODO: filter
                pass

            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        """Handle a keypress.
        space -> Take a screenshot
        tab -> Start/stop recording a screencast
        escape -> Quit
        """
        if keycode == 32:
            print("ord(space) ==", ord(' '))
            self._captureManager.writeImage('screenshot_'+str(int(time.time() % 10000))+'.png')
        elif keycode == 9:
            print("ord(tab) ==", ord('\t'))
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('screencast_'+str(int(time.time() % 10000))+'.mp4')
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 27:
            self._windowManager.destroyWindow()