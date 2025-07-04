import cv2
import threading
import time
from threading import Thread

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)  # Use default camera
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video.set(cv2.CAP_PROP_FPS, 30)
        
        # Threading variables
        self.frame = None
        self.thread = None
        self.stopped = False
        
        # Start the camera thread
        self.start()
    
    def start(self):
        """Start the camera thread"""
        if self.thread is None or not self.thread.is_alive():
            self.stopped = False
            self.thread = Thread(target=self.update)
            self.thread.daemon = True
            self.thread.start()
    
    def update(self):
        """Update frame continuously in background thread"""
        while not self.stopped:
            ret, frame = self.video.read()
            if ret:
                self.frame = frame
            time.sleep(0.03)  # ~30 FPS
    
    def get_frame(self):
        """Get the latest frame from camera"""
        if self.frame is not None:
            return self.frame.copy()
        return None
    
    def stop(self):
        """Stop the camera thread"""
        self.stopped = True
        if self.thread is not None:
            self.thread.join()
    
    def __del__(self):
        """Clean up camera resources"""
        self.stop()
        if self.video.isOpened():
            self.video.release()

class IPCamera:
    """For IP camera support (future enhancement)"""
    def __init__(self, ip_url):
        self.video = cv2.VideoCapture(ip_url)
        self.frame = None
        self.thread = None
        self.stopped = False
        self.start()
    
    def start(self):
        """Start the IP camera thread"""
        if self.thread is None or not self.thread.is_alive():
            self.stopped = False
            self.thread = Thread(target=self.update)
            self.thread.daemon = True
            self.thread.start()
    
    def update(self):
        """Update frame continuously in background thread"""
        while not self.stopped:
            ret, frame = self.video.read()
            if ret:
                self.frame = frame
            time.sleep(0.03)
    
    def get_frame(self):
        """Get the latest frame from IP camera"""
        if self.frame is not None:
            return self.frame.copy()
        return None
    
    def stop(self):
        """Stop the IP camera thread"""
        self.stopped = True
        if self.thread is not None:
            self.thread.join()
    
    def __del__(self):
        """Clean up IP camera resources"""
        self.stop()
        if self.video.isOpened():
            self.video.release()