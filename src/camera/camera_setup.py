"""
Handles camera initialization with DirectShow backend.
Configures resolution, FPS, and disables auto controls.
"""

import cv2

class Camera:
    def __init__(self, config):
        self.config = config
        self.cap = None
    
    
    def initialize(self):
        """Initialize camera with DirectShow backend"""
        self.cap = cv2.VideoCapture(self.config['device_index'], cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.config['device_index'])
            raise RuntimeError(f"Failed to open camera at index {self.config['device_index']}")

        self.apply_settings()

    def apply_settings(self):
        """Apply camera settings from config"""

        # Resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['resolution']['width'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['resolution']['height'])

        # FPS
        self.cap.set(cv2.CAP_PROP_FPS, self.config['fps']['target'])
        
        # Auto exposure
        if not self.config['controls']['auto_exposure']:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.config['controls']['exposure_value'])
        
        # Auto focus
        if not self.config['controls']['auto_focus']:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Manual focus
        
        # Auto white balance (if supported)
        if not self.config['controls']['auto_white_balance']:
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        
        # Buffer size for low latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config['optimization']['buffer_size'])
    

    def release(self):
        """Clean up camera"""
        if self.cap:
            self.cap.release()