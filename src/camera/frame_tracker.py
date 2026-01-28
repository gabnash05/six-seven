"""
Tracks frame timing, calculates FPS, detects drops.
Uses circular buffer for timestamps.
"""

from collections import deque
import numpy as np
import time

class FrameTracker:
    def __init__(self, buffer_size=300):
        self.buffer_size = buffer_size
        self.timestamps = deque(maxlen=buffer_size)
        self.dropped_frames = 0
        self.last_frame = None


    def capture_frame(self, frame):
        """Record timestamp when frame is captured"""
        
        if self._is_duplicate(frame):
            self.dropped_frames += 1
            return False
        
        timestamp = time.perf_counter()
        self.timestamps.append(timestamp)

        self.last_frame = frame.copy()
        return True
    
    
    def _is_duplicate(self, frame):
        """Check if frame is identical to last frame"""

        if self.last_frame is None:
            return False
    
        if frame.shape != self.last_frame.shape:
            return False
        
        return np.array_equal(frame, self.last_frame)


    def calculate_fps(self):
        """Calculate current FPS from timestamps"""

        if len(self.timestamps) < 2:
            return 0
        
        time_diff = self.timestamps[-1] - self.timestamps[0]
        if time_diff <= 0:
            return 0
        
        return (len(self.timestamps) - 1) / time_diff


    def get_stats(self):
        """Return timing statistics"""

        if len(self.timestamps) < 2:
            return {"fps": 0, "drops": self.dropped_frames}

        times = list(self.timestamps)
        frame_intervals = np.diff(times)

        return {
            "fps": self.calculate_fps(),
            "drops": self.dropped_frames,
            "avg_interval_ms": np.mean(frame_intervals) * 1000,
            "std_interval_ms": np.std(frame_intervals) * 1000 if len(frame_intervals) > 1 else 0,
            "buffer_fill": len(self.timestamps)
        }
        