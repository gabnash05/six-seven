"""
Vertical position normalization and temporal smoothing.
Converts raw landmarks into stable, comparable vertical positions.
"""

import time
import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict, Any

class TemporalBuffer:
    """Buffer for temporal smoothing of hand positions"""

    def __init__(self, window_size: int = 7):
        self.window_size = window_size
        self.left_buffer = deque(maxlen=window_size)
        self.right_buffer = deque(maxlen=window_size)

    
    def update(self, left_y: Optional[float], right_y: Optional[float], timestamp: float):
        """Update buffers with new hand positions"""

        if left_y is not None:
            self.left_buffer.append((left_y, timestamp))
        if right_y is not None:
            self.right_buffer.append((right_y, timestamp))
        
    def get_smoothed(self) -> Tuple[Optional[float], Optional[float]]:
        """Get median-smoothed positions (rejects outliers)"""

        left_smooth = None
        right_smooth = None
        
        if self.left_buffer:
            left_smooth = np.median([y for y, _ in self.left_buffer])
        
        if self.right_buffer:
            right_smooth = np.median([y for y, _ in self.right_buffer])
        
        return left_smooth, right_smooth


    def estimate_velocity(self, hand: str = "left") -> float:
        """Estimate velocity using last 3 positions and times"""
        
        buffer = self.left_buffer if hand == "left" else self.right_buffer
        
        if len(buffer) < 3:
            return 0.0
        
        # Extract positions and timestamps
        positions = [y for y, t in buffer]
        times = [t for y, t in buffer]
        
        # Simple velocity calculation
        velocity = (positions[-1] - positions[0]) / (times[-1] - times[0])
        return velocity
    

    def get_buffer_stats(self, hand: str = "left") -> Dict[str, Any]:
        """Get statistics about the buffer contents"""

        buffer = self.left_buffer if hand == "left" else self.right_buffer
        
        if not buffer:
            return {
                "size": 0,
                "filled": False,
                "mean": None,
                "std": None,
                "range": None,
                "stability": 0.0
            }
        
        positions = [y for y, _ in buffer]
        
        return {
            "size": len(buffer),
            "filled": len(buffer) == self.window_size,
            "mean": float(np.mean(positions)),
            "std": float(np.std(positions)),
            "range": float(np.ptp(positions)),
            "stability": 1.0 / (1.0 + np.std(positions))  # Higher = more stable
        }    

class HandNormalizer:
    """Normalizes hand positions to torso-relative coordinates"""

    def __init__(self, config):
        self.config = config

        buffer_size = config.get('temporal_buffer_size', 7)
        self.temporal_buffer = TemporalBuffer(window_size=buffer_size)

        self.torso_center_y = 0.5  # Initial guess (middle of frame)
        self.shoulder_width = 0.2  # Initial guess (20% of frame width)
        self.torso_height = 0.6    # Initial guess

        self.last_update_time = time.perf_counter()
        self.smoothing_enabled = config.get('enable_smoothing', True)
        self.reference_established = False

        self.normalized_positions_history = deque(maxlen=100)
        self.raw_positions_history = deque(maxlen=100)

        self.debug_mode = config.get('debug_mode', False)


    def update_reference_frame(self, left_hand, right_hand, frame_shape):
        """
        Dynamically estimate torso reference points using wrist positions
        
        Args:
            left_hand: HandData object or None
            right_hand: HandData object or None
            frame_shape: (height, width, channels) of frame
        """

        frame_height, frame_width = frame_shape[:2]

        left_wrist_y = left_hand.wrist_position[1] if left_hand else None
        right_wrist_y = right_hand.wrist_position[1] if right_hand else None
        left_wrist_x = left_hand.wrist_position[0] if left_hand else None
        right_wrist_x = right_hand.wrist_position[0] if right_hand else None

        if None in [left_wrist_y, right_wrist_y, left_wrist_x, right_wrist_x]:
            return False
        
        left_wrist_y_norm = left_wrist_y / frame_height
        right_wrist_y_norm = right_wrist_y / frame_height
        left_wrist_x_norm = left_wrist_x / frame_width
        right_wrist_x_norm = right_wrist_x / frame_width

        self.torso_center_y = (left_wrist_y_norm + right_wrist_y_norm) / 2
        self.shoulder_width = abs(left_wrist_x_norm - right_wrist_x_norm)
        self.torso_height = self.shoulder_width * 2.5

        min_torso_height = 0.2
        if self.torso_height < min_torso_height:
            self.torso_height = min_torso_height

        self.reference_established = True

        if self.debug_mode:
            print(f"Reference updated: center={self.torso_center_y:.3f}, "
                  f"width={self.shoulder_width:.3f}, height={self.torso_height:.3f}")
        
        return True
    

    def normalize_vertical_position(self, hand_y: float, frame_height: int) -> Optional[float]:
        """
        Convert raw Y position to torso-relative normalized position
        
        Args:
            hand_y: Raw Y coordinate in pixels
            frame_height: Frame height in pixels
            
        Returns:
            Normalized position where:
            0 = torso centerline
            +0.33 ≈ shoulder height
            -0.33 ≈ waist height
            None if reference not established
        """

        if not self.reference_established:
            return None
        
        hand_y_norm = hand_y / frame_height

        normalized_y = (hand_y_norm - self.torso_center_y) / self.torso_height
        normalized_y = np.clip(normalized_y, -1.0, 1.0)
        
        return normalized_y
    

    def process_hands(self, left_hand, right_hand, frame_shape):
        """
        Process hand data to get normalized, smoothed positions
        
        Args:
            left_hand: HandData object or None
            right_hand: HandData object or None
            frame_shape: (height, width, channels) of frame
            
        Returns:
            Dictionary with normalized and smoothed positions
        """
        timestamp = time.perf_counter()
        frame_height, frame_width = frame_shape[:2]
        
        if left_hand is not None and right_hand is not None:
            self.update_reference_frame(left_hand, right_hand, frame_shape)
        
        left_normalized = None
        right_normalized = None

        if left_hand is not None:
            left_normalized = self.normalize_vertical_position(left_hand.wrist_position[1], frame_height)
        
        if right_hand is not None:
            right_normalized = self.normalize_vertical_position(right_hand.wrist_position[1], frame_height)
        
        self.raw_positions_history.append((left_normalized, right_normalized, timestamp))

        # Temporal smoothing
        if self.smoothing_enabled:
            self.temporal_buffer.update(left_normalized, right_normalized, timestamp)
            left_smoothed, right_smoothed = self.temporal_buffer.get_smoothed()
        else:
            left_smoothed, right_smoothed = left_normalized, right_normalized
        
        if left_smoothed is not None or right_smoothed is not None:
            self.normalized_positions_history.append((left_smoothed, right_smoothed, timestamp))
        
        left_velocity = 0.0
        right_velocity = 0.0

        if left_smoothed is not None and len(self.temporal_buffer.left_buffer) >= 3:
            left_velocity = self.temporal_buffer.estimate_velocity("left")
        
        if right_smoothed is not None and len(self.temporal_buffer.right_buffer) >= 3:
            right_velocity = self.temporal_buffer.estimate_velocity("right")

        return {
            "left": {
                "raw": left_normalized,
                "smoothed": left_smoothed,
                "velocity": left_velocity,
                "detected": left_hand is not None,
                "confidence": left_hand.handedness_confidence if left_hand else 0.0
            } if left_hand is not None else None,
            "right": {
                "raw": right_normalized,
                "smoothed": right_smoothed,
                "velocity": right_velocity,
                "detected": right_hand is not None,
                "confidence": right_hand.handedness_confidence if right_hand else 0.0
            } if right_hand is not None else None,
            "reference": {
                "established": self.reference_established,
                "torso_center_y": self.torso_center_y,
                "shoulder_width": self.shoulder_width,
                "torso_height": self.torso_height
            },
            "timestamp": timestamp,
            "frame_height": frame_height
        }
    
    def get_normalization_stats(self) -> Dict[str, Any]:
        """Get statistics about normalization performance"""
        
        # Extract position arrays from history
        left_positions = []
        right_positions = []
        
        for left, right, _ in self.normalized_positions_history:
            if left is not None:
                left_positions.append(left)
            if right is not None:
                right_positions.append(right)
        
        left_positions = np.array(left_positions)
        right_positions = np.array(right_positions)
        
        stats = {
            "reference_established": self.reference_established,
            "temporal_buffer": {
                "left": self.temporal_buffer.get_buffer_stats("left"),
                "right": self.temporal_buffer.get_buffer_stats("right")
            },
            "history_size": len(self.normalized_positions_history),
            "smoothing_enabled": self.smoothing_enabled
        }
        
        if len(left_positions) > 0:
            stats["left"] = {
                "mean": float(np.mean(left_positions)),
                "std": float(np.std(left_positions)),
                "min": float(np.min(left_positions)),
                "max": float(np.max(left_positions)),
                "range": float(np.ptp(left_positions))
            }
        
        if len(right_positions) > 0:
            stats["right"] = {
                "mean": float(np.mean(right_positions)),
                "std": float(np.std(right_positions)),
                "min": float(np.min(right_positions)),
                "max": float(np.max(right_positions)),
                "range": float(np.ptp(right_positions))
            }
        
        return stats
    

    def reset(self):
        """Reset normalization state"""
        
        self.temporal_buffer = TemporalBuffer(window_size=self.temporal_buffer.window_size)
        self.reference_established = False
        self.normalized_positions_history.clear()
        self.raw_positions_history.clear()