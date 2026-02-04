"""
MediaPipe Hand Detection Integration
Handles bilateral hand tracking with minimal latency.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import queue
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from mediapipe.tasks import python
from collections import deque
from mediapipe.tasks.python import vision
from mediapipe import ImageFormat


@dataclass
class HandData:
    """Container for all data about a detected hand"""
    
    landmarks: np.ndarray               # 21 landmarks in 3D (x, y, z)
    world_landmarks: np.ndarray        # World coordinates landmarks
    handedness: str                     # "Left" or "Right"
    handedness_confidence: float        # Handedness confidence (0-1)
    wrist_position: Tuple[float, float] # (x, y) of wrist
    palm_centroid: Tuple[float, float]  # Center of palm
    palm_normal: np.ndarray             # Vector normal to palm
    timestamp: float                    # When detected
    id: int                             # Persistent ID across frames
    frame_id: int                       # Frame ID

    def __init__(self, 
                 landmarks: np.ndarray,
                 world_landmarks: np.ndarray,
                 handedness: str,
                 handedness_confidence: float,
                 wrist_position: Tuple[float, float],
                 palm_centroid: Tuple[float, float],
                 timestamp: float,
                 id: int,
                 frame_id: int = -1):
        self.landmarks = landmarks
        self.world_landmarks = world_landmarks
        self.handedness = handedness
        self.handedness_confidence = handedness_confidence
        self.wrist_position = wrist_position
        self.palm_centroid = palm_centroid
        self.timestamp = timestamp
        self.id = id
        self.frame_id = frame_id
        
        # Calculate palm normal
        self.palm_normal = self._calculate_palm_normal(world_landmarks)
    
    @property
    def confidence(self) -> float:
        """Alias for handedness_confidence for compatibility"""
        return self.handedness_confidence

    @property
    def is_left(self) -> bool:
        return self.handedness == "Left"
    
    @property
    def is_right(self) -> bool:
        return self.handedness == "Right"

    def _calculate_palm_normal(self, world_landmarks: np.ndarray) -> np.ndarray:
        """Calculate palm normal vector from world landmarks"""
        # Simple approximation using palm triangle
        wrist = world_landmarks[0]
        index_mcp = world_landmarks[5]
        pinky_mcp = world_landmarks[17]
        
        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        else:
            normal = np.array([0, 0, 1])  # Default upward normal
        return normal


class HandDetector:
    """Wrapper for MediaPipe Hands with optimized configuration for real-time games."""

    LANDMARK_INDICES = {
        'WRIST': 0,

        'THUMB_CMC': 1,
        'THUMB_MCP': 2,
        'THUMB_IP': 3,
        'THUMB_TIP': 4,
        
        'INDEX_FINGER_MCP': 5,
        'INDEX_FINGER_PIP': 6,
        'INDEX_FINGER_DIP': 7,
        'INDEX_FINGER_TIP': 8,
        
        'MIDDLE_FINGER_MCP': 9,
        'MIDDLE_FINGER_PIP': 10,
        'MIDDLE_FINGER_DIP': 11,
        'MIDDLE_FINGER_TIP': 12,
        
        'RING_FINGER_MCP': 13,
        'RING_FINGER_PIP': 14,
        'RING_FINGER_DIP': 15,
        'RING_FINGER_TIP': 16,
        
        'PINKY_MCP': 17,
        'PINKY_PIP': 18,
        'PINKY_DIP': 19,
        'PINKY_TIP': 20,
    }


    def __init__(self, config):
        self.config = config
        self.result_queue = queue.Queue(maxsize=5)  # Buffer up to 5 results
        self.last_frame_id = 0
        self.frame_id_lock = threading.Lock()

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        model_path = config.get('model_path', 'models/hand_landmarker.task')

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands=self.config.get('num_hands', 2),
            min_hand_detection_confidence=self.config.get('min_hand_detection_confidence', 0.5),
            min_hand_presence_confidence=self.config.get('min_hand_presence_confidence', 0.4),
            min_tracking_confidence=self.config.get('min_tracking_confidence', 0.4),
            result_callback=self._result_callback
        )
        
        # Create the landmarker
        self.landmarker = HandLandmarker.create_from_options(options)

        # Tracking state
        self.left_hand: Optional[HandData] = None
        self.right_hand: Optional[HandData] = None
        self.left_hand_history = deque(maxlen=10)  # Track recent positions
        self.right_hand_history = deque(maxlen=10)
        self.miss_counter = 0
        self.hand_id_counter = 0
        
        # Performance tracking
        self.detection_latencies = deque(maxlen=100)
        self.processing_times = deque(maxlen=100)
        self.successful_detections = 0
        self.total_frames = 0
        self.last_detection_time = time.perf_counter()
        
        # Frame buffer for when detection lags
        self.frame_buffer = deque(maxlen=3)
        
        # Debug flag
        self.debug_mode = config.get('debug_mode', False)
    

    def _result_callback(self, result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        """
        Async callback - puts result in queue immediately.
        """
        try:
            # Put result in queue (non-blocking if possible)
            self.result_queue.put_nowait((result, timestamp_ms))
        except queue.Full:
            # Queue full, drop oldest result
            try:
                self.result_queue.get_nowait()  # Remove oldest
                self.result_queue.put_nowait((result, timestamp_ms))  # Add new
            except queue.Empty:
                pass
                

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[HandData], Optional[HandData]]:
        """
        Process a single frame to detect hands using new Tasks API.
        
        Workflow:
        1. Convert BGR to RGB and create MediaPipe Image
        2. Pass to landmarker.detect_async() with timestamp
        3. Wait for callback result
        4. Process results into HandData objects
        
        Returns:
            Tuple of (left_hand_data, right_hand_data) - either can be None
        """
        
        self.total_frames += 1
        process_start = time.perf_counter()

        # Get unique frame ID for synchronization
        with self.frame_id_lock:
            frame_id = self.last_frame_id
            self.last_frame_id += 1
        
        # Store frame in buffer (with metadata)
        frame_data = {
            'id': frame_id,
            'timestamp': time.perf_counter(),
            'shape': frame.shape
        }
        self.frame_buffer.append(frame_data)

        # Convert BGR to RGB
        # MediaPipe requires RGB, OpenCV captures in BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=ImageFormat.SRGB, data=rgb_frame)

        # Pass to mediapipe
        self.landmarker.detect_async(mp_image, frame_id)

        self._check_for_results(frame_id)

        # Use predictive tracking if no recent detection
        if time.perf_counter() - self.last_detection_time > 0.1:  # 100ms since last detection
            self._predict_missing_hands()
        
        processing_time = (time.perf_counter() - process_start) * 1000
        self.processing_times.append(processing_time)

        return self.left_hand, self.right_hand
    

    def _check_for_results(self, current_frame_id: int):
        """Check for and process any available results without blocking."""

        try:
            while True:
                result, result_frame_id = self.result_queue.get_nowait()
                
                # Process results
                self._process_detection_result(result, result_frame_id)
                self.last_detection_time = time.perf_counter()
                
                # Performance
                latency = (time.perf_counter() - 
                          self._get_frame_timestamp(result_frame_id)) * 1000
                self.detection_latencies.append(latency)

        except queue.Empty:
            pass
    

    def _get_frame_timestamp(self, frame_id: int) -> float:
        """Find the timestamp when a frame was submitted"""

        for frame_data in self.frame_buffer:
            if frame_data['id'] == frame_id:
                return frame_data['timestamp']
        
        # Fallback
        return time.perf_counter()
    

    def _get_handedness_from_result(self, handedness_list, hand_idx):
        """Extract handedness from MediaPipe Tasks API result structure"""
        if not handedness_list or hand_idx >= len(handedness_list):
            return "Right", 0.5
        
        try:
            categories = handedness_list[hand_idx]
            if categories and len(categories) > 0:
                category = categories[0]
                
                if hasattr(category, 'category_name'):
                    label = category.category_name
                
                score = category.score if hasattr(category, 'score') else 0.5
                return label, score
        except Exception as e:
            if self.debug_mode:
                print(f"Error getting handedness: {e}")
        
        return "Right", 0.5

    #TODO: OPTIMIZE
    def _process_detection_result(self, result: vision.HandLandmarkerResult, frame_id: int):
        """Process a single detection result"""

        if not result.hand_landmarks:
            self.miss_counter += 1
            return
        
        self.successful_detections += 1
        detected_hands = []

        # Get hands
        for hand_idx in range(len(result.hand_landmarks)):
            # Use helper method to get handedness
            handedness, handedness_confidence = self._get_handedness_from_result(
                result.handedness, hand_idx
            )

            landmarks = result.hand_landmarks[hand_idx]
            world_landmarks = None

            if result.hand_world_landmarks and hand_idx < len(result.hand_world_landmarks):
                world_landmarks = result.hand_world_landmarks[hand_idx]

            # Convert to numpy arrays
            frame_shape = (480, 640, 3)  # Default
            for frame_data in self.frame_buffer:
                if frame_data['id'] == frame_id:
                    frame_shape = frame_data['shape']
                    break
            
            landmarks_np = self._landmarks_to_numpy(landmarks, frame_shape)

            if world_landmarks:
                world_landmarks_np = self._world_landmarks_to_numpy(world_landmarks)
            else:
                # If no world landmarks, use normalized landmarks as fallback
                world_landmarks_np = self._normalized_landmarks_to_world(landmarks)
        
            hand_data = HandData(
                landmarks=landmarks_np,
                world_landmarks=world_landmarks_np,
                handedness=handedness,
                handedness_confidence=handedness_confidence,
                wrist_position=(landmarks_np[0, 0], landmarks_np[0, 1]),
                palm_centroid=self._calculate_palm_centroid(landmarks_np),
                timestamp=time.perf_counter(),
                id=self.hand_id_counter,
                frame_id=frame_id
            )

            self.hand_id_counter += 1
            detected_hands.append(hand_data)

        # Update hand tracking
        self._update_hand_tracking(detected_hands)
        
        self.miss_counter = 0


    def _normalized_landmarks_to_world(self, normalized_landmarks) -> np.ndarray:
        """
        Convert normalized landmarks to approximate world coordinates.
        This is a fallback when world landmarks aren't available.
        """
        array = np.zeros((21, 3))
        
        for i, landmark in enumerate(normalized_landmarks):
            # Approximate world coordinates by scaling normalized coordinates
            # This is an approximation - real world landmarks would be in meters
            array[i] = [
                landmark.x * 0.2 - 0.1,  # Roughly -0.1 to 0.1 meters
                landmark.y * 0.2 - 0.1,
                landmark.z * 0.1  # Depth scaling
            ]
            
        return array
    

    def _predict_missing_hands(self):
        """
        Use history to predict hand positions when detection is delayed.
        This maintains smooth tracking even with occasional missed frames.
        """
        

        if self.left_hand and len(self.left_hand_history) > 1:
            last = self.left_hand_history[-1]
            second_last = self.left_hand_history[-2]
            
            # Create predicted hand with linear extrapolation
            predicted_landmarks = last.landmarks + (last.landmarks - second_last.landmarks) * 0.5
            predicted_world_landmarks = last.world_landmarks + (last.world_landmarks - second_last.world_landmarks) * 0.5
            
            self.left_hand = HandData(
                landmarks=predicted_landmarks,
                world_landmarks=predicted_world_landmarks,
                handedness=last.handedness,
                handedness_confidence=last.handedness_confidence * 0.8,  # Reduced confidence
                wrist_position=(predicted_landmarks[0, 0], predicted_landmarks[0, 1]),
                palm_centroid=self._calculate_palm_centroid(predicted_landmarks),
                timestamp=time.perf_counter(),
                id=last.id,
                frame_id=-1  # Mark as predicted frame
            )
        
        # Similar prediction for right hand
        if self.right_hand and len(self.right_hand_history) > 1:
            last = self.right_hand_history[-1]
            second_last = self.right_hand_history[-2]
            
            predicted_landmarks = last.landmarks + (last.landmarks - second_last.landmarks) * 0.5
            predicted_world_landmarks = last.world_landmarks + (last.world_landmarks - second_last.world_landmarks) * 0.5
            
            self.right_hand = HandData(
                landmarks=predicted_landmarks,
                world_landmarks=predicted_world_landmarks,
                handedness=last.handedness,
                handedness_confidence=last.handedness_confidence * 0.8,
                wrist_position=(predicted_landmarks[0, 0], predicted_landmarks[0, 1]),
                palm_centroid=self._calculate_palm_centroid(predicted_landmarks),
                timestamp=time.perf_counter(),
                id=last.id,
                frame_id=-1
            )

    
    def _update_hand_tracking(self, detected_hands: List[HandData]):
        """Update hand tracking with new detections"""

        # Classify hands by handedness
        new_left = None
        new_right = None

        for hand in detected_hands:
            if hand.is_left:
                new_left = hand
            elif hand.is_right:
                new_right = hand

        # Handle ambiguous cases (2 hands but only one classified)
        if len(detected_hands) == 2:
            if new_left and not new_right:
                new_right = next((h for h in detected_hands if h.id != new_left.id), None)
                if new_right:
                    new_right.handedness = "Right"
            elif new_right and not new_left:
                new_left = next((h for h in detected_hands if h.id != new_right.id), None)
                if new_left:
                    new_left.handedness = "Left"
                # Update with continuity tracking
        if new_left:
            new_left.id = self._assign_hand_id(new_left, self.left_hand, self.left_hand_history)
            self.left_hand = new_left
            self.left_hand_history.append(new_left)
        
        if new_right:
            new_right.id = self._assign_hand_id(new_right, self.right_hand, self.right_hand_history)
            self.right_hand = new_right
            self.right_hand_history.append(new_right)
        

    def _assign_hand_id(self, new_hand: HandData, old_hand: Optional[HandData], history: deque) -> int:
        """Assign persistent ID based on position continuity"""
        if not old_hand:
            return new_hand.id
        
        # Calculate distance to previous position
        distance = np.sqrt(
            (new_hand.wrist_position[0] - old_hand.wrist_position[0])**2 +
            (new_hand.wrist_position[1] - old_hand.wrist_position[1])**2
        )
        
        # Use old ID if close enough (continuity)
        if distance < self.config.get("hand_tracking_distance_threshold", 100):  # pixels threshold
            return old_hand.id
        
        return new_hand.id
    

    def _calculate_palm_centroid(self, landmarks: np.ndarray) -> Tuple[float, float]:
        """Calculate palm center from MCP joints"""

        indices = [self.LANDMARK_INDICES[name] for name in 
                  ['INDEX_FINGER_MCP', 'MIDDLE_FINGER_MCP', 'PINKY_MCP']]
        
        points = landmarks[indices]
        return (np.mean(points[:, 0]), np.mean(points[:, 1]))


    def _landmarks_to_numpy(self, landmarks, frame_shape) -> np.ndarray:
        """Convert landmarks to numpy array"""

        height, width, _ = frame_shape
        array = np.zeros((21, 3))
        
        for i, landmark in enumerate(landmarks):
            array[i] = [landmark.x * width, landmark.y * height, landmark.z]
            
        return array
    

    def _world_landmarks_to_numpy(self, world_landmarks) -> np.ndarray:
        """Convert world landmarks to numpy array"""

        array = np.zeros((21, 3))
        
        for i, landmark in enumerate(world_landmarks):
            array[i] = [landmark.x, landmark.y, landmark.z]
            
        return array
    

    def _calculate_palm_orientation(self, world_landmarks: np.ndarray, handedness: str) -> Dict[str, Any]:
        """
        Calculate detailed palm orientation using multiple methods.
        
        Returns dict with:
        - normal: normalized palm normal vector
        - up: bool if palm is facing up
        - confidence: confidence of orientation
        - method: which method was used
        - angles: angles relative to camera
        """
        # Method 1: Plane normal using wrist, index MCP, pinky MCP
        wrist = world_landmarks[0]
        index_mcp = world_landmarks[5]
        pinky_mcp = world_landmarks[17]
        
        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist
        normal = np.cross(v1, v2)
        
        # Method 2: Alternative using palm center and finger bases
        # More robust for some orientations
        palm_center = np.mean(world_landmarks[[0, 5, 9, 13, 17]], axis=0)
        index_mcp_alt = world_landmarks[5]
        pinky_mcp_alt = world_landmarks[17]
        
        v1_alt = index_mcp_alt - palm_center
        v2_alt = pinky_mcp_alt - palm_center
        normal_alt = np.cross(v1_alt, v2_alt)
        
        # Average the two normals for better stability
        normal_combined = (normal + normal_alt) / 2
        
        # Normalize
        norm = np.linalg.norm(normal_combined)
        if norm > 0:
            normal_combined = normal_combined / norm
        else:
            normal_combined = np.array([0, 0, 1])  # Default
        
        # Determine if palm is facing up
        is_palm_up = self._determine_palm_up(normal_combined, handedness)
        
        # Calculate confidence based on normal magnitude
        normal_magnitude = np.linalg.norm(normal) * np.linalg.norm(normal_alt)
        confidence = min(1.0, normal_magnitude * 5)  # Scale to 0-1
        
        # Calculate angles relative to camera
        # Camera looks along negative z-axis
        camera_vector = np.array([0, 0, -1])
        
        # Angle between palm normal and camera direction
        dot_product = np.dot(normal_combined, camera_vector)
        angle_to_camera = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
        
        # Angle in horizontal plane (for side view detection)
        horizontal_normal = normal_combined.copy()
        horizontal_normal[1] = 0  # Remove vertical component
        if np.linalg.norm(horizontal_normal) > 0:
            horizontal_normal = horizontal_normal / np.linalg.norm(horizontal_normal)
            dot_horizontal = np.dot(horizontal_normal, np.array([0, 0, -1]))
            horizontal_angle = np.degrees(np.arccos(np.clip(dot_horizontal, -1.0, 1.0)))
        else:
            horizontal_angle = 0
        
        return {
            'normal': normal_combined,
            'up': is_palm_up,
            'confidence': confidence,
            'method': 'combined',
            'angles': {
                'to_camera': angle_to_camera,
                'horizontal': horizontal_angle
            }
        }
    
    
    def _determine_palm_up(self, normal: np.ndarray, handedness: str) -> bool:
        """
        Determine if palm is facing up using multiple heuristics.
        """
        handedness = handedness.lower()
        
        # Heuristic 1: Check z-component (camera-facing)
        # Negative z = pointing toward camera = palm up
        if normal[2] < -0.4:
            return True
        
        # Heuristic 2: Check y-component (overhead view)
        if handedness == "right":
            # Right hand: palm up = normal points downward
            if normal[1] < -0.3:
                return True
        else:
            # Left hand: palm up = normal points upward
            if normal[1] > 0.3:
                return True
        
        # Heuristic 3: Combined check for angled views
        # For angled views, we look at the projection onto vertical plane
        vertical_component = abs(normal[1])
        toward_camera_component = -normal[2]  # Negative z = toward camera
        
        # If hand is mostly vertical AND facing camera
        if vertical_component > 0.5 and toward_camera_component > 0.3:
            # Check handedness-specific direction
            if handedness == "right" and normal[1] < 0:
                return True
            elif handedness == "left" and normal[1] > 0:
                return True
        
        return False
    

    def get_palm_orientation_stats(self, hand: Optional[HandData]) -> Dict[str, Any]:
        """Get detailed palm orientation statistics"""
        if hand is None or hand.world_landmarks is None:
            return {
                'up': False,
                'confidence': 0.0,
                'normal': [0, 0, 0],
                'angles': {'to_camera': 0, 'horizontal': 0}
            }
        
        orientation = self._calculate_palm_orientation(
            hand.world_landmarks, 
            hand.handedness
        )
        
        return {
            'up': orientation['up'],
            'confidence': orientation['confidence'],
            'normal': orientation['normal'].tolist(),
            'angles': orientation['angles'],
            'handedness': hand.handedness
        }
    

    def get_detector_status(self) -> Dict[str, Any]:
        """Get current detector status for debugging"""
        return {
            "left_hand_detected": self.left_hand is not None,
            "right_hand_detected": self.right_hand is not None,
            "queue_size": self.result_queue.qsize(),
            "frame_buffer_size": len(self.frame_buffer),
            "miss_counter": self.miss_counter,
            "total_frames": self.total_frames,
            "successful_detections": self.successful_detections,
            "detection_rate": (self.successful_detections / self.total_frames * 100) 
                             if self.total_frames > 0 else 0,
            "avg_processing_time": np.mean(self.processing_times) if self.processing_times else 0,
            "avg_detection_latency": np.mean(self.detection_latencies) if self.detection_latencies else 0
        }
    
    
    def cleanup(self):
        """Clean up resources"""
        
        if hasattr(self, 'landmarker'):
            self.landmarker.close()