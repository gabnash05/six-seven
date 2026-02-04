"""
Simple finite state machine for hand gesture game.
Tracks dominance and manages scoring transitions.
"""

from enum import Enum
import time
import cv2
from typing import Optional, Dict, Any


class State(Enum):
    IDLE = 0
    LEFT_DOMINANT_LOCKED = 1
    RIGHT_DOMINANT_LOCKED = 2
    COOLDOWN = 3


class GameState:
    """Minimal game state machine for hand gesture detection"""
    
    def __init__(self, config=None):
        # Basic configuration with sensible defaults
        self.config = config
        
        self.current_state = State.IDLE
        self.score = 0
        self.last_score_time = 0.0
        self.state_entry_time = time.perf_counter()
        self.consecutive_dominant_frames = 0
        self.last_valid_dominance = None
        
        self.valid_frame_count = 0
        
        self.debug = self.config.get('debug_mode', False)
    

    def update(self, normalization_result: Dict[str, Any]) -> None:
        """
        Update game state based on normalized hand positions.
        
        Args:
            normalization_result: Output from HandNormalizer.process_hands()
        """
        
        current_time = time.perf_counter()
        time_in_state = current_time - self.state_entry_time
        
        # Check if both hands are detected
        left_data = normalization_result.get('left')
        right_data = normalization_result.get('right')
        
        both_detected = (
            left_data is not None and left_data['detected'] and
            right_data is not None and right_data['detected']
        )
        
        if not both_detected:
            # Reset to IDLE if hands are missing
            if self.current_state != State.IDLE:
                self._change_state(State.IDLE, current_time)
            return
        
        # Get smoothed positions for dominance calculation
        left_y = left_data.get('smoothed') if left_data else None
        right_y = right_data.get('smoothed') if right_data else None
        
        if left_y is None or right_y is None:
            return
        
        dominance_diff = left_y - right_y
        threshold = self.config['dominance_threshold']
        hysteresis = self.config['hysteresis']
        
        if self.current_state == State.IDLE:
            self._handle_idle_state(dominance_diff, threshold, current_time)
            
        elif self.current_state == State.LEFT_DOMINANT_LOCKED:
            self._handle_left_locked_state(dominance_diff, threshold, hysteresis, 
                                          time_in_state, current_time)
            
        elif self.current_state == State.RIGHT_DOMINANT_LOCKED:
            self._handle_right_locked_state(dominance_diff, threshold, hysteresis,
                                           time_in_state, current_time)
            
        elif self.current_state == State.COOLDOWN:
            self._handle_cooldown_state(time_in_state, current_time)
    

    def _handle_idle_state(self, dominance_diff: float, threshold: float, current_time: float) -> None:
        """Handle IDLE state transitions"""
        
        if dominance_diff > threshold:
            self.consecutive_dominant_frames += 1
            if self.consecutive_dominant_frames >= self.config['min_lock_frames']:
                self._change_state(State.LEFT_DOMINANT_LOCKED, current_time)
                self.last_valid_dominance = "LEFT"
                
        elif dominance_diff < -threshold:
            self.consecutive_dominant_frames += 1
            if self.consecutive_dominant_frames >= self.config['min_lock_frames']:
                self._change_state(State.RIGHT_DOMINANT_LOCKED, current_time)
                self.last_valid_dominance = "RIGHT"
                
        else:
            self.consecutive_dominant_frames = 0
    

    def _handle_left_locked_state(self, dominance_diff: float, threshold: float, hysteresis: float, time_in_state: float, current_time: float) -> None:
        """Handle LEFT_DOMINANT_LOCKED state transitions"""
        
        if dominance_diff < -(threshold + hysteresis):
            self.consecutive_dominant_frames += 1
            if self.consecutive_dominant_frames >= self.config['min_lock_frames']:
                self.score += 1
                self.last_score_time = current_time
                self.last_valid_dominance = "RIGHT"
                self._change_state(State.COOLDOWN, current_time)
                
        elif dominance_diff > threshold:
            self.consecutive_dominant_frames = 0
            
        else:
            return
    

    def _handle_right_locked_state(self, dominance_diff: float, threshold: float, hysteresis: float, time_in_state: float, current_time: float) -> None:
        """Handle RIGHT_DOMINANT_LOCKED state transitions"""
        
        if dominance_diff > (threshold + hysteresis):
            self.consecutive_dominant_frames += 1
            if self.consecutive_dominant_frames >= self.config['min_lock_frames']:
                self.score += 1
                self.last_score_time = current_time
                self.last_valid_dominance = "LEFT"
                self._change_state(State.COOLDOWN, current_time)
                
        elif dominance_diff < -threshold:
            self.consecutive_dominant_frames = 0
            
        else:
            return
    
    
    def _handle_cooldown_state(self, time_in_state: float, current_time: float) -> None:
        """Handle COOLDOWN state transitions"""
        
        if time_in_state >= self.config['cooldown_duration']:
            self._change_state(State.IDLE, current_time)
    

    def _change_state(self, new_state: State, timestamp: float) -> None:
        """Change state and update tracking variables"""
        if new_state == self.current_state:
            return
            
        old_state = self.current_state
        self.current_state = new_state
        self.state_entry_time = timestamp
        self.consecutive_dominant_frames = 0
        
        if self.debug:
            print(f"State change: {old_state.name} -> {new_state.name}")
    

    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information"""
        current_time = time.perf_counter()
        
        return {
            'state': self.current_state.name,
            'score': self.score,
            'time_in_state': current_time - self.state_entry_time,
            'last_valid_dominance': self.last_valid_dominance,
            'last_score_time': self.last_score_time,
            'in_cooldown': self.current_state == State.COOLDOWN
        }
    

    def reset(self) -> None:
        """Reset game to initial state"""
        self.current_state = State.IDLE
        self.score = 0
        self.last_score_time = 0.0
        self.state_entry_time = time.perf_counter()
        self.consecutive_dominant_frames = 0
        self.last_valid_dominance = None
        self.valid_frame_count = 0


def create_game_visualization(frame, game_state, normalization_result, frame_shape):
    """
    Create visualization overlay for the game.
    
    Args:
        frame: Original frame
        game_state: GameState instance
        normalization_result: From HandNormalizer
        frame_shape: Frame dimensions
    """
    height, width = frame_shape[:2]
    
    # Create mirrored copy for display
    display_frame = cv2.flip(frame, 1).copy()
    
    # State indicator (top center)
    state = game_state.current_state
    state_colors = {
        State.IDLE: (200, 200, 200),
        State.LEFT_DOMINANT_LOCKED: (100, 200, 255),  # Blue
        State.RIGHT_DOMINANT_LOCKED: (255, 100, 200), # Pink
        State.COOLDOWN: (255, 255, 100)               # Yellow
    }
    
    state_color = state_colors.get(state, (255, 255, 255))
    state_text = f"STATE: {state.name.replace('_', ' ')}"
    
    cv2.putText(display_frame, state_text,
                (width // 2 - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
    
    # Score display (top right)
    score_text = f"SCORE: {game_state.score}"
    cv2.putText(display_frame, score_text,
                (width - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Show dominance indicator
    left_data = normalization_result.get('left')
    right_data = normalization_result.get('right')
    
    if left_data and right_data and left_data['detected'] and right_data['detected']:
        bar_width = 40
        bar_spacing = 20
        
        left_y = int((0.5 - left_data.get('smoothed', 0) * 0.4) * height)
        left_height = int(abs(left_data.get('smoothed', 0)) * height * 0.4)
        left_x = width // 2 - bar_width - bar_spacing
        
        cv2.rectangle(display_frame,
                     (left_x, left_y),
                     (left_x + bar_width, height // 2),
                     (100, 200, 255), -1)
        
        right_y = int((0.5 - right_data.get('smoothed', 0) * 0.4) * height)
        right_height = int(abs(right_data.get('smoothed', 0)) * height * 0.4)
        right_x = width // 2 + bar_spacing
        
        cv2.rectangle(display_frame,
                     (right_x, right_y),
                     (right_x + bar_width, height // 2),
                     (255, 100, 200), -1)
        
        # Draw center line
        cv2.line(display_frame, (0, height // 2), (width, height // 2),
                 (0, 255, 255), 2)
        
        # Show delta value
        if left_data.get('smoothed') is not None and right_data.get('smoothed') is not None:
            delta = left_data['smoothed'] - right_data['smoothed']
            delta_text = f"Δ: {delta:+.2f}"
            cv2.putText(display_frame, delta_text,
                       (width // 2 - 40, height // 2 - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Cooldown timer
    if state == State.COOLDOWN:
        cooldown_left = max(0, game_state.config['cooldown_duration'] - 
                           (time.perf_counter() - game_state.state_entry_time))
        timer_text = f"Next in: {cooldown_left:.1f}s"
        cv2.putText(display_frame, timer_text,
                   (width // 2 - 70, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 100), 2)
    
    return display_frame