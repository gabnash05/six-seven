"""
Utility functions for testing the hand detector.
"""

import cv2
import numpy as np

def draw_hand_landmarks(frame, hand_data, color=(0, 255, 0), thickness=2):
    """
    Draw hand landmarks and connections on a frame.
    
    Args:
        frame: Input frame
        hand_data: HandData object
        color: BGR color tuple
        thickness: Line thickness
    
    Returns:
        Frame with drawn landmarks
    """
    if hand_data is None:
        return frame.copy()
    
    annotated = frame.copy()    
    # Define hand connections (same as MediaPipe)
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),           # Index finger
        (0, 9), (9, 10), (10, 11), (11, 12),      # Middle finger
        (0, 13), (13, 14), (14, 15), (15, 16),    # Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20),    # Pinky
        (5, 9), (9, 13), (13, 17)                 # Palm
    ]
    
    # Draw landmarks as circles
    for i in range(21):
        x = int(hand_data.landmarks[i, 0])
        y = int(hand_data.landmarks[i, 1])
        cv2.circle(annotated, (x, y), 3, color, -1)
    
    # Draw connections
    for start_idx, end_idx in HAND_CONNECTIONS:
        start_x = int(hand_data.landmarks[start_idx, 0])
        start_y = int(hand_data.landmarks[start_idx, 1])
        end_x = int(hand_data.landmarks[end_idx, 0])
        end_y = int(hand_data.landmarks[end_idx, 1])
        cv2.line(annotated, (start_x, start_y), (end_x, end_y), color, thickness)
    
    # Draw wrist with larger circle
    wrist_x = int(hand_data.wrist_position[0])
    wrist_y = int(hand_data.wrist_position[1])
    cv2.circle(annotated, (wrist_x, wrist_y), 8, color, -1)
    
    # Label with hand type and confidence
    label = f"{hand_data.handedness[0]}:{hand_data.handedness_confidence:.2f}"
    cv2.putText(annotated, label, (wrist_x - 20, wrist_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Draw palm centroid
    palm_x = int(hand_data.palm_centroid[0])
    palm_y = int(hand_data.palm_centroid[1])
    cv2.circle(annotated, (palm_x, palm_y), 5, (255, 255, 0), -1)
    
    return annotated

def create_performance_overlay(frame, stats_dict, position=(10, 30)):
    """
    Add performance statistics overlay to a frame.
    
    Args:
        frame: Input frame
        stats_dict: Dictionary of statistics
        position: (x, y) position for first line
    
    Returns:
        Frame with overlay
    """
    annotated = frame.copy()
    x, y = position
    line_height = 25
    
    for i, (key, value) in enumerate(stats_dict.items()):
        if isinstance(value, float):
            text = f"{key}: {value:.2f}"
        else:
            text = f"{key}: {value}"
        
        cv2.putText(annotated, text, (x, y + i * line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return annotated

def save_test_results(results, filename="test_results.json"):
    """Save test results to JSON file"""
    import json
    import os
    from datetime import datetime
    
    results_to_save = results.copy()
    results_to_save['timestamp'] = datetime.now().isoformat()
    
    os.makedirs('test_results', exist_ok=True)
    with open(f'test_results/{filename}', 'w') as f:
        json.dump(results_to_save, f, indent=2)    
        
    print(f"Results saved to test_results/{filename}")