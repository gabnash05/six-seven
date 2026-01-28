"""
Test script for vertical position normalization and temporal smoothing.
"""

import yaml
import cv2
import time
import sys
import numpy as np
from datetime import datetime
import json
import os

# Import your modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import your modules
try:
    from camera.camera_setup import Camera
    from camera.hand_detector import HandDetector
except ImportError as e:
    print(f"Import Error: {e}")
    print("Make sure you're running from the project root directory!")
    print("Or check that your modules are in src/camera/")
    sys.exit(1)

# Try to import normalizer if it exists
try:
    from camera.normalizer import HandNormalizer
    normalizer_available = True
except ImportError:
    print("Warning: HandNormalizer not found. Will create a simple test version.")
    normalizer_available = False

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

def load_camera_config():
    """Load camera configuration"""
    with open('config/camera_config.yaml') as file:
        return yaml.safe_load(file)

def load_mediapipe_config():
    """Load MediaPipe configuration"""
    with open('config/mediapipe_config.yaml') as file:
        return yaml.safe_load(file)

def visualize_normalization(frame, normalization_result, frame_shape):
    """Visualize normalized positions on frame"""
    height, width = frame_shape[:2]
    
    # Draw torso reference lines
    if normalization_result["reference"]["established"]:
        torso_center_y = int(normalization_result["reference"]["torso_center_y"] * height)
        
        # Torso centerline
        cv2.line(frame, (0, torso_center_y), (width, torso_center_y), (0, 255, 255), 2)
        
        # Shoulder line (torso_center_y - torso_height/3)
        shoulder_y = int((normalization_result["reference"]["torso_center_y"] - 
                         normalization_result["reference"]["torso_height"] / 3) * height)
        cv2.line(frame, (0, shoulder_y), (width, shoulder_y), (0, 200, 100), 1)
        
        # Waist line (torso_center_y + torso_height/3)
        waist_y = int((normalization_result["reference"]["torso_center_y"] + 
                      normalization_result["reference"]["torso_height"] / 3) * height)
        cv2.line(frame, (0, waist_y), (width, waist_y), (0, 100, 200), 1)
    
    # Draw hand positions
    for hand_name in ["left", "right"]:
        hand_data = normalization_result.get(hand_name)
        if hand_data and hand_data["detected"]:
            # Get raw position
            if hand_data["raw"] is not None:
                # Convert normalized position back to pixel coordinates for visualization
                if normalization_result["reference"]["established"]:
                    torso_center_y = normalization_result["reference"]["torso_center_y"]
                    torso_height = normalization_result["reference"]["torso_height"]
                    norm_y = torso_center_y + (hand_data["raw"] * torso_height)
                    pixel_y = int(norm_y * height)
                    
                    # Draw marker
                    color = (255, 0, 0) if hand_name == "left" else (0, 0, 255)
                    cv2.circle(frame, (width // 2, pixel_y), 10, color, 2)
                    
                    # Add text
                    text = f"{hand_name}: {hand_data['raw']:.3f}"
                    cv2.putText(frame, text, (10, 30 if hand_name == "left" else 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Add reference status
    status_text = f"Reference: {'ESTABLISHED' if normalization_result['reference']['established'] else 'WAITING'}"
    cv2.putText(frame, status_text, (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

def main():
    """Main test function"""
    # Load configurations
    camera_config = load_camera_config()
    mediapipe_config = load_mediapipe_config()
    
    print("Initializing camera...")
    camera = Camera(camera_config["camera"])
    camera.initialize()
    
    print("Initializing hand detector...")
    detector = HandDetector(mediapipe_config)
    
    print("Initializing hand normalizer...")
    normalizer = HandNormalizer(mediapipe_config.get("normalization", {}))
    
    print("\nStarting normalization test...")
    print("Move your hands to establish torso reference")
    print("Press 'q' to quit")
    print("Press 'r' to reset normalization")
    print("Press 's' to toggle smoothing")
    
    frame_count = 0
    smoothing_enabled = True
    
    # Create output directory for logs
    os.makedirs("data/logs", exist_ok=True)
    
    while True:
        ret, frame = camera.cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        frame = cv2.flip(frame, 1)
        
        # Detect hands
        left_hand, right_hand = detector.process_frame(frame)
        
        # Normalize positions
        normalization_result = normalizer.process_hands(
            left_hand, right_hand, frame.shape
        )
        
        # Visualize
        frame = visualize_normalization(frame, normalization_result, frame.shape)
        
        # Display frame
        cv2.imshow("Hand Normalization Test", frame)
        
        # Display stats occasionally
        if frame_count % 30 == 0:
            stats = normalizer.get_normalization_stats()
            print(f"\nFrame {frame_count}:")
            print(f"  Reference established: {stats['reference_established']}")
            
            if stats.get('left'):
                print(f"  Left: mean={stats['left']['mean']:.3f}, std={stats['left']['std']:.3f}")
            if stats.get('right'):
                print(f"  Right: mean={stats['right']['mean']:.3f}, std={stats['right']['std']:.3f}")
        
        frame_count += 1
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            normalizer.reset()
            print("Normalization reset")
        elif key == ord('s'):
            smoothing_enabled = not smoothing_enabled
            normalizer.smoothing_enabled = smoothing_enabled
            print(f"Smoothing {'enabled' if smoothing_enabled else 'disabled'}")
        elif key == ord('d'):
            # Save debug data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/logs/normalization_debug_{timestamp}.json"
            
            debug_data = {
                "timestamp": timestamp,
                "normalization_stats": normalizer.get_normalization_stats(),
                "detector_status": detector.get_detector_status(),
                "normalization_result": normalization_result
            }
            
            with open(filename, 'w') as f:
                json.dump(debug_data, f, indent=2, cls=NumpyEncoder)            
            print(f"Debug data saved to {filename}")
    
    # Cleanup
    camera.release()
    detector.cleanup()
    cv2.destroyAllWindows()
    
    # Print final statistics
    print("\n" + "="*50)
    print("NORMALIZATION TEST COMPLETE")
    print("="*50)
    
    final_stats = normalizer.get_normalization_stats()
    print(f"Total frames processed: {frame_count}")
    print(f"Reference established: {final_stats['reference_established']}")
    buffer_size = mediapipe_config.get("normalization", {}).get("temporal_buffer_size", 7)
    print(f"Temporal buffer fill: {final_stats['temporal_buffer']['left']['size']}/{buffer_size}")    
    if final_stats.get('left'):
        print(f"\nLeft hand statistics:")
        print(f"  Mean position: {final_stats['left']['mean']:.3f}")
        print(f"  Standard deviation: {final_stats['left']['std']:.3f}")
        print(f"  Range: {final_stats['left']['range']:.3f}")
    
    if final_stats.get('right'):
        print(f"\nRight hand statistics:")
        print(f"  Mean position: {final_stats['right']['mean']:.3f}")
        print(f"  Standard deviation: {final_stats['right']['std']:.3f}")
        print(f"  Range: {final_stats['right']['range']:.3f}")

if __name__ == "__main__":
    main()