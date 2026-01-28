"""
Main loop with hand detection and normalization.
Clean display focusing only on essential information.
"""

import os
import yaml
import time
import cv2

from camera.camera_setup import Camera
from camera.frame_tracker import FrameTracker
from camera.hand_detector import HandDetector
from camera.normalizer import HandNormalizer

def load_camera_config():
    with open('config/camera_config.yaml') as file:
        return yaml.safe_load(file)

def load_mediapipe_config():
    with open('config/mediapipe_config.yaml') as file:
        return yaml.safe_load(file)

def create_minimal_display(frame, stats, frame_shape, left_hand=None, right_hand=None):
    """Create a clean, minimal display with only essential information"""
    
    height, width = frame_shape[:2]
    
    # Create mirrored copy for display (mirrored AFTER hand detection)
    display_frame = cv2.flip(frame, 1).copy()
    
    # Draw minimal status at top
    status_y = 30
    
    # FPS counter (top right)
    cv2.putText(display_frame, f"FPS: {stats.get('fps', 0):.1f}", 
                (width - 150, status_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    status_y += 30
    
    # Hand detection status
    left_status = f"L: {left_hand.handedness_confidence:.2f}" if left_hand is not None else "L:"
    right_status = f"R: {right_hand.handedness_confidence:.2f}" if right_hand is not None else "R:"    
    left_color = (100, 200, 255) if stats.get('left_detected', False) else (100, 100, 100)
    right_color = (255, 100, 200) if stats.get('right_detected', False) else (100, 100, 100)
    
    cv2.putText(display_frame, left_status, 
                (20, status_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, left_color, 2)
    cv2.putText(display_frame, right_status, 
                (100, status_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, right_color, 2)
    
    # Reference status
    ref_y = status_y + 30
    if stats.get('ref_established', False):
        cv2.putText(display_frame, "REF: ACTIVE", 
                    (20, ref_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(display_frame, "REF: INACTIVE", 
                    (20, ref_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
    
    # Draw hand positions on MIRRORED display frame
    if left_hand is not None:
        # Get wrist position from ORIGINAL coordinates, then mirror for display
        wrist_x_orig = int(left_hand.wrist_position[0])
        wrist_y = int(left_hand.wrist_position[1])
        
        # Mirror the x-coordinate for display
        wrist_x_display = width - wrist_x_orig - 1
        
        # Draw circle at mirrored wrist position
        cv2.circle(display_frame, (wrist_x_display, wrist_y), 25, (100, 200, 255), -1)
        cv2.circle(display_frame, (wrist_x_display, wrist_y), 25, (255, 255, 255), 2)
        
        # Show normalized value if available
        if stats.get('left_normalized') is not None:
            norm_val = stats['left_normalized']
            norm_text = f"{norm_val:+.2f}"
            text_color = (255, 255, 255) if abs(norm_val) < 0.5 else (255, 255, 0)
            cv2.putText(display_frame, norm_text, 
                        (wrist_x_display - 25, wrist_y - 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    
    if right_hand is not None:
        # Get wrist position from ORIGINAL coordinates, then mirror for display
        wrist_x_orig = int(right_hand.wrist_position[0])
        wrist_y = int(right_hand.wrist_position[1])
        
        # Mirror the x-coordinate for display
        wrist_x_display = width - wrist_x_orig - 1
        
        # Draw circle at mirrored wrist position
        cv2.circle(display_frame, (wrist_x_display, wrist_y), 25, (255, 100, 200), -1)
        cv2.circle(display_frame, (wrist_x_display, wrist_y), 25, (255, 255, 255), 2)
        
        # Show normalized value if available
        if stats.get('right_normalized') is not None:
            norm_val = stats['right_normalized']
            norm_text = f"{norm_val:+.2f}"
            text_color = (255, 255, 255) if abs(norm_val) < 0.5 else (255, 255, 0)
            cv2.putText(display_frame, norm_text, 
                        (wrist_x_display - 25, wrist_y - 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    
    # Draw torso reference if established
    if stats.get('ref_established', False):
        torso_y = int(stats.get('torso_center_y', 0.5) * height)
        
        # Draw torso center line
        cv2.line(display_frame, (0, torso_y), (width, torso_y), (0, 255, 255), 2)
        
        # Draw simple scale markers on the right
        scale_height = int(stats.get('torso_height', 0.3) * height)
        
        for i in range(-1, 2):
            marker_y = torso_y + int(i * scale_height / 2)
            cv2.line(display_frame, (width - 30, marker_y), (width - 10, marker_y), 
                     (200, 200, 200), 1)
            if i != 0:
                cv2.putText(display_frame, f"{i/2:.1f}", 
                            (width - 60, marker_y + 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Add simple instruction at bottom
    if not stats.get('ref_established', False):
        cv2.putText(display_frame, "Show both wrists to calibrate", 
                    (width // 2 - 150, height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
    
    return display_frame

def main():
    # Load configurations
    camera_config = load_camera_config()
    mediapipe_config = load_mediapipe_config()
    
    print("=" * 50)
    print("HAND TRACKING SYSTEM")
    print("=" * 50)
    
    # Initialize all modules
    print("\n1. Initializing Camera...")
    camera = Camera(camera_config["camera"])
    try:
        camera.initialize()
        print("   ✓ Camera ready")
        
    except Exception as e:
        print(f"   ✗ Camera initialization failed: {e}")
        return
    
    print("\n2. Initializing Frame Tracker...")
    tracker = FrameTracker(buffer_size=camera_config["frame_processing"]["circular_buffer_size"])
    
    print("\n3. Initializing Hand Detector...")
    try:
        detector = HandDetector(mediapipe_config)
        print("   ✓ Hand tracking ready")
    except Exception as e:
        print(f"   ✗ Hand detector initialization failed: {e}")
        camera.release()
        return
    
    print("\n4. Initializing Position Normalizer...")
    normalizer = HandNormalizer(mediapipe_config.get("normalization", {}))
    
    print("\n" + "=" * 50)
    print("SYSTEM READY")
    print("=" * 50)
    print("\nINSTRUCTIONS:")
    print("- Show both wrists to calibrate")
    print("- Blue = Left hand, Pink = Right hand")
    print("- Numbers show normalized position")
    print("- 0 = torso center, + = above, - = below")
    print("- Press 'Q' to exit")
    print("\n" + "=" * 50)
    
    frame_count = 0
    last_stats_time = time.time()
    tracking_start_time = time.time()
    calibration_start_time = None
    last_left_hand = None
    last_right_hand = None
    
    while True:
        # Capture frame
        ret, frame = camera.cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            break
        
        # Track frame timing
        tracker.capture_frame(frame)
        
        # Detect hands (on ORIGINAL non-mirrored frame)
        try:
            left_hand, right_hand = detector.process_frame(frame)
            # Store hand data for display
            if left_hand is not None:
                last_left_hand = left_hand
            if right_hand is not None:
                last_right_hand = right_hand
        except Exception as e:
            left_hand, right_hand = None, None
        
        # Normalize positions
        normalization_result = normalizer.process_hands(
            left_hand, right_hand, frame.shape
        )
        
        # Track calibration time
        if normalization_result['reference']['established'] and calibration_start_time is None:
            calibration_start_time = time.time()
            print(f"\n✅ Reference established in {calibration_start_time - tracking_start_time:.1f}s")
            print("   Normalized: 0 = torso, + = above, - = below")
        
        # Collect statistics
        try:
            tracker_stats = tracker.get_stats()
        except:
            tracker_stats = {"fps": 0, "drops": 0}
        
        try:
            detector_stats = detector.get_detector_status()
        except:
            detector_stats = {"left_hand_detected": False, "right_hand_detected": False}
        
        try:
            norm_stats = normalizer.get_normalization_stats()
        except:
            norm_stats = {"reference_established": False}
        
        # Prepare consolidated stats
        display_stats = {
            'fps': tracker_stats.get('fps', 0),
            'left_detected': detector_stats.get('left_hand_detected', False),
            'right_detected': detector_stats.get('right_hand_detected', False),
            'ref_established': norm_stats.get('reference_established', False),
        }
        
        # Add normalized values if available
        if normalization_result['reference']['established']:
            display_stats['torso_center_y'] = normalization_result['reference']['torso_center_y']
            display_stats['torso_height'] = normalization_result['reference']['torso_height']
            
            if normalization_result.get('left') and normalization_result['left']['detected']:
                display_stats['left_normalized'] = normalization_result['left'].get('smoothed', 0)
            if normalization_result.get('right') and normalization_result['right']['detected']:
                display_stats['right_normalized'] = normalization_result['right'].get('smoothed', 0)
        
        display_frame = create_minimal_display(frame, display_stats, frame.shape, 
                                              last_left_hand, last_right_hand)
        
        # Show frame
        cv2.imshow("Hand Tracking", display_frame)
        
        # Print periodic updates
        current_time = time.time()
        if current_time - last_stats_time > 2.0:
            print(f"\nFrame {frame_count}: ", end="")
            if display_stats['ref_established']:
                left_val = display_stats.get('left_normalized', 0)
                right_val = display_stats.get('right_normalized', 0)
                print(f"L:{left_val:+.2f} R:{right_val:+.2f} | ", end="")
            print(f"FPS: {display_stats['fps']:.1f}")
            last_stats_time = current_time
        
        frame_count += 1
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n" + "=" * 50)
            print("SHUTDOWN")
            break
    
    # Cleanup
    print("\nCleaning up...")
    camera.release()
    detector.cleanup()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 50)
    print("SESSION SUMMARY")
    print("=" * 50)
    
    print(f"\nTotal Frames: {frame_count}")
    print(f"Final FPS: {tracker_stats.get('fps', 0):.1f}")
    print(f"Reference Established: {norm_stats.get('reference_established', False)}")
    
    print("\n✅ SYSTEM SHUTDOWN COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()