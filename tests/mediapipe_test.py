"""
Palm Orientation Test for MediaPipe Hand Detector.
Real-time visualization showing hand positions and orientation tracking.
"""

import cv2
import time
import sys
import os
import numpy as np
from datetime import datetime

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import from your modules
try:
    from camera.hand_detector import HandDetector
    from camera.camera_setup import Camera
except ImportError as e:
    print(f"Import Error: {e}")
    print("Make sure you're running from the project root directory!")
    print("Or check that your modules are in src/camera/")
    sys.exit(1)

def load_config():
    """Load configuration with all required keys"""
    return {
        'camera': {
            'device_index': 0,
            'resolution': {'width': 640, 'height': 480},
            'fps': {'target': 30},
            'controls': {
                'auto_exposure': True,
                'auto_focus': True,
                'auto_white_balance': True
            },
            'optimization': {'buffer_size': 2}
        },
        'hand_detector': {
            'model_path': 'models/hand_landmarker.task',
            'num_hands': 2,
            'min_hand_detection_confidence': 0.5,
            'min_hand_presence_confidence': 0.4,
            'min_tracking_confidence': 0.4,
            'hand_tracking_distance_threshold': 100
        }
    }


def draw_hand_landmarks(frame: np.ndarray, hand, is_left: bool = True) -> np.ndarray:
    """Draw detailed hand landmarks and connections"""
    if hand is None:
        return frame
    
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    
    # Colors for left (blue) and right (red) hands - CORRECTED
    if is_left:
        color = (255, 150, 0)  # Blue-orange (BGR)
        hand_label = "L"
    else:
        color = (0, 100, 255)  # Red (BGR)
        hand_label = "R"
    
    # Draw all landmarks as circles with consistent sizing
    for i in range(21):
        x = int(hand.landmarks[i, 0])
        y = int(hand.landmarks[i, 1])
        
        # Different sizes for key landmarks
        if i == 0:  # Wrist
            radius = 6
        elif i in [4, 8, 12, 16, 20]:  # Fingertips
            radius = 4
        else:  # Other landmarks
            radius = 3
        
        cv2.circle(annotated, (x, y), radius, color, -1)
    
    # Draw connections between landmarks
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),           # Index finger
        (0, 9), (9, 10), (10, 11), (11, 12),      # Middle finger
        (0, 13), (13, 14), (14, 15), (15, 16),    # Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20),    # Pinky
        (5, 9), (9, 13), (13, 17)                 # Palm connections
    ]
    
    for start_idx, end_idx in HAND_CONNECTIONS:
        start_x = int(hand.landmarks[start_idx, 0])
        start_y = int(hand.landmarks[start_idx, 1])
        end_x = int(hand.landmarks[end_idx, 0])
        end_y = int(hand.landmarks[end_idx, 1])
        
        cv2.line(annotated, (start_x, start_y), (end_x, end_y), 
                color, 2)
    
    # Draw wrist with label
    wrist_x = int(hand.wrist_position[0])
    wrist_y = int(hand.wrist_position[1])
    
    cv2.circle(annotated, (wrist_x, wrist_y), 8, (255, 255, 255), 2)
    cv2.circle(annotated, (wrist_x, wrist_y), 6, color, -1)
    
    # Draw hand label near wrist (small and clean)
    cv2.putText(annotated, hand_label,
               (wrist_x - 5, wrist_y + 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Draw palm centroid
    palm_x = int(hand.palm_centroid[0])
    palm_y = int(hand.palm_centroid[1])
    cv2.circle(annotated, (palm_x, palm_y), 4, (0, 255, 255), -1)  # Yellow
    
    return annotated


def draw_palm_orientation(frame: np.ndarray, hand, detector, is_left: bool = True) -> np.ndarray:
    """Draw palm orientation visualization - SIMPLIFIED"""
    if hand is None:
        return frame
    
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    
    # Get orientation stats
    orientation = detector.get_palm_orientation_stats(hand)
    
    # Position orientation text near hand (top center of hand bounding box)
    min_x = int(np.min(hand.landmarks[:, 0]))
    max_x = int(np.max(hand.landmarks[:, 0]))
    min_y = int(np.min(hand.landmarks[:, 1]))
    
    display_x = (min_x + max_x) // 2
    display_y = max(min_y - 20, 20)  # Above hand, but not off screen
    
    # Orientation status and color - CORRECTED
    if orientation['up']:
        status_text = "↑ UP"
        color = (0, 255, 0)  # Green
    else:
        status_text = "↓ DOWN"
        color = (0, 0, 255)  # Red
    
    # Confidence level
    confidence = orientation['confidence']
    conf_text = f"{confidence:.0%}"
    
    # Draw small background for text
    text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    
    bg_width = max(text_size[0], conf_size[0]) + 10
    bg_height = text_size[1] + conf_size[1] + 15
    
    bg_x1 = display_x - bg_width // 2
    bg_y1 = display_y - text_size[1] - 5
    bg_x2 = bg_x1 + bg_width
    bg_y2 = bg_y1 + bg_height
    
    # Ensure within bounds
    bg_x1 = max(bg_x1, 5)
    bg_y1 = max(bg_y1, 5)
    bg_x2 = min(bg_x2, width - 5)
    bg_y2 = min(bg_y2, height - 5)
    
    # Draw semi-transparent background
    overlay = annotated.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    annotated = cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0)
    
    # Draw border
    cv2.rectangle(annotated, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 1)
    
    # Draw orientation text
    text_x = display_x - text_size[0] // 2
    text_y = display_y
    cv2.putText(annotated, status_text,
               (text_x, text_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw confidence text below
    conf_x = display_x - conf_size[0] // 2
    conf_y = text_y + conf_size[1] + 8
    cv2.putText(annotated, conf_text,
               (conf_x, conf_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    return annotated


def draw_stats_overlay(frame: np.ndarray, stats: dict) -> np.ndarray:
    """Draw minimal performance statistics - SIMPLIFIED"""
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    
    # Small, clean stats panel in top-left corner
    panel_width = 120
    panel_height = 80
    
    # Position at top-left with small margin
    panel_x1 = 10
    panel_y1 = 10
    panel_x2 = panel_x1 + panel_width
    panel_y2 = panel_y1 + panel_height
    
    # Semi-transparent background
    overlay = annotated.copy()
    cv2.rectangle(overlay,
                 (panel_x1, panel_y1),
                 (panel_x2, panel_y2),
                 (0, 0, 0), -1)
    
    alpha = 0.7
    annotated = cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0)
    
    # Draw border
    cv2.rectangle(annotated,
                 (panel_x1, panel_y1),
                 (panel_x2, panel_y2),
                 (255, 255, 255), 1)
    
    # Title (small)
    cv2.putText(annotated, "STATS",
               (panel_x1 + 5, panel_y1 + 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Draw separator
    cv2.line(annotated,
             (panel_x1 + 5, panel_y1 + 20),
             (panel_x2 - 5, panel_y1 + 20),
             (100, 100, 100), 1)
    
    # Only show essential stats
    y_offset = panel_y1 + 35
    
    # FPS - color coded
    fps = stats.get('fps', 0)
    fps_color = (0, 255, 0) if fps > 20 else (0, 255, 255) if fps > 10 else (0, 0, 255)
    cv2.putText(annotated, f"FPS: {fps:.1f}",
               (panel_x1 + 5, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, fps_color, 1)
    
    # Hands detected (compact)
    left_status = "L:✓" if stats.get('left_detected', False) else "L:✗"
    right_status = "R:✓" if stats.get('right_detected', False) else "R:✗"
    
    left_color = (200, 150, 0) if "✓" in left_status else (100, 100, 100)
    right_color = (0, 100, 255) if "✓" in right_status else (100, 100, 100)
    
    cv2.putText(annotated, left_status,
               (panel_x1 + 5, y_offset + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, left_color, 1)
    
    cv2.putText(annotated, right_status,
               (panel_x1 + 5, y_offset + 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, right_color, 1)
    
    return annotated


def draw_simple_controls(frame: np.ndarray) -> np.ndarray:
    """Draw minimal controls info in bottom-left corner"""
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    
    # Small panel in bottom-left corner
    lines = [
        "Q = Quit",
        "S = Screenshot",
        "R = Reset"
    ]
    
    # Calculate needed height
    line_height = 20
    panel_height = len(lines) * line_height + 10
    
    panel_x1 = 10
    panel_y1 = height - panel_height - 10
    panel_x2 = 120
    panel_y2 = height - 10
    
    # Semi-transparent background
    overlay = annotated.copy()
    cv2.rectangle(overlay,
                 (panel_x1, panel_y1),
                 (panel_x2, panel_y2),
                 (0, 0, 0), -1)
    
    annotated = cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0)
    
    # Draw controls
    for i, line in enumerate(lines):
        y_pos = panel_y1 + 20 + (i * line_height)
        cv2.putText(annotated, line,
                   (panel_x1 + 5, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    
    return annotated


def test_palm_orientation():
    """Real-time test showing hand positions and orientation tracking"""
    print("\n" + "="*60)
    print("REAL-TIME HAND POSITION & ORIENTATION TEST")
    print("="*60)
    
    config = load_config()
    
    print("\nThis test shows:")
    print("1. Full hand skeleton with landmarks")
    print("2. Palm orientation (UP/DOWN)")
    print("3. Real-time tracking of both hands")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("  'r' - Reset statistics")
    print("\nColor coding:")
    print("  Orange = Left hand")
    print("  Red = Right hand")
    print("  Green = Palm UP")
    print("  Red = Palm DOWN")
    
    try:
        # Initialize
        print("\nInitializing camera...")
        camera = Camera(config['camera'])
        camera.initialize()
        
        print("Initializing hand detector...")
        detector = HandDetector(config.get('hand_detector', {}))
        
        # Create output directory
        os.makedirs('tests/palm_test_output', exist_ok=True)
        
        # Statistics
        start_time = time.perf_counter()
        frame_count = 0
        last_fps_time = time.perf_counter()
        fps_frames = 0
        current_fps = 0
        
        # Palm orientation counters
        left_up_frames = 0
        left_down_frames = 0
        right_up_frames = 0
        right_down_frames = 0
        
        # Performance tracking
        detection_frames = 0
        
        print("\nStarting real-time tracking...")
        print("Show your hands to the camera!")
        print("Window will show hand positions and orientation.")
        
        # Main test loop
        while True:
            # Capture frame
            ret, frame = camera.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            frame_count += 1
            fps_frames += 1
            
            # Process with hand detector
            left_hand, right_hand = detector.process_frame(frame)
            
            # Start with original frame
            display_frame = frame.copy()
            
            # Draw hand landmarks for both hands
            if left_hand:
                detection_frames += 1
                display_frame = draw_hand_landmarks(display_frame, left_hand, is_left=True)
                
                # Track palm orientation
                orientation = detector.get_palm_orientation_stats(left_hand)

                if orientation['up']:
                    left_up_frames += 1
                else:
                    left_down_frames += 1
                
                # Draw palm orientation info
                display_frame = draw_palm_orientation(display_frame, left_hand, detector, is_left=True)
            
            if right_hand:
                detection_frames += 1
                display_frame = draw_hand_landmarks(display_frame, right_hand, is_left=False)
                
                # Track palm orientation
                orientation = detector.get_palm_orientation_stats(right_hand)

                if orientation['up']:
                    right_up_frames += 1
                else:
                    right_down_frames += 1
                    
                # Draw palm orientation info
                display_frame = draw_palm_orientation(display_frame, right_hand, detector, is_left=False)
            
            # Calculate FPS
            current_time = time.perf_counter()
            if current_time - last_fps_time >= 1.0:
                current_fps = fps_frames / (current_time - last_fps_time)
                fps_frames = 0
                last_fps_time = current_time
            
            # Prepare minimal statistics
            stats = {
                'fps': current_fps,
                'left_detected': left_hand is not None,
                'right_detected': right_hand is not None
            }
            
            # Draw minimal stats overlay
            display_frame = draw_stats_overlay(display_frame, stats)
            
            # Draw simple controls
            display_frame = draw_simple_controls(display_frame)
            
            # Show frame
            cv2.imshow("Hand Tracking & Orientation", display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nStopping test...")
                break
            elif key == ord('r'):
                print("Resetting statistics...")
                left_up_frames = 0
                left_down_frames = 0
                right_up_frames = 0
                right_down_frames = 0
                detection_frames = 0
                start_time = time.perf_counter()
                frame_count = 0
                fps_frames = 0
                current_fps = 0
            elif key == ord('s'):
                # Save screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = f"tests/palm_test_output/screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"✓ Saved screenshot to {filename}")
        
        # Calculate final statistics
        elapsed = time.perf_counter() - start_time
        
        print("\n" + "="*60)
        print("TEST COMPLETE")
        print("="*60)
        
        print(f"\nTest duration: {elapsed:.1f} seconds")
        print(f"Total frames: {frame_count}")
        print(f"Average FPS: {frame_count/elapsed:.1f}" if elapsed > 0 else "Average FPS: N/A")
        
        print("\nPALM ORIENTATION RESULTS:")
        print("-"*40)
        
        # Left hand results
        left_total = left_up_frames + left_down_frames
        if left_total > 0:
            left_up_percent = (left_up_frames / left_total) * 100
            print(f"Left Hand:")
            print(f"  Palm UP:  {left_up_frames} frames ({left_up_percent:.1f}%)")
            print(f"  Palm DOWN: {left_down_frames} frames ({100-left_up_percent:.1f}%)")
        else:
            print("Left Hand: No data (hand not detected)")
        
        # Right hand results
        right_total = right_up_frames + right_down_frames
        if right_total > 0:
            right_up_percent = (right_up_frames / right_total) * 100
            print(f"\nRight Hand:")
            print(f"  Palm UP:  {right_up_frames} frames ({right_up_percent:.1f}%)")
            print(f"  Palm DOWN: {right_down_frames} frames ({100-right_up_percent:.1f}%)")
        else:
            print("\nRight Hand: No data (hand not detected)")
        
        # Cleanup
        camera.release()
        cv2.destroyAllWindows()
        
        # Save results
        os.makedirs('tests/palm_test_results', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"tests/palm_test_results/results_{timestamp}.txt"
        
        with open(result_file, 'w') as f:
            f.write("Real-Time Hand Tracking Test Results\n")
            f.write("="*60 + "\n\n")
            f.write(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test duration: {elapsed:.1f} seconds\n")
            f.write(f"Total frames: {frame_count}\n")
            f.write(f"Average FPS: {frame_count/elapsed:.1f}\n\n" if elapsed > 0 else "Average FPS: N/A\n\n")            
            f.write("LEFT HAND RESULTS:\n")
            f.write("-"*40 + "\n")
            if left_total > 0:
                left_up_percent = (left_up_frames / left_total) * 100
                f.write(f"Palm UP:  {left_up_frames} frames ({left_up_percent:.1f}%)\n")
                f.write(f"Palm DOWN: {left_down_frames} frames ({100-left_up_percent:.1f}%)\n")
            else:
                f.write("No data (hand not detected)\n")
            
            f.write("\nRIGHT HAND RESULTS:\n")
            f.write("-"*40 + "\n")
            if right_total > 0:
                right_up_percent = (right_up_frames / right_total) * 100
                f.write(f"Palm UP:  {right_up_frames} frames ({right_up_percent:.1f}%)\n")
                f.write(f"Palm DOWN: {right_down_frames} frames ({100-right_up_percent:.1f}%)\n")
            else:
                f.write("No data (hand not detected)\n")
        
        print(f"\nDetailed results saved to: {result_file}")
        
        # Final assessment
        print("\n" + "="*60)
        print("ASSESSMENT:")
        print("="*60)
        
        if left_total > 10 or right_total > 10:
            print("✅ Real-time tracking test completed successfully!")
        else:
            print("⚠️  Limited data collected.")
            print("   Make sure both hands are visible to the camera!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to cleanup
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        return False


def main():
    """Main function - runs the enhanced hand tracking test"""
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('tests/palm_test_output', exist_ok=True)
    os.makedirs('tests/palm_test_results', exist_ok=True)
    
    print("Real-Time Hand Position & Orientation Test")
    print("="*60)
    print("This test shows real-time tracking of:")
    print("1. Hand positions (full skeleton)")
    print("2. Palm orientation (UP/DOWN)")
    print("3. Wrist and palm center positions")
    print("\nFeatures:")
    print("  • Orange skeleton = Left hand")
    print("  • Red skeleton = Right hand")
    print("  • White wrist circle")
    print("  • Yellow palm center")
    print("  • Orientation arrows (↑UP/↓DOWN)")
    
    if len(sys.argv) > 1 and sys.argv[1] == 'help':
        print("\nUsage: python mediapipe_test.py")
        print("\nControls:")
        print("  'q' - Quit test")
        print("  's' - Save screenshot")
        print("  'r' - Reset statistics")
        return
    
    print("\nStarting test... (Press 'q' to quit)")
    test_palm_orientation()

if __name__ == "__main__":
    main()