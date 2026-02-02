"""
Main loop with hand detection, normalization, and simple game state.
"""

import os
import yaml
import time
import cv2

from camera.camera_setup import Camera
from camera.frame_tracker import FrameTracker
from camera.hand_detector import HandDetector
from camera.normalizer import HandNormalizer
from game.game_state import SimpleGameState, create_game_visualization

def load_camera_config():
    with open('config/camera_config.yaml') as file:
        return yaml.safe_load(file)

def load_mediapipe_config():
    with open('config/mediapipe_config.yaml') as file:
        return yaml.safe_load(file)

def load_game_config():
    """Load game configuration if it exists, otherwise use defaults"""
    try:
        with open('config/game_config.yaml') as file:
            config = yaml.safe_load(file)
            return config.get('game', {})
    except FileNotFoundError:
        print("Game config not found, using defaults")
        return {}

def main():
    # Load configurations
    camera_config = load_camera_config()
    mediapipe_config = load_mediapipe_config()
    game_config = load_game_config()
    
    print("=" * 50)
    print("HAND GESTURE GAME")
    print("=" * 50)
    
    # Initialize modules
    print("\n1. Initializing Camera...")
    camera = Camera(camera_config["camera"])
    try:
        camera.initialize()
        print("   ✓ Camera ready")
    except Exception as e:
        print(f"   ✗ Camera initialization failed: {e}")
        return
    
    print("\n2. Initializing Hand Detector...")
    try:
        detector = HandDetector(mediapipe_config)
        print("   ✓ Hand tracking ready")
    except Exception as e:
        print(f"   ✗ Hand detector failed: {e}")
        camera.release()
        return
    
    print("\n3. Initializing Normalizer...")
    normalizer = HandNormalizer(mediapipe_config.get("normalization", {}))
    
    print("\n4. Initializing Game State...")
    game_state = SimpleGameState(game_config)
    
    print("\n" + "=" * 50)
    print("GAME READY")
    print("=" * 50)
    print("\nINSTRUCTIONS:")
    print("- Show both wrists to calibrate")
    print("- Raise LEFT hand to lock BLUE state")
    print("- Raise RIGHT hand to lock PINK state")
    print("- Switch hands to score points!")
    print("- Press Q to quit")
    print("\n" + "=" * 50)
    
    frame_count = 0
    last_stats_time = time.time()
    
    while True:
        # Capture frame
        ret, frame = camera.cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            break
        
        # Detect hands
        left_hand, right_hand = detector.process_frame(frame)
        
        # Normalize positions
        normalization_result = normalizer.process_hands(
            left_hand, right_hand, frame.shape
        )
        
        # Update game state
        game_state.update(normalization_result)
        
        # Create game visualization
        game_frame = create_game_visualization(
            frame, game_state, normalization_result, frame.shape
        )
        
        # Display
        cv2.imshow("Hand Gesture Game", game_frame)
        
        # Periodic status updates
        if frame_count % 60 == 0:  # Every ~1 second at 60fps
            state_info = game_state.get_state_info()
            print(f"\nFrame {frame_count}:")
            print(f"  State: {state_info['state']}")
            print(f"  Score: {state_info['score']}")
            
            left_data = normalization_result.get('left')
            right_data = normalization_result.get('right')
            if left_data and right_data and left_data['detected'] and right_data['detected']:
                delta = left_data.get('smoothed', 0) - right_data.get('smoothed', 0)
                print(f"  Δ: {delta:+.2f}")
        
        frame_count += 1
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n" + "=" * 50)
            print("GAME OVER")
            break
    
    # Cleanup
    print("\nFinal Score:", game_state.score)
    print("Cleaning up...")
    
    camera.release()
    detector.cleanup()
    cv2.destroyAllWindows()
    
    print("\n✅ GAME SESSION COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()