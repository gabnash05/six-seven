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
from game.game_state import GameState, create_game_visualization

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
    
    print("\n2. Initializing Frame Tracker...")
    frame_tracker = FrameTracker()
    
    print("\n3. Initializing Hand Detector...")
    detector = HandDetector(mediapipe_config)
    
    print("\n4. Initializing Normalizer...")
    normalizer = HandNormalizer(mediapipe_config.get("normalization", {}))
    
    print("\n5. Initializing Game State...")
    game_state = GameState(game_config)
    
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
        start_time = time.perf_counter()
        
        ret, frame = camera.cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            break

        frame_tracker.capture_frame(frame)

        left_hand, right_hand = detector.process_frame(frame)

        if game_config.get("debug_mode"):
            process_end = time.perf_counter()
            process_time = process_end - start_time

        normalization_result = normalizer.process_hands(
            left_hand, right_hand, frame.shape
        )

        if game_config.get("debug_mode"):
            normalization_end = time.perf_counter()
            normalization_time = normalization_end - process_end

        game_state.update(normalization_result)

        game_frame = create_game_visualization(
            frame, game_state, normalization_result, frame.shape
        )

        if game_config.get("debug_mode"):
            game_update_time = time.perf_counter() - normalization_end
        
        # Display
        cv2.imshow("Hand Gesture Game", game_frame)


        if game_config.get("debug_mode"):
            print("-------------")
            print(process_time)
            print(normalization_time)
            print(game_update_time)
            print(time.perf_counter() - start_time)
            print("-------------")
        
        # Periodic status updates
        if frame_count % 60 == 0 and game_config.get("debug_mode"):  # Every ~1 second at 60fps
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