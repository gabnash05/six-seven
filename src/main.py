"""
Main loop with validation.
Captures frames, tracks timing, verifies FPS performance.
"""

import os
import yaml
import json
import time
import cv2
from datetime import datetime

from camera.camera_setup import Camera
from camera.frame_tracker import FrameTracker

def load_config():
    """Load camera configuration from YAML file"""

    with open('config/camera_config.yaml') as file:
        return yaml.safe_load(file)

def validate_performance(tracker, config):
    """Check if performance meets requirements"""

    stats = tracker.get_stats()
    
    print("\n" + "="*50)
    print("PERFORMANCE VALIDATION")
    print("="*50)
    
    fps_ok = stats["fps"] >= config["validation"]['pass_criteria']["min_fps"]
    variance_ok = stats["std_interval_ms"] <= config["validation"]['pass_criteria']["max_variance_ms"]
    drops_ok = stats["drops"] <= config["validation"]['pass_criteria']["max_drops"]
    
    print(f"FPS: {stats['fps']:.1f} / {config['camera']['fps']['target']} target → {'PASS' if fps_ok else 'FAIL'}")    
    print(f"Variance: {stats['std_interval_ms']:.2f}ms → {'PASS' if variance_ok else 'FAIL'}")
    print(f"Frame drops: {stats['drops']} → {'PASS' if drops_ok else 'FAIL'}")
    
    return fps_ok and variance_ok and drops_ok

def main():
    config = load_config()

    print("Initializing camera...")
    camera = Camera(config["camera"])   
    camera.initialize()

    tracker = FrameTracker(buffer_size=config["frame_processing"]["circular_buffer_size"])

    print(f"Running validation for {config['validation']['benchmark_duration_frames']} frames...")
    print("Press 'q' to quit early")

    frame_count = 0
    running = True

    while running and frame_count < config['validation']['benchmark_duration_frames']:
        t_start = time.perf_counter()

        ret, frame = camera.cap.read()
        t_capture = time.perf_counter()

        if not ret:
            print("Failed to capture frame")
            break

        is_valid = tracker.capture_frame(frame)

        if not is_valid:
            print(f"Frame drop detected! Total: {tracker.dropped_frames}")
        
        # Update display occasionally
        if frame_count % 100 == 0:
            stats = tracker.get_stats()
            print(f"Frame {frame_count}: {stats['fps']:.1f} FPS, Drops: {stats['drops']}")
        
        t_end = time.perf_counter()

        capture_time = (t_capture - t_start) * 1000
        processing_time = (t_end - t_capture) * 1000
        total_time = (t_end - t_start) * 1000

        if config['logging']['performance_log']['enabled'] and frame_count % 50 == 0:
            file_path = config['logging']['performance_log']['file_location']
    
            log_directory = os.path.dirname(file_path)
            if log_directory: 
                os.makedirs(log_directory, exist_ok=True)
            
            with open(file_path, "a") as file:
                log_entry = {
                    "frame": frame_count,
                    "capture_ms": capture_time,
                    "processing_ms": processing_time,
                    "total_ms": total_time,
                    "timestamp": datetime.now().isoformat()
                }
                file.write(json.dumps(log_entry) + "\n")

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

    print("\n" + "="*50)
    print("VALIDATION COMPLETE")
    print("="*50)
    
    passed = validate_performance(tracker, config)
    
    if passed:
        print("\n✅ Phase 1 PASSED: Camera ready for gesture detection")
    else:
        print("\n❌ Phase 1 FAILED: Camera performance insufficient")
        print("Check configuration and try lower resolution/FPS")
    
    camera.release()
    cv2.destroyAllWindows()

    if config['logging']['validation_log']['enabled']:
        final_stats = tracker.get_stats()
        
        file_path = config['logging']['validation_log']['file_location']
    
        log_directory = os.path.dirname(file_path)
        if log_directory:
            os.makedirs(log_directory, exist_ok=True)
        
        with open(file_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "passed": passed,
                "config": config,
                "stats": final_stats
            }, f, indent=2)
        
        print(f"\nResults saved to {file_path}")

if __name__ == "__main__":
    main()
