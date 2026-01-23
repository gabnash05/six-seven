import cv2
import time
import numpy as np

def diagnose_camera_problems():
    """Diagnose why camera is performing poorly"""
    print("\n" + "="*60)
    print("DIAGNOSTIC MODE")
    print("="*60)
    
    problems = []
    
    # Test 1: Basic camera access with default settings
    print("\n1. Testing basic camera access (no settings)...")
    cap_default = cv2.VideoCapture(0)
    if not cap_default.isOpened():
        problems.append("Camera 0 cannot be opened at all")
        print("  ✗ Camera 0 failed to open")
    else:
        default_fps = cap_default.get(cv2.CAP_PROP_FPS)
        default_w = cap_default.get(cv2.CAP_PROP_FRAME_WIDTH)
        default_h = cap_default.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"  ✓ Opens at {default_w}x{default_h} @ {default_fps:.1f} FPS")
        cap_default.release()
    
    # Test 2: Check if DirectShow is the issue
    print("\n2. Testing DirectShow specifically...")
    cap_dshow = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap_dshow.isOpened():
        # DirectShow often reports 0 FPS but still works
        dshow_fps = cap_dshow.get(cv2.CAP_PROP_FPS)
        print(f"  ✓ DirectShow opens (reports {dshow_fps:.1f} FPS)")
        if dshow_fps == 0:
            problems.append("DirectShow reports 0 FPS - this is normal but confusing")
    else:
        problems.append("DirectShow backend fails completely")
        print("  ✗ DirectShow fails")
    
    # Test 3: Try camera index 1
    print("\n3. Testing camera index 1 (might be different camera)...")
    cap1 = cv2.VideoCapture(1, cv2.CAP_MSMF)
    if cap1.isOpened():
        fps1 = cap1.get(cv2.CAP_PROP_FPS)
        w1 = cap1.get(cv2.CAP_PROP_FRAME_WIDTH)
        h1 = cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"  ✓ Camera 1 exists: {w1}x{h1} @ {fps1:.1f} FPS")
        cap1.release()
    else:
        print("  ✗ No camera at index 1")
    
    # Test 4: Check what settings actually work
    print("\n4. Testing achievable resolutions...")
    test_resolutions = [(320, 240), (640, 480), (800, 600), (1280, 720)]
    working_resolutions = []
    
    for w, h in test_resolutions:
        cap_test = cv2.VideoCapture(0, cv2.CAP_MSMF)
        if cap_test.isOpened():
            # Try to set the resolution
            success_w = cap_test.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            success_h = cap_test.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            
            actual_w = cap_test.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = cap_test.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            # Quick FPS test
            start = time.perf_counter()
            frames = 0
            for _ in range(10):
                ret, _ = cap_test.read()
                if ret:
                    frames += 1
            elapsed = time.perf_counter() - start
            fps = frames / elapsed if elapsed > 0 else 0
            
            if success_w and success_h:
                working_resolutions.append((actual_w, actual_h, fps))
                print(f"  ✓ {int(actual_w)}x{int(actual_h)}: {fps:.1f} FPS")
            else:
                print(f"  ✗ {w}x{h} not supported (got {actual_w}x{actual_h})")
            
            cap_test.release()
    
    # Analyze problems
    print("\n" + "="*60)
    print("DIAGNOSIS:")
    print("="*60)
    
    if problems:
        print("Detected issues:")
        for i, problem in enumerate(problems, 1):
            print(f"  {i}. {problem}")
    else:
        print("No obvious hardware/driver issues detected")
    
    print(f"\nWorking resolutions:")
    for w, h, fps in working_resolutions:
        print(f"  {int(w)}x{int(h)}: {fps:.1f} FPS")
    
    # Recommendation
    if working_resolutions:
        best_w, best_h, best_fps = max(working_resolutions, key=lambda x: x[2])
        print(f"\nRECOMMENDATION: Use {int(best_w)}x{int(best_h)}")
        return int(best_w), int(best_h), "MSMF"
    
    return 640, 480, "MSMF"  # Default fallback

def setup_camera_adaptive():
    """Adaptive camera setup based on diagnostics"""
    print("=" * 60)
    print("ADAPTIVE CAMERA SETUP")
    print("=" * 60)
    
    # First run diagnostics
    best_w, best_h, best_backend = diagnose_camera_problems()
    
    print(f"\nUsing configuration:")
    print(f"  Backend: {best_backend}")
    print(f"  Resolution: {best_w}x{best_h}")
    
    # Map backend name to code
    backend_map = {
        "MSMF": cv2.CAP_MSMF,
        "DSHOW": cv2.CAP_DSHOW,
        "ANY": cv2.CAP_ANY
    }
    backend_code = backend_map.get(best_backend, cv2.CAP_ANY)
    
    # Open camera
    cap = cv2.VideoCapture(0, backend_code)
    
    if not cap.isOpened():
        print("ERROR: Cannot open camera with recommended backend")
        print("Falling back to default...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("FATAL: Cannot open any camera")
            return None
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, best_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, best_h)
    
    # Try to set FPS - but don't fail if it doesn't work
    target_fps = 60 if best_w <= 640 else 30
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    
    # Only try manual controls if we have decent FPS already
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps > 20:
        try:
            # Try to disable auto-exposure
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            # Try to set exposure - but accept failure
            cap.set(cv2.CAP_PROP_EXPOSURE, -4)
        except:
            print("Note: Manual exposure controls not available")
    
    # Verify settings
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"\nCamera configured:")
    print(f"  Resolution: {int(actual_width)}x{int(actual_height)}")
    print(f"  FPS: {actual_fps:.1f}")
    print(f"  Backend: {cap.getBackendName()}")
    
    return cap

def benchmark_capture_smart(cap, num_frames=300):
    """
    Smarter benchmark that adapts based on performance
    """
    print(f"\nBenchmarking {num_frames} frames...")
    
    frame_times = []
    frame_sizes = []
    last_time = time.perf_counter()
    
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"ERROR: Frame capture failed at frame {i}")
            return 0, 0, 0, False
        
        # Measure time and size
        current_time = time.perf_counter()
        frame_time = (current_time - last_time) * 1000
        frame_times.append(frame_time)
        
        if frame is not None:
            frame_sizes.append(frame.shape)
        
        last_time = current_time
        
        # Show progress
        if i % 50 == 0:
            if i > 0:
                current_fps = 1000 / np.mean(frame_times[-50:])
                print(f"  Frame {i}/{num_frames}: {current_fps:.1f} FPS")
    
    # Calculate statistics
    if len(frame_times) < 10:
        print("ERROR: Not enough frames captured")
        return 0, 0, 0, False
    
    frame_times_array = np.array(frame_times[1:])
    
    avg_frame_time = np.mean(frame_times_array)
    avg_fps = 1000 / avg_frame_time
    std_dev_ms = np.std(frame_times_array)
    
    # Check frame consistency
    if frame_sizes:
        unique_sizes = set(frame_sizes)
        size_consistent = len(unique_sizes) == 1
    
    print(f"\nBenchmark Results:")
    print(f"  Average FPS: {avg_fps:.1f}")
    print(f"  Average frame time: {avg_frame_time:.2f} ms")
    print(f"  Standard deviation: {std_dev_ms:.2f} ms")
    
    if frame_sizes and not size_consistent:
        print(f"  ⚠️ Frame size inconsistent: {unique_sizes}")
    
    # Performance assessment
    print(f"\nPerformance Assessment:")
    
    if avg_fps >= 60:
        print(f"  ✓ Excellent: {avg_fps:.1f} FPS (ready for game)")
        status = "EXCELLENT"
    elif avg_fps >= 30:
        print(f"  ✓ Good: {avg_fps:.1f} FPS (playable)")
        status = "GOOD"
    elif avg_fps >= 15:
        print(f"  ⚠️ Marginal: {avg_fps:.1f} FPS (will be laggy)")
        status = "MARGINAL"
    else:
        print(f"  ✗ Poor: {avg_fps:.1f} FPS (unplayable)")
        status = "POOR"
    
    if std_dev_ms < 5:
        print(f"  ✓ Stable timing: {std_dev_ms:.2f} ms std dev")
    elif std_dev_ms < 15:
        print(f"  ⚠️ Moderate variance: {std_dev_ms:.2f} ms std dev")
    else:
        print(f"  ✗ High variance: {std_dev_ms:.2f} ms std dev")
    
    return avg_fps, std_dev_ms, avg_frame_time, True, status

def optimize_for_game(cap, initial_fps, status):
    """
    Try to optimize settings based on initial performance
    """
    print("\n" + "="*60)
    print("OPTIMIZATION ATTEMPT")
    print("="*60)
    
    if status == "POOR":
        print("Performance is too poor. Trying emergency measures...")
        
        # Release current camera
        cap.release()
        
        # Try absolute minimum settings
        print("\n1. Trying absolute minimum settings (320x240)...")
        cap_min = cv2.VideoCapture(0)
        cap_min.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap_min.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        
        # Quick test
        start = time.perf_counter()
        frames = 0
        for _ in range(30):
            ret, _ = cap_min.read()
            if ret:
                frames += 1
        elapsed = time.perf_counter() - start
        min_fps = frames / elapsed if elapsed > 0 else 0
        
        print(f"  Minimum config: {min_fps:.1f} FPS")
        
        if min_fps > 20:
            print("  ✓ Minimum config works")
            return cap_min
        else:
            print("  ✗ Even minimum config fails")
            cap_min.release()
            return None
    
    elif status == "MARGINAL":
        print("Trying to improve marginal performance...")
        
        # Get current resolution
        current_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        current_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        if current_w > 640:
            print(f"Reducing from {int(current_w)}x{int(current_h)} to 640x480...")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Test improvement
            fps_before = initial_fps
            avg_fps, _, _, _, _ = benchmark_capture_smart(cap, 100)
            
            if avg_fps > fps_before * 1.2:
                print(f"  ✓ Improved from {fps_before:.1f} to {avg_fps:.1f} FPS")
            else:
                print(f"  ✗ No improvement: {avg_fps:.1f} FPS")
    
    return cap

def main():
    """Main function with adaptive setup"""
    print("=" * 60)
    print("ADAPTIVE CAMERA SETUP FOR GESTURE GAME")
    print("=" * 60)
    
    # Step 1: Adaptive setup
    camera = setup_camera_adaptive()
    
    if camera is None:
        print("\nFailed to setup camera. Cannot proceed.")
        return
    
    # Step 2: Initial benchmark
    print("\n" + "="*60)
    print("INITIAL BENCHMARK")
    print("="*60)
    
    fps, std_dev, avg_time, success, status = benchmark_capture_smart(camera, 200)
    
    # Step 3: Optimization if needed
    if status in ["POOR", "MARGINAL"]:
        camera = optimize_for_game(camera, fps, status)
        if camera:
            # Re-benchmark after optimization
            print("\n" + "="*60)
            print("POST-OPTIMIZATION BENCHMARK")
            print("="*60)
            fps, std_dev, avg_time, success, status = benchmark_capture_smart(camera, 200)
    
    # Step 4: Final assessment
    print("\n" + "="*60)
    print("FINAL ASSESSMENT")
    print("="*60)
    
    if not camera:
        print("✗ CAMERA FAILED: Cannot achieve playable performance")
        print("\nTROUBLESHOOTING:")
        print("1. Try a different camera (external webcam)")
        print("2. Update camera drivers")
        print("3. Test on another computer")
        return
    
    final_w = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    final_h = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    final_fps = camera.get(cv2.CAP_PROP_FPS)
    
    print(f"Final configuration:")
    print(f"  Resolution: {int(final_w)}x{int(final_h)}")
    print(f"  Reported FPS: {final_fps:.1f}")
    print(f"  Measured FPS: {fps:.1f}")
    print(f"  Stability: {std_dev:.2f} ms std dev")
    
    print(f"\nGame readiness:")
    
    # MediaPipe needs about 30ms per frame on CPU
    mediapipe_time = 30  # ms
    total_frame_time = avg_time + mediapipe_time
    achievable_fps = 1000 / total_frame_time if total_frame_time > 0 else 0
    
    if achievable_fps >= 30:
        print(f"  ✓ READY: Estimated {achievable_fps:.1f} FPS with MediaPipe")
        print(f"  Proceed with game development")
    elif achievable_fps >= 20:
        print(f"  ⚠️ MARGINAL: Estimated {achievable_fps:.1f} FPS with MediaPipe")
        print(f"  Game will be laggy but might work")
    else:
        print(f"  ✗ NOT READY: Estimated {achievable_fps:.1f} FPS with MediaPipe")
        print(f"  Camera too slow for real-time gesture game")
    
    # Cleanup
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()