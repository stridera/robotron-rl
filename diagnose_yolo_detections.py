"""
Diagnose YOLO detector performance by comparing detections to ground truth.

This will help us understand why the YOLO+RL performance is so low.
"""
import numpy as np
from robotron import RobotronEnv
from wrappers import MultiDiscreteToDiscrete, FrameSkipWrapper
from yolo_detector_wrapper import YOLODetectorWrapper
import cv2

# Configuration
YOLO_MODEL = '../labelbox_to_yolo/runs/detect/train3/weights/best.pt'
NUM_FRAMES = 100

print("=" * 80)
print("YOLO DETECTOR DIAGNOSTIC")
print("=" * 80)
print()

# Create environment to get ground truth
print("Creating environment for ground truth...")
env_gt = RobotronEnv(
    level=3,  # Use level 3 (3 grunts) for testing
    lives=5,
    fps=0,
    config_path='progressive_curriculum.yaml',
    always_move=True,
    headless=True
)
env_gt = MultiDiscreteToDiscrete(env_gt)
env_gt = FrameSkipWrapper(env_gt, skip=4)

# Get a few frames and check what YOLO sees
print(f"Analyzing {NUM_FRAMES} frames...")
print("-" * 80)

# Load YOLO model
from ultralytics import YOLO
yolo_model = YOLO(YOLO_MODEL)

detection_counts = []
sprite_types_detected = {}

obs = env_gt.reset()
for frame_idx in range(NUM_FRAMES):
    # Get current frame - use the observation from the environment
    # The unwrapped env returns RGB frames
    raw_obs = env_gt.unwrapped.render()

    # Get ground truth sprite count from engine
    engine = env_gt.unwrapped.engine
    ground_truth_count = len(engine.all_group)

    # Run YOLO detection on the raw frame
    # YOLO expects RGB, robotron returns RGB
    results = yolo_model(raw_obs, conf=0.25, device='cuda', verbose=False)

    # Count detections
    detected_count = 0
    if results[0].boxes is not None:
        detected_count = len(results[0].boxes)

        # Track sprite types
        for box in results[0].boxes:
            class_id = int(box.cls[0].cpu().numpy())
            class_name = yolo_model.names[class_id]
            sprite_types_detected[class_name] = sprite_types_detected.get(class_name, 0) + 1

    detection_counts.append((ground_truth_count, detected_count))

    if frame_idx % 20 == 0:
        print(f"Frame {frame_idx:3d}: Ground truth={ground_truth_count:2d}, "
              f"YOLO detected={detected_count:2d}")

    # Take random action
    action = env_gt.action_space.sample()
    step_result = env_gt.step(action)

    # Handle both gym and gymnasium API (4 or 5 returns)
    if len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        done = terminated or truncated
    else:
        obs, reward, done, info = step_result

    if done:
        obs = env_gt.reset()

env_gt.close()

print("-" * 80)
print()
print("=" * 80)
print("DETECTION STATISTICS")
print("=" * 80)

gt_counts = [gt for gt, det in detection_counts]
det_counts = [det for gt, det in detection_counts]

print(f"Average ground truth sprites: {np.mean(gt_counts):.1f} ± {np.std(gt_counts):.1f}")
print(f"Average YOLO detections:      {np.mean(det_counts):.1f} ± {np.std(det_counts):.1f}")
print(f"Detection rate: {np.mean(det_counts) / np.mean(gt_counts) * 100:.1f}%")
print()

print("Sprite types detected:")
for sprite_type, count in sorted(sprite_types_detected.items(), key=lambda x: x[1], reverse=True):
    print(f"  {sprite_type:20s}: {count:4d} detections")
print()

print("=" * 80)
print("DIAGNOSIS")
print("=" * 80)

detection_rate = np.mean(det_counts) / np.mean(gt_counts)
if detection_rate < 0.5:
    print("❌ CRITICAL: YOLO is detecting < 50% of sprites!")
    print()
    print("Possible issues:")
    print("  1. Resolution mismatch: YOLO trained on different resolution than simulator")
    print("     - Simulator outputs 665x492, but YOLO was trained on Xbox frames")
    print("     - Xbox likely has different resolution/aspect ratio")
    print("  2. Visual appearance mismatch: Xbox graphics != simulator graphics")
    print("  3. Confidence threshold too high (try lowering from 0.25 to 0.15)")
    print()
    print("Recommendations:")
    print("  1. Check the resolution/size of Xbox training images")
    print("  2. Visualize YOLO predictions on simulator frames to see what it's missing")
    print("  3. Consider training YOLO on simulator frames instead")
    print("  4. Lower confidence threshold in wrapper")
elif detection_rate < 0.8:
    print("⚠️  WARNING: YOLO is detecting < 80% of sprites")
    print("Performance may be degraded. Consider:")
    print("  - Lowering confidence threshold")
    print("  - Training YOLO on simulator frames")
    print("  - Adding simulator frames to YOLO training data")
else:
    print("✅ YOLO detection rate looks good (>80%)")
    print()
    print("If RL performance is still poor, check:")
    print("  - Position accuracy (are sprite positions correct?)")
    print("  - Sprite type classification accuracy")
    print("  - VecNormalize stats compatibility")
