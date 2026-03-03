# Phase 2: Sprite Detector Training - Status

## Overview

Building a CNN sprite detector to bridge pixels → positions → policy pipeline.

**Goal:** Train a detector that can identify sprite positions from raw frames, enabling the position-based RL policy to work with pixel inputs instead of ground truth positions.

---

## Current Status

### ✅ Completed

1. **Position-Based RL Training (Phase 1.5)**
   - Trained on progressive curriculum (1 grunt → full game)
   - Result: 145 kills average, reaches level 11
   - Action diversity maintained (8/8 actions)
   - **Conclusion:** Position-based RL scales to realistic scenarios

2. **Data Collection Script**
   - Created `collect_detector_data.py`
   - Runs random policy to collect diverse gameplay
   - Saves frames (84×84 grayscale) + ground truth sprite positions
   - Tested successfully (500 frames in 13s)

3. **Detector Training Script**
   - Created `train_detector.py`
   - Grid-based CNN architecture (simpler than YOLO)
   - Multi-task loss: sprite presence + type + position offsets
   - Ready to train once data collection completes

### 🔄 In Progress

**Data Collection** (Running in background, PID: 446382)
- Target: 1000 episodes × 500 steps = 500k frames
- Progress: ~5/1000 episodes (started recently)
- Rate: ~13 seconds per episode
- **Estimated completion:** 3-4 hours from start
- Output: `detector_dataset.npz` (~1-2 GB)

Monitor progress:
```bash
tail -f data_collection.log
```

Check if complete:
```bash
ls -lh detector_dataset.npz
```

---

## Architecture Details

### Data Collection

**Environment:**
- Config: `progressive_curriculum.yaml` (diverse scenarios)
- Preprocessing: Grayscale + resize to 84×84
- Random policy for maximum diversity

**Output Format:**
- Frames: (N, 84, 84) uint8 numpy array
- Sprite data: List of N lists of (x, y, sprite_type) tuples
- Compressed .npz file

**Expected Statistics:**
- ~500k frames total
- ~5-7 sprites per frame average
- Sprite types: Grunt (25%), Bullet (55%), Player (18%), Others (2%)

### Detector Architecture

**Grid-Based Detection:**
```
Input: (1, 84, 84) grayscale frame
  ↓
Conv1: 32 filters, 3×3
  ↓
Conv2: 64 filters, 3×3, stride 2  (42×42)
  ↓
Conv3: 128 filters, 3×3, stride 2  (21×21)
  ↓
Conv4: 256 filters, 3×3
  ↓
AdaptivePool → 16×16 grid
  ↓
Prediction Head: 1×1 conv
  ↓
Output: (16, 16, 1+17+2) per-cell predictions
  - has_sprite: 1 (binary)
  - sprite_type: 17 (one-hot)
  - offset_x, offset_y: 2 (position within cell)
```

**Why grid-based?**
- Simpler than full object detection (YOLO/Faster R-CNN)
- Sufficient for Robotron (sprites are small, grid covers whole screen)
- Faster training (~50 epochs vs 100-200 for YOLO)
- Easier to debug and interpret

### Loss Function

**Multi-task loss:**
1. **Binary Cross-Entropy** for sprite presence
   - Predicts: Is there a sprite in this grid cell?

2. **Binary Cross-Entropy** for sprite type (masked)
   - Predicts: What type of sprite? (only where sprite exists)

3. **MSE** for position offset (masked)
   - Predicts: Exact position within cell (only where sprite exists)

**Total loss:** `loss = has_loss + type_loss + offset_loss`

---

## Next Steps (After Data Collection Completes)

### Step 1: Train Detector (~2-3 hours)

```bash
poetry run python train_detector.py \
    --dataset detector_dataset.npz \
    --output models/detector \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-3 \
    --device cuda
```

**Expected outcomes:**
- Best model saved to `models/detector/best_detector.pth`
- Checkpoints every 10 epochs
- Validation loss should decrease to ~0.2-0.3

**Success criteria:**
- Detection rate > 90% (finds 9/10 sprites)
- Position error < 10 pixels (accurate localization)
- Type accuracy > 85% (correctly identifies sprite types)

### Step 2: Evaluate Detector (~30 minutes)

Create evaluation script to:
1. Load test frames from dataset
2. Run detector predictions
3. Compare to ground truth positions
4. Calculate metrics:
   - Precision/Recall for sprite detection
   - Position error (mean absolute error in pixels)
   - Type classification accuracy
5. Visualize predictions on sample frames

**Script to create:** `eval_detector.py`

### Step 3: Create DetectorWrapper (~1 hour)

Replace `GroundTruthPositionWrapper` with detector-based version:

```python
class DetectorWrapper(gym.ObservationWrapper):
    """Use trained detector instead of ground truth positions."""

    def __init__(self, env, detector_model_path, max_sprites=20):
        super().__init__(env)
        self.detector = load_detector(detector_model_path)
        self.max_sprites = max_sprites
        # Same observation space as GroundTruthPositionWrapper
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(442,))

    def observation(self, obs):
        # obs: (84, 84, 4) pixel observation

        # Run detector
        detections = self.detector(obs)

        # Convert grid predictions to sprite list
        sprites = self._grid_to_sprites(detections)

        # Convert to same format as GroundTruthPositionWrapper
        position_features = self._sprites_to_features(sprites)

        return position_features
```

### Step 4: Test Detector + RL Pipeline (~1 hour)

```bash
# Test with trained position-based policy
poetry run python test_detector_rl.py \
    --detector models/detector/best_detector.pth \
    --policy models/6l9t4lpc/checkpoints/ppo_progressive_checkpoint_3000000_steps.zip \
    --episodes 10
```

**Success criteria:**
- Detector + RL achieves >80% of ground truth performance
- Target: 115+ kills (80% of 145)
- Action diversity maintained
- No catastrophic failures

### Step 5: (Optional) Add Temporal Tracking

If detector errors cause issues, add tracking across frames:

```python
class TemporalTracker:
    """Track sprites across frames to handle occlusions."""

    def update(self, detections):
        # Match new detections to existing tracks (Hungarian algorithm)
        # Predict positions of occluded sprites (velocity extrapolation)
        # Remove old tracks not seen for >10 frames
        return tracked_sprites
```

**Benefits:**
- Handles temporary occlusions (sprites overlap)
- Smooths jittery detections
- Reduces false negatives

---

## Timeline

### Completed (So Far)
- Phase 1: Position-based RL proof of concept - ✅ **3 days**
- Phase 1.5: Progressive curriculum training - ✅ **3.5 hours**
- Data collection script - ✅ **2 hours**
- Detector training script - ✅ **1 hour**

### Remaining (Estimated)
- Data collection - ⏳ **3-4 hours** (running now)
- Detector training - 🔜 **2-3 hours**
- Detector evaluation - 🔜 **30 minutes**
- DetectorWrapper implementation - 🔜 **1 hour**
- Testing detector + RL - 🔜 **1 hour**
- (Optional) Temporal tracking - 🔜 **2-3 hours**

**Total Phase 2 time:** ~10-14 hours (as estimated in Phase 1 plan)

---

## Success Metrics

### Detector Performance

**Minimum (Good Enough):**
- ✅ Detection rate > 85%
- ✅ Position error < 15 pixels
- ✅ Type accuracy > 75%
- ✅ Inference time < 20ms per frame

**Target (Ideal):**
- ✅ Detection rate > 90%
- ✅ Position error < 10 pixels
- ✅ Type accuracy > 85%
- ✅ Inference time < 10ms per frame

### End-to-End Performance

**Minimum:**
- ✅ Detector + RL achieves >70% of ground truth (100+ kills)
- ✅ Action diversity maintained
- ✅ Stable gameplay (no crashes or freezes)

**Target:**
- ✅ Detector + RL achieves >85% of ground truth (120+ kills)
- ✅ Reaches level 9+ consistently
- ✅ Smooth, human-like gameplay

---

## Risk Assessment

### Risk 1: Detector Accuracy Too Low

**Symptoms:**
- Detection rate < 80%
- Large position errors (>20 pixels)
- Many false positives/negatives

**Mitigations:**
1. Collect more data (2M+ frames instead of 500k)
2. Use stronger backbone (ResNet18 instead of custom CNN)
3. Try different architecture (YOLO-tiny)
4. Add more data augmentation
5. Increase training epochs (100 instead of 50)

**Fallback:** Use larger grid (32×32 instead of 16×16) for finer localization

### Risk 2: Detector Works But RL Fails

**Symptoms:**
- Detector metrics look good (>90% accuracy)
- But RL performance is poor (<50 kills)
- Policy collapses or acts erratically

**Mitigations:**
1. Add detection confidence to features
2. Train RL with synthetic detector errors (noise injection)
3. Use ensemble of detectors (majority vote)
4. Add temporal tracking to smooth detections

**Fallback:** Fine-tune position-based policy on noisy positions (simulating detector errors)

### Risk 3: Sprites Too Small/Occluded

**Symptoms:**
- Detector misses small sprites (bullets, family members)
- Occlusions cause many false negatives

**Mitigations:**
1. Use multi-scale detection (different grid sizes)
2. Add temporal tracking (predict occluded positions)
3. Increase input resolution (168×168 instead of 84×84)
4. Train on augmented data with synthetic occlusions

**Fallback:** Hybrid system (detector for large sprites, heuristics for small ones)

### Risk 4: Real-Time Performance Issues

**Symptoms:**
- Detector inference too slow (>50ms per frame)
- Can't run at 60 FPS for real-time play

**Mitigations:**
1. Optimize model (prune weights, quantization)
2. Use smaller backbone (MobileNet instead of ResNet)
3. Reduce grid size (8×8 instead of 16×16)
4. Use GPU acceleration (TensorRT)

**Fallback:** Run at lower frame rate (30 FPS) or skip frames

---

## Files Created

### Phase 2 Files
1. `collect_detector_data.py` - Data collection script
2. `train_detector.py` - Detector training script
3. `PHASE_2_STATUS.md` - This status document

### To Be Created
4. `eval_detector.py` - Detector evaluation script
5. `detector_wrapper.py` - Gym wrapper using detector
6. `test_detector_rl.py` - End-to-end testing script
7. `temporal_tracker.py` - (Optional) Temporal tracking

---

## Monitoring Data Collection

Check progress anytime:
```bash
# View live progress
tail -f data_collection.log

# Check file size (should grow to ~1-2 GB)
watch -n 60 ls -lh detector_dataset.npz

# Check process is running
ps aux | grep collect_detector_data

# Estimated time remaining (based on episodes completed)
# Current rate: ~13s per episode
# Total: 1000 episodes × 13s = ~3.6 hours
```

If data collection crashes or stalls:
```bash
# Kill the process
kill 446382

# Restart from scratch (or adjust --episodes to continue)
nohup poetry run python collect_detector_data.py \
    --episodes 1000 \
    --max-steps 500 \
    --output detector_dataset.npz \
    > data_collection.log 2>&1 &
```

---

## Current Session Summary

**Phase 1.5 Results:**
- ✅ 145 kills average (5x target!)
- ✅ Reaches level 11 (Stage 4 curriculum)
- ✅ Full action diversity
- ✅ Position-based RL scales to realistic scenarios

**Phase 2 Progress:**
- ✅ Data collection started (PID: 446382)
- ✅ Detector training script ready
- ⏳ Waiting for data collection (~3 hours remaining)
- 🔜 Train detector after collection completes

**Next Action:** Wait for data collection, then train detector.

**Estimated time to Phase 2 completion:** ~6-8 hours from now.
