# YOLO11 + RL Integration Results

## Summary

Successfully integrated your pre-trained YOLO11 detector with the position-based RL policy. However, performance is significantly degraded due to a **domain shift** between Xbox 360 frames (where YOLO was trained) and simulator frames (where RL runs).

---

## Integration Status

### ✅ Completed

1. **Mapped YOLO class names to RL sprite types**
   - YOLO model: `/home/strider/Code/labelbox_to_yolo/runs/detect/train3/weights/best.pt`
   - 13 classes: Player, Civilian, Grunt, Hulk, Sphereoid, Enforcer, Brain, Tank, Quark, Electrode, Enforcer Bullet, Converted Civilian, Brain Bullet
   - Mapped to RL's 17 sprite types (some missing: Mommy, Daddy, Cruise, TankShell)

2. **Created YOLODetectorWrapper**
   - File: `yolo_detector_wrapper.py`
   - Converts YOLO detections → position features (442-dim) → RL policy actions
   - Compatible with trained position-based policy (max_sprites=20)
   - Uses ultralytics YOLO11 API

3. **Tested integration successfully**
   - YOLO model loads correctly
   - Position features generated properly
   - RL policy accepts features and produces actions
   - No crashes or errors

### ❌ Critical Issue: Domain Shift

**Problem:** YOLO trained on Xbox 360 frames != simulator frames

**Detection Performance:**
- Ground truth sprites: 5.5 ± 1.1 per frame
- YOLO detections: 2.5 ± 1.4 per frame
- **Detection rate: 44.9%** (detecting less than half of sprites!)

**RL Performance:**
- With YOLO detector: 2.3 kills average
- With ground truth positions: 145 kills average
- **Performance degradation: 98.4%**

---

## Diagnostic Results

Ran 100 frames through diagnostic script (`diagnose_yolo_detections.py`):

### Detection Statistics

| Metric | Value |
|--------|-------|
| Average sprites (ground truth) | 5.5 ± 1.1 |
| Average YOLO detections | 2.5 ± 1.4 |
| Detection rate | **44.9%** |

### Sprite Types Detected (100 frames)

| Sprite Type | Detections | Notes |
|------------|------------|-------|
| Grunt | 88 | Most common |
| Civilian | 58 | Good detection |
| Brain | 43 | Moderate |
| Enforcer | 28 | Lower detection |
| Electrode | 17 | Lower detection |
| Player | 13 | Surprisingly low! |

**Missing:** No detections for Hulk, Sphereoid, Quark, Tank, or any bullets. This is concerning.

---

## Root Cause Analysis

### Why is YOLO Missing 55% of Sprites?

1. **Visual Appearance Mismatch**
   - Xbox 360: Real hardware graphics, specific visual style
   - Simulator: Python/Pygame recreation, different rendering
   - Sprites look visually different between the two

2. **Resolution/Scaling Differences**
   - YOLO trained on Xbox frames (unknown resolution)
   - Simulator outputs 665×492 RGB frames
   - Possible aspect ratio or scaling mismatch

3. **Missing Sprite Types**
   - YOLO only detects 6/13 classes in diagnostics
   - Many enemy types (Hulk, Sphereoid, Tank, Quark) not detected at all
   - Could be rare in training data OR visual differences

4. **Confidence Threshold**
   - Currently using 0.25 (same as training validation)
   - Could try lowering to 0.15, but won't fix visual mismatch

---

## Solutions (Ordered by Effectiveness)

### Solution 1: Train YOLO on Simulator Frames ⭐ **RECOMMENDED**

**Pros:**
- Eliminates domain shift completely
- Can use ground truth positions from simulator directly
- Fast data collection (500k frames in ~3 hours)
- Already have collection script (`collect_detector_data.py`)

**Cons:**
- Requires 2-3 hours training time
- Won't work on real Xbox without transfer learning

**Implementation:**
1. Run existing data collection script (already started earlier)
2. Train YOLO11 on simulator frames
3. Use new detector with RL policy
4. Expected detection rate: >90% (vs current 45%)

**Status:** Data collection was started (PID 446382) but may have stopped. Can restart or use ground truth detector training approach.

---

### Solution 2: Fine-tune Xbox YOLO on Simulator Frames

**Pros:**
- Keeps some Xbox knowledge
- Might transfer better to real Xbox later
- Potentially better generalization

**Cons:**
- More complex (transfer learning)
- Still requires collecting simulator data
- Training time longer (~5-10 epochs)

**Implementation:**
1. Collect simulator frames (as above)
2. Fine-tune your existing YOLO11 model on simulator data
3. Test on both simulator and Xbox frames

---

### Solution 3: Use Ground Truth Positions (Current Approach)

**Pros:**
- Perfect detection (100%)
- Proven to work (145 kills)
- No detector training needed

**Cons:**
- Only works in simulator
- Can't deploy to real Xbox
- Defeats purpose of image-based detection

**Status:** This is Phase 1.5, already completed and validated.

---

### Solution 4: Domain Adaptation / Data Augmentation

**Pros:**
- No new YOLO training required
- Could improve Xbox YOLO's robustness

**Cons:**
- Complex to implement
- Uncertain improvement
- Still likely <70% detection rate

**Implementation:**
- Apply color/contrast/brightness augmentations to simulator frames
- Hope to bridge visual gap
- Unlikely to fully solve problem

---

## Recommendation

**Go with Solution 1: Train YOLO on Simulator Frames**

### Reasoning:

1. **Your ultimate goal is Xbox deployment**, not simulator performance
2. The current Xbox-trained YOLO has 56.3% mAP on Xbox images
3. Training a **separate simulator YOLO** lets you:
   - Validate the full pipeline (pixels → detections → positions → actions)
   - Test if detection quality is the bottleneck
   - Keep your Xbox YOLO pristine for later Xbox deployment

4. **Two-model approach:**
   - **Simulator YOLO**: For testing/development in simulator (train now)
   - **Xbox YOLO**: For final deployment on real hardware (already have)

5. Later, you can combine approaches:
   - Use simulator to train RL policy
   - Use domain adaptation to align simulator → Xbox
   - Deploy trained policy + Xbox YOLO to real hardware

### Timeline:

| Step | Time | Status |
|------|------|--------|
| Collect simulator data | 3-4 hours | ⏳ Can restart collection |
| Train YOLO11 on simulator | 2-3 hours | 🔜 Pending |
| Test detector accuracy | 30 min | 🔜 Pending |
| Test YOLO+RL performance | 1 hour | 🔜 Pending |
| **Total** | **6-8 hours** | |

---

## Alternative: If You Want to Use Xbox YOLO

If you want to stick with your Xbox-trained YOLO despite the domain shift:

### Quick Fixes to Try:

1. **Lower confidence threshold**
   ```python
   # In yolo_detector_wrapper.py, line 39
   env = YOLODetectorWrapper(env, yolo_model_path, max_sprites=20, confidence_threshold=0.15)
   ```
   - Might increase detection rate from 45% to 55-60%
   - Won't fully solve problem

2. **Increase image size**
   - YOLO trained on 640×640 images
   - Simulator outputs 665×492
   - Could rescale simulator frames to match YOLO training

3. **Visual preprocessing**
   - Adjust contrast/brightness of simulator frames
   - Try to match Xbox visual style
   - Unlikely to get >70% detection rate

**Expected outcome:** With all tricks, might reach 60-70% detection rate, yielding 20-40 kills (still far from 145).

---

## Files Created

1. **`yolo_detector_wrapper.py`** - Main wrapper integrating YOLO11 with RL
2. **`test_yolo_integration.py`** - Quick integration test (3 episodes)
3. **`diagnose_yolo_detections.py`** - Diagnostic tool analyzing detection performance
4. **`YOLO_INTEGRATION_RESULTS.md`** - This document

---

## Next Steps

### Option A: Train Simulator YOLO (Recommended)

```bash
# 1. Check if data collection is still running
ps aux | grep collect_detector_data

# 2. If not, restart it:
nohup poetry run python collect_detector_data.py \
    --episodes 1000 \
    --max-steps 500 \
    --output detector_dataset.npz \
    > data_collection.log 2>&1 &

# 3. Monitor progress:
tail -f data_collection.log

# 4. After completion, train YOLO:
cd ../labelbox_to_yolo
poetry run yolo train \
    data=/home/strider/Code/robotron-rl/detector_dataset.npz \
    model=yolo11n.pt \
    epochs=50 \
    imgsz=640 \
    batch=16

# 5. Update yolo_detector_wrapper.py to use new model:
# Change line 271: yolo_model_path = '../labelbox_to_yolo/runs/detect/train4/weights/best.pt'

# 6. Test integration:
poetry run python test_yolo_integration.py
```

### Option B: Use Ground Truth (Phase 1.5)

```bash
# Already validated - just use the position-based RL policy as-is
poetry run python check_progressive_model.py \
    models/6l9t4lpc/checkpoints/ppo_progressive_checkpoint_3000000_steps.zip
```

---

## Performance Comparison

| Approach | Detection Rate | RL Performance (kills) | Deployment |
|----------|---------------|----------------------|------------|
| Ground truth positions | 100% | 145 | Simulator only |
| Xbox YOLO (current) | 45% | 2.3 | Xbox ready |
| Simulator YOLO (proposed) | 90%+ (est.) | 115-130 (est.) | Simulator only |
| Fine-tuned YOLO | 70-80% (est.) | 70-100 (est.) | Both (maybe) |

**Conclusion:** Need simulator-trained YOLO to validate the detection→RL pipeline. Later can work on Xbox deployment with domain adaptation.

---

## Questions?

1. **Should I train a simulator YOLO?**
   - Yes, if you want to validate the full pixel-based pipeline
   - No, if you only care about Xbox deployment (but then need Xbox simulator or data)

2. **Can I improve Xbox YOLO performance on simulator?**
   - Marginally (45% → 60%), but not enough for good RL
   - Would need significant domain adaptation work

3. **What about the vision-based RL (Phase 3)?**
   - Still viable, but will face same domain shift issue
   - Recommendation: Train vision-based RL on simulator, then transfer to Xbox

