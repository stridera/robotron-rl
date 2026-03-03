# Strategy to Reach Level 40 on Real Config

## Current Situation

**Goal:** Reach level 40 on `config.yaml` (real game difficulty)

**Current Performance:**
- Progressive curriculum model: Reaches level 11 (145 kills average)
- Real config evaluation: **Level 1.4 average, max level 2**
- **Gap: 38 levels to goal**

---

## Problem Analysis

### Why Does the Model Fail on Real Config?

Comparing `progressive_curriculum.yaml` vs `config.yaml`:

| Aspect | Progressive Level 1 | Real Level 1 | Difference |
|--------|-------------------|--------------|------------|
| **Grunts** | 1 | 15 | **15x more enemies!** |
| **Electrodes** | 0 | 5 | +5 obstacles |
| **Family** | 0 | 2 (Mommy, Daddy) | +2 family |
| **Total sprites** | ~2 | ~23 | **11.5x more crowded** |
| **Grunt speed** | 6 | 7 | 17% faster |
| **Grunt move delay** | [8, 30] | [5, 25] | More aggressive |

**Root cause:** Model trained on 1-35 sprites gradually. Real config throws 23 sprites at level 1. This is like training level 12-15 difficulty!

### Additional Challenges

Real config Level 1 ≈ Progressive Level 12-15 difficulty
Real config Level 5 ≈ Progressive Level 20+ difficulty
Real config Level 10+ = Beyond anything the model has seen

**Enemy count by level (real config):**
- Level 1: 23 sprites
- Level 5: 73 sprites
- Level 10: 70+ sprites
- Level 20: 100+ sprites
- Level 40: 120+ sprites

**Current model max_sprites=20** - immediately truncating 3+ sprites on level 1!

---

## Proposed Strategy

### Phase 1: Immediate Fixes (1-2 hours training)

#### 1.1: Increase max_sprites to Handle Density

```python
# Current: max_sprites = 20
# Real level 1: 23 sprites
# Real level 40: 120+ sprites

# Options:
max_sprites = 30  # Handle up to level 10 (conservative, 664-dim obs)
max_sprites = 50  # Handle up to level 30 (aggressive, 1104-dim obs)
max_sprites = 80  # Handle level 40+ (very large, 1764-dim obs)
```

**Recommendation:** Start with `max_sprites=50`
- Observation dim: 2 + 50 * 22 = 1102 (manageable)
- Covers most realistic scenarios (level 30)
- Can train in reasonable time

#### 1.2: Scale Up Network Capacity

```python
# Current: net_arch = [512, 512]
# With obs_dim=1102, need much larger network

# Progressive scaling:
net_arch = [1024, 1024]         # 2x current (good for max_sprites=30)
net_arch = [1024, 1024, 512]    # Deeper (better for max_sprites=50)
net_arch = [2048, 2048]         # Very large (for max_sprites=80)
```

**Recommendation:** `[1024, 1024, 512]`
- 3-layer MLP for better capacity
- Not too large (trains faster)
- Can upgrade to [2048, 2048] if needed

#### 1.3: Bridge Curriculum

Create `real_config_curriculum.yaml` that gradually introduces real difficulty:

```yaml
waves:
  # Stage 1: Progressive→Real transition (Levels 1-5)
  - [5, 2, 0, 0, 0, 0, 1, 1, 0]   # Level 1: Easier start
  - [8, 3, 0, 0, 0, 0, 1, 1, 0]   # Level 2
  - [10, 4, 0, 0, 0, 0, 1, 1, 1]  # Level 3
  - [12, 5, 0, 0, 0, 0, 1, 1, 1]  # Level 4
  - [15, 5, 0, 0, 0, 0, 1, 1, 0]  # Level 5: Now matching real Level 1

  # Stage 2: Real levels 1-10 (Levels 6-15)
  - [15, 5, 0, 0, 0, 0, 1, 1, 0]  # Real Level 1
  - [17, 15, 5, 0, 1, 0, 1, 1, 1] # Real Level 2
  - [22, 25, 6, 0, 3, 0, 2, 2, 2] # Real Level 3
  ... # Continue with real levels 1-10

  # Stage 3: Real levels 10-20 (Levels 16-25)
  ... # Real levels 10-20

  # Stage 4: Real levels 20-40 (Levels 26-45)
  ... # Real levels 20-40, repeated for practice
```

**Key idea:** Start easier than real, gradually match real difficulty, then practice real levels.

---

### Phase 2: Training Protocol (10-20M steps, ~12-24 hours)

#### 2.1: Curriculum Training

```bash
# Step 1: Train on bridge curriculum (5M steps, ~6 hours)
poetry run python train_positions.py \
    --config real_config_curriculum.yaml \
    --max-sprites 50 \
    --net-arch 1024 1024 512 \
    --total-timesteps 5000000 \
    --num-envs 8 \
    --project robotron \
    --group real_config_bridge

# Step 2: Fine-tune on full real config (10M steps, ~12 hours)
poetry run python train_positions.py \
    --config config.yaml \
    --max-sprites 50 \
    --net-arch 1024 1024 512 \
    --total-timesteps 10000000 \
    --num-envs 8 \
    --resume models/{run_id}/checkpoints/checkpoint_5000000_steps.zip \
    --project robotron \
    --group real_config_full

# Step 3: Extended training if needed (10M+ steps)
# Continue training until level 40 reached
```

#### 2.2: Training Hyperparameters

```python
# PPO hyperparameters tuned for dense environments
ppo_kwargs = {
    'learning_rate': 3e-4,      # Standard
    'n_steps': 2048,            # More steps per update (dense levels)
    'batch_size': 256,          # Larger batches
    'n_epochs': 10,             # Standard
    'gamma': 0.99,              # Standard discount
    'gae_lambda': 0.95,         # Standard GAE
    'clip_range': 0.2,          # Standard clip
    'ent_coef': 0.01,           # Encourage exploration
    'vf_coef': 0.5,             # Standard value loss weight
    'max_grad_norm': 0.5,       # Gradient clipping
}
```

#### 2.3: Reward Shaping

Add reward shaping to prioritize survival and level progression:

```python
# In wrapper or training script:
def shaped_reward(info, prev_info):
    reward = 0

    # 1. Level progression bonus
    if info['level'] > prev_info['level']:
        reward += 1000  # Big reward for clearing a level

    # 2. Survival bonus (every step alive)
    reward += 0.1

    # 3. Kill rewards (scaled by enemy type)
    score_delta = info['score'] - prev_info['score']
    reward += score_delta / 100.0  # Standard kill rewards

    # 4. Death penalty
    if info['lives'] < prev_info['lives']:
        reward -= 10

    return reward
```

---

### Phase 3: Advanced Techniques (If Phase 2 Insufficient)

#### 3.1: Hierarchical RL

Separate high-level (strategic) and low-level (tactical) policies:

```python
# High-level policy: Decide strategy (attack, defend, collect family)
# Low-level policy: Execute movement/shooting for chosen strategy

# Observation:
# - High-level: Enemy counts, player health, level, global state
# - Low-level: Nearby sprites (position features)

# Action:
# - High-level: Strategy choice (0=aggressive, 1=defensive, 2=collect)
# - Low-level: Movement + shooting (current 64-action space)
```

**Benefits:**
- High-level learns when to be aggressive vs defensive
- Low-level learns combat tactics
- Easier to train (smaller action/obs spaces per policy)

#### 3.2: Curriculum with Level Repetition

Focus training on bottleneck levels:

```yaml
# If agent struggles on levels 10-15, repeat them:
waves:
  - ... # levels 1-9
  - [level_10_config]  # Repeat 3x
  - [level_10_config]
  - [level_10_config]
  - [level_11_config]  # Repeat 3x
  - [level_11_config]
  - [level_11_config]
  ...
```

#### 3.3: Imitation Learning from FSM

Use the FSM player (`robotron_fsm.py`) to collect expert demonstrations:

```bash
# Collect expert demonstrations
poetry run python collect_fsm_demos.py \
    --episodes 1000 \
    --config config.yaml \
    --output fsm_demos.pkl

# Pre-train with behavior cloning
poetry run python pretrain_bc.py \
    --demos fsm_demos.pkl \
    --epochs 10

# Fine-tune with RL
poetry run python train_positions.py \
    --resume models/bc_pretrained/best_model.zip \
    --total-timesteps 10000000
```

**Benefits:**
- Jumpstart with expert behavior
- Learn good positioning from FSM
- RL fine-tuning optimizes beyond FSM

#### 3.4: Multi-Task Learning

Train on multiple configurations simultaneously:

```python
# Sample from different configs during training
configs = [
    'progressive_curriculum.yaml',  # 30% of time
    'real_config_curriculum.yaml',  # 40% of time
    'config.yaml',                  # 30% of time
]

# Each episode, randomly sample a config
# Agent learns robust strategy across all difficulties
```

---

## Implementation Plan

### Week 1: Foundation (Current → Level 10)

**Day 1-2:**
- [ ] Increase max_sprites to 50
- [ ] Scale network to [1024, 1024, 512]
- [ ] Create `real_config_curriculum.yaml`
- [ ] Add reward shaping wrapper

**Day 3-5:**
- [ ] Train on bridge curriculum (5M steps)
- [ ] Evaluate: Target level 5-10 on real config

**Day 6-7:**
- [ ] Fine-tune on real config (5M steps)
- [ ] Evaluate: Target level 8-12 on real config

### Week 2: Scaling (Level 10 → Level 25)

**Day 8-10:**
- [ ] Extended training on real config (10M steps)
- [ ] Monitor level progression
- [ ] Add level-specific reward bonuses if stuck

**Day 11-14:**
- [ ] If progress stalls, implement imitation learning
- [ ] Collect FSM demonstrations
- [ ] Pre-train with behavior cloning
- [ ] Fine-tune with RL (5M steps)

### Week 3-4: Advanced (Level 25 → Level 40)

**Day 15-21:**
- [ ] Very long training runs (20M-50M steps)
- [ ] Consider hierarchical RL if needed
- [ ] Tune reward shaping for late-game priorities
- [ ] Possibly increase max_sprites to 80 if truncation is issue

**Day 22-28:**
- [ ] Final optimization
- [ ] Hyperparameter sweeps
- [ ] Ensemble policies (if single policy insufficient)
- [ ] Extensive evaluation on levels 30-40

---

## Expected Milestones

| Milestone | Training Steps | Wall Time | Expected Level |
|-----------|---------------|-----------|----------------|
| Baseline (current) | 3M | ~3.5 hrs | 1.4 |
| After bridge curriculum | 5M | ~6 hrs | 8-12 |
| After real config training | 15M | ~18 hrs | 15-20 |
| After extended training | 30M | ~36 hrs | 25-30 |
| After imitation learning | 40M | ~48 hrs | 30-35 |
| **Level 40 (GOAL)** | **50M-100M** | **60-120 hrs** | **40+** |

---

## Quick Start: Begin Training Now

### Option A: Conservative (Likely to reach level 20-25)

```bash
# 1. Create bridge curriculum (manual - copy levels from this doc)
nano real_config_curriculum.yaml

# 2. Train with moderate scaling
poetry run python train_positions.py \
    --config real_config_curriculum.yaml \
    --max-sprites 30 \
    --net-arch 1024 1024 \
    --total-timesteps 10000000 \
    --num-envs 8 \
    --project robotron \
    --group level40_attempt1
```

### Option B: Aggressive (Best chance at level 40)

```bash
# 1. Create bridge curriculum
nano real_config_curriculum.yaml

# 2. Train with large scaling
poetry run python train_positions.py \
    --config real_config_curriculum.yaml \
    --max-sprites 50 \
    --net-arch 1024 1024 512 \
    --total-timesteps 20000000 \
    --num-envs 8 \
    --learning-rate 3e-4 \
    --ent-coef 0.01 \
    --project robotron \
    --group level40_attempt1
```

### Option C: Maximum Effort (Highest chance at level 40)

```bash
# 1. Create bridge curriculum
# 2. Collect FSM demonstrations
# 3. Pre-train with behavior cloning
# 4. Extended RL training (50M+ steps)
# 5. Hierarchical RL if needed

# Total time: 3-4 weeks
# Success probability: 70-80%
```

---

## Success Criteria

**Minimum Success (Level 20):**
- Completes bridge curriculum training
- Reaches level 15-20 consistently
- Total training: 15M steps (~18 hours)

**Target Success (Level 30):**
- Extended training on real config
- Reaches level 25-30 consistently
- Total training: 30M steps (~36 hours)
- May require imitation learning

**Goal Success (Level 40):**
- Very long training (50M-100M steps)
- Reaches level 40+ at least once
- Total training: 60-120 hours (2.5-5 days continuous)
- Likely requires imitation learning + advanced techniques

---

## Risk Assessment

**High Risk Factors:**
1. **Max sprites truncation** - Level 40 has 120+ sprites, even max_sprites=50 truncates 70
2. **Training time** - May need 100M+ steps (5+ days)
3. **Network capacity** - MLP may not be powerful enough for late-game complexity
4. **Local minima** - Agent may get stuck at level 20-30 plateau

**Mitigation:**
- Test with max_sprites=80 or 100 if truncation is bottleneck
- Use distributed training (16+ envs) to speed up
- Consider CNN policy (spatial awareness) if MLP insufficient
- Heavy use of curriculum + imitation learning to guide training

---

## Next Action

**Choose your approach:**

1. **Quick Test (2 hours):** Run Option A above, see if we can reach level 10
2. **Serious Attempt (1 week):** Implement bridge curriculum + extended training
3. **Full Commitment (3-4 weeks):** All advanced techniques for best chance at level 40

Which would you like to pursue?
