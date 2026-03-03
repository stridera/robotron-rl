# FSM-Based Training Plan for Level 40

## Strategy Overview

**Goal:** Build a near-perfect FSM player, then use imitation learning to train RL policy

**Why this approach:**
1. **Perfect information** - Simulator gives us all sprite positions/types
2. **No time pressure** - Can think as long as we want (fps=0)
3. **Deterministic** - Can test and iterate quickly
4. **Proven path** - Humans reach level 100+ with similar strategies
5. **Better than RL alone** - Imitation learning provides strong initialization

---

## Current Status

### Test Results

| Approach | Level | Performance |
|----------|-------|-------------|
| Current RL (progressive curriculum) | 11 | 145 kills |
| Current RL (real config) | 1.4 | 74 kills |
| Expert FSM v1 (real config) | 1.3 | 34 kills |
| Original FSM (real config) | ??? | Timed out |

**Analysis:**
- Expert FSM v1 performs similarly to RL on real config
- Both struggle with level 1's 23 sprites (15 grunts + 5 electrodes + family)
- Need to improve FSM before using for imitation learning

---

## FSM Design Philosophy

### Expert Strategies (from human players)

1. **Edge Circling**
   - Stay near walls, avoid center
   - Enemies converge in center = death trap
   - Edges give escape routes

2. **Priority System**
   - Spawners first (Brain, Sphereoid, Quark)
   - Shooting enemies second (Enforcer, Tank)
   - Regular enemies third (Grunt)
   - Always collect family when safe

3. **Kiting**
   - Maintain optimal distance from enemies
   - Shoot while retreating
   - Never let enemies surround you

4. **Projectile Dodging**
   - Highest priority - dodge bullets/missiles
   - Shoot projectiles when close

5. **Predictive Movement**
   - Anticipate where enemies will be
   - Don't walk into clusters
   - Create clear escape routes

### Current FSM Implementation (expert_fsm.py)

**Decision hierarchy:**
1. Emergency dodge projectiles (distance < 60)
2. Retreat to edge when surrounded (>5 enemies close or in center)
3. Collect family (when path is clear)
4. Kite enemies (maintain optimal distance)
5. Default: patrol edges

**Problems identified:**
- Distance thresholds may be too conservative
- Not aggressive enough at clearing spawners
- Doesn't predict enemy convergence well
- Kiting behavior could be smarter

---

## Improvement Iteration Plan

### Phase 1: Debug and Tune (1-2 hours)

**Step 1: Add Debugging/Visualization**

Create diagnostic tool to watch FSM play and understand failures:

```python
# diagnose_fsm.py
# - Record FSM decisions per frame
# - Visualize sprite positions and FSM choices
# - Identify failure patterns (why does it die?)
```

**Step 2: Tune Distance Thresholds**

Current thresholds may be suboptimal:
- IMMEDIATE_DANGER = 60
- DANGER_ZONE = 120
- OPTIMAL_DISTANCE = 180
- SAFE_DISTANCE = 250

Test variations:
- More aggressive (shorter distances)
- More defensive (longer distances)
- Adaptive (based on enemy count)

**Step 3: Improve Prioritization**

Test if spawner priority is working:
- Are we killing brains/sphereoids fast enough?
- Should we prioritize them even more?
- Add explicit "hunt spawner" mode when detected

### Phase 2: Advanced Tactics (2-4 hours)

**Tactic 1: Predictive Enemy Tracking**

Instead of reacting to current positions, predict where enemies will be:

```python
def predict_enemy_position(enemy, timesteps=10):
    # Grunts move toward player
    # Enforcers chase player
    # Sphereoids/Quarks move randomly
    # Calculate likely future position
    return predicted_x, predicted_y
```

**Tactic 2: Cluster Avoidance**

Detect when enemies are forming clusters and avoid those areas:

```python
def find_enemy_clusters():
    # Group enemies within 80 pixels of each other
    # Mark cluster centers as danger zones
    # Never path through clusters
    return clusters
```

**Tactic 3: Clearing Waves**

Systematic approach to clearing levels:

```python
1. Identify all spawners (Brain, Sphereoid, Quark)
2. Pick nearest spawner
3. Position at optimal distance from spawner
4. Kill spawner while dodging its spawns
5. Repeat until all spawners dead
6. Clean up remaining enemies
7. Collect family
```

**Tactic 4: Dynamic Risk Assessment**

Calculate "danger score" for current position:

```python
def get_danger_score(position):
    score = 0
    for enemy in enemies:
        distance = get_distance(position, enemy)
        threat = enemy_threat_level[enemy.type]
        score += threat / (distance + 1)
    return score

# Move toward lowest danger score
```

**Tactic 5: Safe Zones**

Maintain knowledge of "safe zones" (corners/edges with escape routes):

```python
SAFE_ZONES = [
    (50, 50),         # Top-left corner
    (615, 50),        # Top-right corner
    (50, 442),        # Bottom-left corner
    (615, 442),       # Bottom-right corner
]

def retreat_to_safe_zone():
    nearest_safe = min(SAFE_ZONES, key=lambda z: distance(player, z))
    return navigate_to(nearest_safe)
```

### Phase 3: Level-Specific Strategies (2-4 hours)

Different levels need different approaches:

**Level 1-5:** Grunt-heavy
- Aggressive family collection
- Kite grunts while shooting
- Prioritize electrodes (obstacles)

**Level 5-15:** Spawner introduction
- Hunt spawners immediately
- Deal with spawned enemies
- Family collection secondary

**Level 15-25:** Dense enemies
- More defensive
- Stick to edges religiously
- Only collect family when enemies thin

**Level 25-40:** Maximum difficulty
- Surgical spawner elimination
- Perfect kiting
- Minimal risks

---

## Testing Protocol

### Metrics to Track

1. **Survival Time**
   - Steps before death
   - Level reached

2. **Kill Efficiency**
   - Kills per step
   - Spawners killed
   - Family collected

3. **Positioning Quality**
   - Time spent near edges vs center
   - Average distance to nearest enemy
   - Times surrounded (>5 enemies within 120 pixels)

4. **Decision Quality**
   - % time shooting
   - % time moving
   - Direction changes per second (jitteriness)

### Automated Testing

```python
# test_fsm_improvements.py

def test_fsm_variant(fsm, episodes=10):
    results = []
    for ep in range(episodes):
        score, level, kills, metrics = run_episode(fsm)
        results.append({
            'score': score,
            'level': level,
            'kills': kills,
            'survival_time': metrics['steps'],
            'spawners_killed': metrics['spawners'],
            'family_collected': metrics['family'],
        })
    return results

# A/B test different FSM variants
variant_a = ExpertFSM(threshold_multiplier=1.0)
variant_b = ExpertFSM(threshold_multiplier=0.8)  # More aggressive

results_a = test_fsm_variant(variant_a)
results_b = test_fsm_variant(variant_b)

print(f"Variant A: Level {avg_level(results_a)}")
print(f"Variant B: Level {avg_level(results_b)}")
```

---

## Success Criteria

### Milestone 1: Beat Current RL (Target: 1-2 days)
- FSM reaches level 5+ consistently
- Average 50+ kills
- Demonstrates clear improvement over random/basic strategies

### Milestone 2: Strong Expert (Target: 3-5 days)
- FSM reaches level 10+ consistently
- Average 100+ kills
- Ready for imitation learning

### Milestone 3: Near-Perfect Play (Target: 1-2 weeks)
- FSM reaches level 20+ consistently
- Average 200+ kills
- Demonstrates mastery of early/mid game

### Milestone 4: Level 40 Capable (Target: 2-4 weeks)
- FSM reaches level 40 at least once
- Can survive in dense scenarios (80+ sprites)
- Optimal for imitation learning

---

## Imitation Learning Pipeline

Once FSM reaches Milestone 2 (level 10+), start collecting demonstrations:

### Step 1: Collect Demonstrations

```python
# collect_expert_demos.py

fsm = ImprovedExpertFSM()

demonstrations = []
for episode in range(1000):
    episode_data = {
        'observations': [],
        'actions': [],
        'rewards': [],
    }

    obs = env.reset()
    done = False

    while not done:
        # Get sprite data (ground truth positions)
        sprite_data = env.get_sprites()

        # Convert to position features (same as RL training)
        position_obs = sprite_to_position_features(sprite_data)

        # FSM decision
        move, fire = fsm.decide(sprite_data)
        action = encode_action(move, fire)

        # Store demonstration
        episode_data['observations'].append(position_obs)
        episode_data['actions'].append(action)

        obs, reward, done, info = env.step(action)
        episode_data['rewards'].append(reward)

    demonstrations.append(episode_data)

save_demonstrations(demonstrations, 'fsm_expert_demos.pkl')
```

### Step 2: Behavior Cloning (Supervised Learning)

```python
# train_bc.py

from stable_baselines3 import PPO
from imitation.algorithms import bc

# Load demonstrations
demos = load_demonstrations('fsm_expert_demos.pkl')

# Create RL environment (same as training setup)
env = make_position_env(config='config.yaml', max_sprites=50)

# Initialize policy
policy = PPO('MlpPolicy', env, policy_kwargs={'net_arch': [1024, 1024, 512]})

# Behavior cloning
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=demos,
    policy=policy.policy,
)

# Train to imitate expert
bc_trainer.train(n_epochs=20)

# Save pre-trained policy
policy.save('models/bc_pretrained.zip')
```

### Step 3: RL Fine-Tuning

```python
# finetune_rl.py

# Load BC pre-trained policy
model = PPO.load('models/bc_pretrained.zip', env=env)

# Fine-tune with RL (PPO)
model.learn(total_timesteps=10_000_000)

# Evaluation
# Expected: BC gives strong initialization (level 8-10)
# RL fine-tuning optimizes beyond FSM (level 15-20+)
```

---

## Why This Approach Will Work

1. **FSM provides strong baseline**
   - Better than random exploration
   - Encodes human expert knowledge
   - Deterministic and debuggable

2. **Imitation learning jumpstarts RL**
   - RL starts from competent policy (level 10+)
   - Avoids wasting time learning basics
   - Focuses on optimization

3. **RL optimization goes beyond FSM**
   - Learns patterns FSM can't encode
   - Adapts to specific scenarios
   - Discovers emergent strategies

4. **Proven in research**
   - Many papers show IL + RL > pure RL
   - Especially for complex games
   - Atari games reached superhuman with similar approach

---

## Next Steps

### Option A: Iterative FSM Improvement (Recommended)

1. Add FSM debugging/visualization
2. Test 5-10 FSM variants with different parameters
3. Identify what works (A/B testing)
4. Implement advanced tactics (cluster avoidance, prediction)
5. Reach Milestone 2 (level 10+)
6. Collect demonstrations
7. Train RL with imitation learning

**Time estimate:** 1-2 weeks to strong expert FSM, 1 week for IL+RL training

### Option B: Parallel Development

1. Continue improving FSM (target level 5-10)
2. Meanwhile, collect demos from current FSM (level 2-3)
3. Train RL with weak demonstrations
4. As FSM improves, collect better demos
5. Iteratively retrain RL with better demos

**Time estimate:** Similar timeline, but more complex

### Option C: Human Demonstrations

1. You play the game yourself for 1-2 hours
2. Record your demonstrations
3. Train RL directly from your gameplay
4. Skip FSM development entirely

**Time estimate:** 1-2 days for data collection, 1 week for training

---

## Recommendation

**Start with Option A**: Iterative FSM improvement

**Why:**
- FSM is debuggable and deterministic
- Can run millions of episodes quickly
- Don't need to manually play
- Once FSM is strong, demonstrations are unlimited

**First concrete steps:**
1. Create `diagnose_fsm.py` - visualize why FSM dies
2. Tune distance thresholds (try 3-5 variants)
3. Add spawner hunting mode
4. Test on 20 episodes each, pick best variant
5. Iterate until reaching level 5+

Would you like me to start implementing the diagnostic tool and FSM improvements?
