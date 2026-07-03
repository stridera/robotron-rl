# Robotron RL Progress Review and Suggestions

## Overall assessment

The project has made strong engineering progress, but the actual wave-1-to-100
objective remains far away. The most important achievement is not the current
policy depth; it is the faithful, instrumented experimental platform and the
number of misleading explanations that have been eliminated.

The honest current scorecard is:

- The MAME environment has been validated against XBLA dynamics.
- Observations, entity classification, death forensics, RNG reseeding, state
  capture, and continuous evaluation are substantially more reliable.
- The best continuous learned policy reaches roughly mean wave 9, depending on
  checkpoint and evaluation sample.
- The best evolved FSM teacher reaches mean wave 12.72 and median wave 12 over
  the documented 40-game evaluation.
- Save-state specialists reached wave 30, but this did not translate into
  continuous wave-1 play.
- The discovery that waves 41-100 reuse configurations 21-40 reduces the
  variety that must be learned, but it does not remove the survival and life
  economy requirements of a 100-wave run.

At the time of this review, the `curric1` training process did not appear in
the process list. Its log stopped at 688,128 steps, so it probably terminated
instead of continuing detached.

## What has gone well

The project has unusually strong failure analysis. It found and corrected
several issues that could otherwise have invalidated the entire effort:

- The native environment had incorrect player dynamics.
- Deep save-state results were initially mistaken for marathon ability.
- Deterministic resets inflated evaluations and FSM evolution results.
- Save-state indices wrapped at 256.
- Entity classification and velocity calculations contained important errors.
- Several apparent observation and architecture bottlenecks were tested and
  disproved.
- Evaluation noise was eventually recognized as large enough to reverse some
  earlier conclusions.

The pivot toward teacher improvement was also supported by the evidence:

> Better teacher -> better anchored policy.

The evolved FSM lifting the learned policy beyond the old 8.3-wave plateau is
the clearest recent positive result.

## Main critiques

### 1. Evaluation samples remain too small

Robotron results are high variance and heavy-tailed. The history shows
20-game evaluations moving by approximately one wave, while even separate
40-game samples have produced apparently contradictory conclusions.

For important promotion decisions, use:

- At least 100 independently reseeded games.
- Paired seeds: candidate and baseline play the same seed set.
- Bootstrap confidence intervals for mean and median.
- A permanent held-out evaluation seed bank that is never used for training
  or FSM evolution.
- Survival probabilities such as:
  - `P(reach wave 5)`
  - `P(reach wave 10)`
  - `P(reach wave 15)`
  - `P(reach wave 20)`

Survival curves will show where a policy improves or regresses more clearly
than a single mean-wave number.

### 2. Too many proxy metrics have competed with the real objective

Highest wave from a deep reset, episode reward, BC loss, score, and teacher
fidelity are useful diagnostics, but none is the deployment objective.

The primary metric should be:

> Probability of reaching wave N in one continuous game from a genuine
> wave-1 start under held-out RNG seeds.

Every model promotion should be gated on this metric. Deep-reset performance
should remain diagnostic only.

### 3. The deep curriculum may recreate the specialist failure

The current curriculum pool contains 40 wave-1 entries and 62 deep-state
entries. This is approximately 39% wave-1 starts and 61% deep starts. That is
risky given the earlier finding that deep-reset training produced strong wave
specialists but weak continuous players.

Deep states also contain score, remaining lives, enemy state, and arrival
history produced by the FSM. The learner may become competent on a state
distribution it cannot reach naturally.

Test these separately:

1. Can the policy clear each wave from a captured start?
2. Can it enter that wave naturally and then clear it?
3. Does deep-state training reduce early-wave survival?
4. Does it preserve lives across wave transitions?
5. Does it improve continuous held-out wave-1 evaluation?

Do not promote a curriculum model based only on deep-reset performance.

### 4. Waves 41-100 do not necessarily generalize "for free"

The 20-wave configuration cycle is valuable, but repeating enemy
configurations does not make a continuous 100-wave run automatic:

- The player arrives with different lives and score.
- Errors and deaths accumulate.
- Extra-life economy matters.
- Rare mistakes compound over many waves.
- Any score-, timing-, or state-dependent behavior still needs validation.

For perspective, even a 95% probability of clearing each wave gives only
about a 0.6% chance of clearing 100 consecutive waves.

Mastering configurations 21-40 is probably necessary, but the bot also needs
extremely high per-wave reliability and sustainable life generation.

### 5. The teacher-to-policy pipeline throws away capability

The evolved FSM reaches mean wave 12.72, while the learned policy reaches
around wave 9. The neural policy is being asked to reproduce the whole FSM and
loses several waves in the process.

Unless deployment explicitly requires a neural policy, the FSM itself is
currently the better bot.

If learning is required, use the FSM as a permanent base controller and train
a residual policy:

- The FSM proposes movement and firing actions.
- The policy chooses whether to keep or override each action component.
- Penalize unnecessary overrides.
- Train overrides primarily in states where the FSM is likely to die.
- Fall back to the FSM when policy confidence is low.

This preserves the teacher's existing capability and gives RL a smaller,
better-defined improvement problem.

## Recommended next steps

### 1. Repair and finish `curric1`

Determine why it stopped around 688k steps. Resume or rerun it, then evaluate
late checkpoints through paired continuous tests of at least 100 games.

Do not promote it solely because training reward or deep-start competence
improves.

### 2. Establish one permanent evaluation protocol

Create a leaderboard that records:

- Model and checkpoint.
- Exact held-out seed bank.
- At least 100 continuous wave-1 games.
- Mean and median wave with confidence intervals.
- Survival probability by wave.
- Score distribution.
- Lives gained and lives lost by wave.
- Death causes.
- MAME result and, for promoted candidates, Xenia result.

This would prevent repeated reinterpretation caused by different random draws.

### 3. Build an FSM-residual controller

This is the highest-confidence architectural direction suggested by the
existing results. It avoids losing three or four waves by approximately
cloning a controller that already works better.

A useful first experiment is deliberately conservative:

- Initialize the residual policy to never override.
- Permit movement overrides first while retaining FSM firing.
- Require a measurable improvement over the FSM on paired held-out seeds.
- Add firing overrides only if movement residuals show value.

### 4. Improve the FSM structurally

Threshold evolution has produced real gains but appears closer to saturation.
The next teacher gains probably require additional decision structure rather
than continued tuning of the same thresholds.

Candidate additions include:

- Explicit escape behavior for converging threat fields.
- Risk-adjusted civilian rescue.
- Spawner-suppression strategy.
- A life-preservation mode near extra-life score thresholds.
- Specialized handling for Brain, Tank, Quark, and dense grunt waves.
- Detection of situations where firing toward a target conflicts with moving
  toward a safe region.

Each new behavior should be evaluated through paired ablations so its actual
contribution is measurable.

### 5. Replace short-horizon greedy search with value-guided search

The existing short-horizon search failed because most candidate actions
survived the horizon, making immediate score dominate the decision.

A more promising design is:

- Train a state-value or death-risk model from FSM trajectories.
- Search several movement choices for a moderate horizon.
- Score the endpoint using predicted future survival and life value.
- Retain the FSM's firing decision initially.

This may provide a stronger teacher without requiring very long MAME rollouts
for every decision.

### 6. Track life economy directly

A wave-100 player needs a sustainable balance between deaths and extra lives.
Track by wave:

- Probability of losing a life.
- Expected score gained.
- Probability of earning an extra life.
- Expected net change in lives.
- Civilian rescue rate.

If expected lives lost per wave exceed expected lives earned, wave 100 remains
improbable even when individual deep waves can be cleared.

## Strongest recommendation

Treat the evolved FSM as the base product and develop a learned residual
controller around it.

The accumulated evidence does not support replacing a mean-wave-12.72 teacher
with an approximately mean-wave-9 clone and expecting ordinary PPO to recover
the lost capability. Preserve the FSM's behavior, train only targeted
improvements, and judge every change through paired continuous wave-1
evaluation.
