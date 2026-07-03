# Native chain experiment state

## 2026-06-09 — PIVOT TO MAME GYM (critical fidelity finding)

**The native gym's player dynamics are non-physical.** Measured player-vs-grunt
speed ratio: native 1.30, MAME (faithful Williams emulation) 4.62. The native
trampoline calls player_movement once per env-step (~3 frame-times) instead of
every frame, so every native-trained policy learned a ~3.5x-slower player.
Confirmed empirically: wh4rfkdb scores 4,130 on native wave-1 but only 1,880 on
MAME — **below the random baseline (6,455)**. Native-trained policies will not
transfer to Xenia/real hardware.

**New environment: `mame_gym/`** (MAME 0.264 + verified arcade ROM at
~/mame_robotron/roms):
- `robotron_server.lua` — in-MAME socket server: boot→gameplay, save-state
  reset, synchronous step RPC (client-driven: read cmd → act → write obs)
- `mame_bridge.py` — launches headless MAME, connects, step/reset/quit
- `mame_obs.py` — slot-pool packet → 945-dim obs (same extractor as native)
- `mame_robotron_env.py` — SB3 env, MultiDiscrete([8,8]), same reward shaping
- Verified: addresses identical to XBLA ($BDED/$BDEC/$BDE5-7/$9864/$98D4),
  save-state reset deterministic, full waves playable (no wave-5 bug),
  sprite extraction sane, 4 envs = 5,098 fps (vs native 1,160 @ 16 envs)

**Consequence:** all native chain checkpoints (wh4rfkdb etc.) are suspect for
real-hardware transfer. Fresh training on MAME is the path forward.

## 2026-06-09 — Xenia validation: MAME (arcade) == XBLA confirmed

Identical input/measurement protocol driven in both emulators (probe scripts:
`~/win/code/robotron/xenia_dynamics_probe.py` via player.py gamepad bridge +
XeniaMemory; `/tmp/mame_dynamics_probe.py` via the mame_gym bridge):

| Metric | MAME (arcade ROM) | Xenia (XBLA binary) |
|---|---|---|
| Player spawn position | (74,124) | (74,124) |
| Starting lives | 2 | 2 |
| Wave-1 grunt count | 15 | 15 |
| **Player speed (held RIGHT)** | **25.2 u/s** | **25.7 u/s** (Δ 2%) |
| Grunt speed (median) | 0.0 (bursty motion) | 0.0 (bursty motion) |
| Grunt speed (mean) | 2.0 u/s | 1.2 u/s (phase-dependent) |
| Memory map ($BDED/$BDEC/$BDE5-7/$9864/$98D4) | identical | identical |

Player kinematics — the metric that exposed the native gym's 2.5x-slow player
bug — match within 2%. **The MAME gym is a validated faithful trainer for the
XBLA deployment target.** Native gym equivalent was ~10 u/s (2.5x slow).

Xenia probe gotcha for future use: player.py zeroes the virtual pad if no
command arrives within COMMAND_TIMEOUT — stick commands must be resent every
tick (the brains do this; one-shot probes must too).

## MAME chain — run history

**Two env bugs found and fixed during run #1 (mame_robotron_env.py):**
1. *Attract-mode reward poisoning:* on game over, $BDED flips to attract-mode
   garbage (wave=41) and $BDEC reads lives=2 — episodes banked ~+290k spurious
   wave bonuses ON DEATH (rewarding suicide; ep_rew_mean spiked 1.9k → 15.4k).
   Fixed: terminal step gets death penalty only; wave bonus only on exactly
   +1 transitions; non-sequential wave change ⇒ terminate (left gameplay).
2. *Rare MAME wedge:* one instance hung mid-run (~1 per 13M env-frames).
   Fixed: self-healing bridge — on socket timeout, relaunch MAME, terminate
   episode cleanly (mame_bridge._recover).

| Run | Steps | Envs | ep_rew_mean | highest_score | highest_wave | Notes |
|-----|-------|------|-------------|---------------|--------------|-------|
| pilot d5ul36oy | 500k | 8 | 1,685 | 18,700 | 4 | fresh; validated pipeline |
| #1 f3y6eg9k | 3M | 12 | 2,081 | **24,075** | **5** | fresh; wave 5 reached ORGANICALLY (no snapshot tricks); 0 wedges post-fix |
| #2 tm6kbksn | 3M | 12 | 2,185 | **35,825** (+49%) | 5 | warmstart from #1; clean metrics (terminal-info fix); 2 wedges auto-recovered |
| #3 jrbl8xt0 | 3M | 12 | **2,378** ↑ | 29,725 | 5 | mean still climbing (3rd consecutive ↑); max-score dip is noise |
| #4 im43wiua | 3M | 12 | **2,403** ↑ | **42,325** | 5 | **ALL-TIME RECORD** — beats native ATH 41,850, on faithful dynamics |
| #5 idli5n3k | 3M | 12 | 2,393 → | 29,850 | 5 | first flat link; 6 wedges auto-recovered |
| #6 arubzxev | 3M | 12 | 2,449 | 30,400 | 5 | gains collapsed to +1.9%/2 links; ep_len pinned ~557 ×4 links → wave-5 brain wall confirmed |

## 2026-06-10 — SPRITE CENSUS: policy was BLIND to Progs (user-prompted check)

User asked "have we verified sprite identification?" — answer was no. Census
tool (`mame_gym/sprite_census.py`) run across wave-1 and wave-5 starts found:

1. **Progs ($0390) entirely unmapped — 2,888 sightings.** The hostile
   brainwashed civilians that Brains create in wave 5 were INVISIBLE to the
   policy. We were training it to fight an enemy it could not see, in the
   exact wave where it plateaued. SPRITE_TYPES already had a Prog slot in the
   945-dim obs — never populated.
2. **Animation-variant SWs**: grunts cycle $3A09-$3A86, spheroids $1242-$126A,
   enforcers $1445-$1455 — exact-match made entities flicker out of obs ~5%
   of frames, AND caused phantom kill bonuses (spheroid animating → count
   drop → spawner "kill" reward).
3. Fix: range-based `classify_sw()` in mame_obs.py; kill counters use it too.
   Wave-5 obs density went 86 → 152 nonzero features (+77%).
4. Tank/TankShell/Quark SWs still UNVERIFIED (census only reached wave 5;
   they appear wave 6+). Rerun census when the chain reaches wave 6+.

Wave-5 save states captured (w5_1..w5_8 via capture_wave5_states.py, all
lives=1 — the policy's genuine arrival state). Link 7 = arubzxev warmstart +
50/50 reset pool (wave-1 boot / wave-5 states) + Prog-visible obs.

| #7 vzgyfwgi | 3M | 12 | 1,643* | 45,500 | 6 | DISCARDED (phantom-entity obs); wave-5 starts broke the brain wall |
| #8 thtz1tec | 3M | 12 | 1,715* | **46,825 ATH** | **6** | from arubzxev, list-walk validated obs; beats discarded #7 |
| #9 sd9yf7v2 | 1M (killed by my own cleanup) | 12 | — | 62,400 | **8** | wave-1/5/6 pool; blew through wave 7. Forensics caught Quark variant $4FD5 (was typed TankShell). Watchlist: $0390, $DE6D, $1CB1 |
| #9b vs1rr9pw | 3M (resumed from #9's 1M ckpt) | 12 | 1,694* | **65,125 ATH** | 7 | quark variants typed; ladder marches |
| #10 acjrovjy | 3M | 12 | 1,129* | **67,800 ATH** | **9** | wave-1/5/6/7 pool + teleport guard (respawn-hulks were mislabeled Prog); forensics clean at 94.2% |
| #11 k7r67how | 3M | 12 | 1,494* | **76,375 ATH** | 9 | wave-1/5/6/7/8 pool |
| #12 4tf0ne9n | 3M | 12 | 1,089* | 76,425 | 9 | flat — wave-10 (brain wave) wall; wave-9 seeds thin (4 states/1 episode). Enriching wave-9 coverage |
| #13 w6e7qz9a | 3M | 12 | 979* | **94,025 ATH** | **10** | wave-9 enriched + double-weighted → second brain wave broken in 1M steps |
| #14 y76tl91z | 3M | 12 | 1,254* | **119,350 ATH** | **11** | frontier pool (w10 double-weighted); six figures |
| #15 24e8bktw | 3M | 12 | 824* | **136,600 ATH** | **12** | matches old python-gym depth (wave 12) on REAL dynamics |
| #16 0gl51bp3 | 983k/3M (host went down ~03:30) | 12 | 768* | **143,950 ATH** | **13** | wave-12 pool triple-weighted; ATHs at 983k. 900k checkpoint preserved |
| #16b tdz2cndb | 3M (resumed from #16's 900k ckpt) | 12 | 1,362* | 85,525 | 9 | rebuilt pool only reached wave 8 (post-reboot recapture), so within-run ceilings are shallow-start artifacts, not regression. Head used to recapture wave 9+ |
| #17 k3ckwc9u | 3M | 12 | 1,185* | 92,500 | 10 | wave-9 pool (w5_23..27) triple-weighted; brain-wave-10 re-broken. Reclimb continues |
| #18 mr72nflt | 3M | 12 | 1,694* | 132,000 | 11 | single wave-10 state (w5_28) 5×-weighted broke the wall by 786k. Capture-script fix: episodes starting at an in-range wave no longer re-save their start state (5 dup "wave-10" states deleted) |
| #19 ir1ivaib | 3M | 12 | 1,270* | **152,450 ATH** | 12 | wave-10/11 frontier pool; reclimb complete — beats pre-reboot ATH 143,950 |
| #20 n1s7mh7b | 3M | 12 | 1,265* | 145,525 | 12 | flat (no wave 13, sub-ATH). Response: wave-12/13 seed enrichment before link 21 |
| #21 xvjqslgj | 3M | 12 | 1,380* | **156,175 ATH** | **13** | enriched wave-12 band (5 seeds, 3×) → wave 13 re-broken by 1.18M; full recovery past pre-reboot peak |
| #22 w5kt6ema | 3M | 12 | 1,665* | 147,950 | 13 | flat at wave-13 boundary. Root cause: pool has NO wave-13 starts (single-env capture: 0 entries in 800 eps) while training envs hit wave 13 every link |

## 2026-06-11 — AUTO-CAPTURE: harvest frontier states from training envs

The dedicated capture script can't keep up with the frontier (wave 13 from a
1-life wave-12 start is a sub-1% event; 0/800 episodes). But the 12 training
envs reach the frontier every link — those moments are now harvested:
`MameRobotronEnv(auto_capture_min_wave=N)` saves a settled full-machine state
whenever an episode ENTERS wave >= N (once per wave per episode, guarded
against mid-death frames). Indices: 100 + rank*8 + (wave-N), so they never
collide with hand-built ladder indices and are only added to the NEXT link's
pool (no load/save races). CLI: `train_mame.py --auto-capture-min-wave 13`.
Link 23 runs with it; harvest [auto-capture ...] lines from the trainer log.
Result: 7 wave-13 states harvested in one link (w5_100,108,116,124,132,180,188)
vs 0 from 800 dedicated capture episodes. Mechanism promoted to standard.

| #23 6aj9lp72 | 3M | 12 | 1,694* | 151,225 | 13 | auto-capture debut: 7 wave-13 seeds harvested mid-training |
| #24 8bfojsuq | 3M | 12 | 1,204* | **187,875 ATH** | **15** | wave-13 starts in pool → +2 waves in one link (14 AND brain wave 15); harvested 9× w14 + 10× w15. Auto-capture indices shift with min_wave — base now env-tunable (MAME_AUTO_CAPTURE_BASE), link 25 uses 200 |

2026-06-12 forensics audit (link 24, first play at waves 13-15): new band
CLEAN — waves 13/14/15: 5,275 deaths, 99% explained-direct, 2 unmapped
suspects total. No missing-sprite signal at the second brain wave. One fix:
Quark variants $4DF2/$4FD5 were in _LIST1_SW (obs) but not ENTITY_TYPES, so
1,077 quark deaths at waves 7/12 read "explained-unmapped" and quark kills
missed the +50 spawner bonus. Added (effective link 26+). $0390 (73 deaths,
wave 12) stays on the watchlist — prior ground truth says effect record.

| #25 f80c865e | 3M | 12 | 1,724* | **203,125 ATH** | **16** | first 200k+ score; harvested 12× w15 + 4× w16 (base 200) |
| #26 mygocn3a | 3M | 12 | 1,883* | **219,800 ATH** | **17** | Quark kill-bonus fix active; harvested 7× w16 + 10× w17 (base 300). ~1 wave/link cadence with auto-capture |
| #27 e9kkbwxq | 3M | 12 | 1,533* | **234,850 ATH** | **18** | harvested 6× w17 + 9× w18 (base 400) |
| #28 jcq8gooo | 3M | 12 | 1,719* | **247,275 ATH** | **19** | past run-128 python-gym score (241,475); harvested 9× w19 (base 500) |
| #29 sbso7qy9 | 3M | 12 | 1,364* | **264,025 ATH** | **20** | fourth brain wave broken; harvested 8× w19 + 1× w20 (base 600). Next milestone: all-backend record 286,300 |
| #30 y7u9q9v1 | 3M | 12 | 1,513* | **276,975 ATH** | **21** | wave 21 broken late; 1× w21 harvested (w5_741). Auto-capture min 20 kept for link 31 (w20 band still thin) |
| #31 87dhxt52 | 3M | 12 | 1,543* | **283,325 ATH** | **22** | harvested w21 (857) + w22 (850); 3k from the all-backend record 286,300 |
| #32 fuy8slt9 | 3M | 12 | 1,684* | **309,500 ATH** | **23** | **ALL-BACKEND RECORD BROKEN** (was 286,300, python gym #97) — and on real ROM dynamics. Harvested 2× w21, 2× w22, 1× w23 (base 900) |
| #33 wkmqvxfy | 3M | 12 | 1,459* | **344,225 ATH** | **25** | +2 waves (24 + brain wave 25); 17 states harvested incl. 12× w24 (base 1000). Quarter-way to wave 100 |
| #34 ubllx7fk | 3M | 12 | 2,102* | **353,200 ATH** | **26** | chain-best ep_rew; ~40 states harvested (mostly w24; w25 1117, w26 1142, base 1100). fps ~430-460 at depth (entity load, not a fault) |
| #35 67ywmiz1 | 3M | 12 | 1,945* | **365,725 ATH** | **28** | +2 waves again; harvested w25-28 incl. w28 seed 1243 (base 1200) |
| #36 va530q4s | 3M | 12 | 2,056* | **379,000 ATH** | 28 | no new wave but +13k score; w28 band enriched to 4 seeds (base 1300) |
| #37 k1cqqfc1 | 3M | 12 | 1,720* | **386,650 ATH** | **29** | wave 29 broken late (seed 1409); 8 more w28 states (base 1400) |
| #38 88ir32b3 | 3M | 12 | 1,760* | **389,075 ATH** | 29 | flat wave; enriched w29 band to 5 seeds (1409,1508,1532,1548,1580). Link 39 weights them 3× for the wave-30 (6th brain wave) push |

## 2026-06-12 — CRITICAL: continuous wave-1 runs reach only wave ~3 (user-asked)

User asked the decisive question for hardware deployment: can the chain head
actually play 1→29 in ONE continuous game? `mame_gym/eval_continuous.py` runs
the head from a true wave-1 boot (full lives, NO save-state resets) to death.

Link-38 head (save-state ATH 389,075 / wave 29), 15 stochastic runs:
**wave reached mean 3.1, median 3, max 5; score mean ~10,200.**

So the 389k/wave-29 ATH is a SAVE-STATE artifact — those episodes start from a
deep state with ~370k score preloaded, play one wave, die. Continuous marathon
ability is ~wave 3 / ~10k. The ladder built **wave specialists, not a marathon
player.**

Root cause (two compounding effects):
1. Reset pool is ~98% deep save states; wave-1 continuous play gets ~2%
   representation across 38 links → catastrophic forgetting of the early game.
2. Save-state episodes START AT 1 LIFE and end on death, so the value function
   never learns lives have FUTURE value. The policy learned myopic
   life-spending ("clear THIS wave at any cost"), which burns the life stack
   fast in a marathon. This is structural, not just forgetting.

Implication for the wave-100 goal: per-wave competence ≠ marathon. Need a
"stitching" strategy. Options to discuss with user (NOT yet acted on):
 A. Final continuous-play phase: heavily weight/exclusive wave-1 full-lives
    starts so the policy re-learns to chain waves + value lives.
 B. Multi-life save states + longer episode horizon so lives gain future value.
 C. Mixed pool: keep frontier competence but add a large fraction of
    full-lives starts (wave 1 + deep) with extended horizon.
 D. Eval-gate every link on continuous wave-1 performance, not just frontier
    reach, so the chain optimizes the metric that matters for hardware.
Tooling added: `mame_gym/eval_continuous.py` (stochastic + `det` mode).
Deterministic confirm (6 runs): mean wave 2.7 — not a sampling artifact.

## 2026-06-12 — PIVOT to continuous-play training (user: "do whatever best for 1-100")

User endorsed the pivot and added the key lever: **1-ups come from score, and
the dominant score source is rescuing civilians** — so a marathon player MUST
collect the family, which the chain never learned to prioritize.

Changes (link C1, warm-started from competent head 88ir32b3):
1. **Civilian-rescue reward** (`mame_robotron_env.py`): +75 per family member
   that vanishes WITH a >=900 score jump (rescue band 1000-5000; Hulk/Brain
   kills score 0 so they don't trigger it). Makes rescue first-class.
2. **Full-lives continuous pool**: ~72% wave-1 boot (2 lives) so the value
   function learns lives have FUTURE value (the structural fix), + a wave-6..29
   retention spread so per-wave competence isn't forgotten.
3. **auto_capture_min_wave=5, base 2000**: rebuild a CLEAN deep-state library
   (full-lives where reached) as continuous reach extends.
Success metric is now `eval_continuous` (wave reached from a true wave-1 boot),
NOT frontier ATH. Link 39 (frontier ladder, had hit wave 30) was stopped to
free compute for the pivot — frontier depth is no longer the bottleneck; the
competent head already knows every wave band, it lacks the marathon glue.

## 2026-06-12 — BUG: save-state index wrapped at 256 (found during pivot setup)

`mame_bridge.py` sent the reset/save state index as ONE byte
(`state_idx & 0xFF`), so any index >=256 wrapped mod 256. Every auto-capture
base >=256 (links 24-39: bases 200/300/.../1600) silently aliased onto low
indices, colliding with each other AND overwriting hand-ladder states (w5_1 is
now wave 16, not 5; w5_12 is wave 29). So the recent "frontier-weighted" reset
pools were loading a collision-corrupted subset — the chain still climbed to
wave 30, which speaks to robustness, but the pool was muddier than logged.
Fix: index now 2 bytes (b2 high, b3 low) in both `mame_bridge.py` and
`robotron_server.lua` (b3 was free — only STEP uses it, as fire-dir).
Roundtrip-tested at index 500. Existing w5_0..253 still load correctly.

## 2026-06-12 — Continuous-play results

| link | warm-start | continuous eval (15 runs from wave-1 boot) | note |
|---|---|---|---|
| (baseline 88ir32b3) | — | wave mean 3.1, score mean 10,205 | pre-pivot wave-specialist |
| C1 rz5g3pf5 | 88ir32b3 | wave mean **3.2**, score mean **14,877 (+46%)**, max 39,925 | civilian reward works: collects far more per wave, episodes 3× longer; 1-up flywheel starting (runs that banked a bonus life at ~25k reached wave 5-6) |
| C2 w9hbrcvb | rz5g3pf5 | wave mean **3.1**, score mean 11,342, max 18,175 | mid-game bridge did NOT help (slight regression). Bottleneck isn't mid-game competence — it's early-game survival |
| C3 h5zb13yl | rz5g3pf5 | wave mean **2.6**, score 9,345 | death penalty -200/-75 made it WORSE (more early deaths). Big terminal spike destabilized, didn't teach caution. REVERTED to -20 |
| C4 i2xxuxxe | rz5g3pf5 | wave mean **3.0**, score 11,588 | pure wave-1 + ent 0.02. Training metrics climbed (ep_rew 3447, ep_len 672 highs) but eval depth flat. 4 links / 12M steps stuck at wave ~3 |
| C5 u0a6z8v4 | i2xxuxxe | wave mean **3.2**, score 11,142 (dist nudged to w4: 9/20) | 1-up bonus didn't break the wall either |

PLATEAU CONFIRMED: 5 continuous links / 15M steps, all mean wave 2.6-3.2.
Reward shaping is exhausted (civilian, bridge, death-penalty, pure-wave-1,
entropy, 1-up bonus). NOT a reward problem.

C5 death forensics (19,938 training deaths, all continuous wave-1):
- deaths peak at WAVE 2-3 (7,347 + 7,753), taper at 4 (3,334), ~0 past 6
- 66% explained-direct, 21% closing — legit deaths to known enemies
- **dominant killer: GRUNT (7,328 of ~13k in-range = 55%)**, then Hulk 1,909,
  EnfBullet/Spark 1,771, Electrode 884, Enforcer 731, Spheroid 717
- ("Mom"/"Mikey" 516 = forensic artifact: died NEAR a civilian it was chasing)

INTERPRETATION: the policy that survives wave 29 from a save state dies to
basic GRUNTS in the sparse early game. Frontier training rewarded aggressive
wave-clearing on DENSE boards; it never learned careful evasion/survival in the
open early game, and warm-started reward-tweaks can't transform that core
behavior. Crucially, continuous-from-wave-1 is likely the SAME low-wave
plateau that motivated the save-state ladder originally — so this is probably
not fixable by reward shaping or more of the same. Real options: (a) from-
scratch continuous (no anchored dense-board habits), (b) architecture —
recurrent policy / richer obs for spatial evasion, (c) much more compute, (d)
accept specialists + rethink deployment. HELD for user direction at this fork.

## 2026-06-13 — ROOT CAUSE FOUND & FIXED: garbage entity velocity in obs

From-scratch continuous (no warm-start) evaled at mean wave 2.8 — SAME wall as
warm-started (2.6-3.2). So not anchoring → representational. Checked the obs:
velocity was computed per distance-RANKED slot (`slot_key=(category, rank)`) in
position_wrapper._extract_features. Enemies reorder by distance every frame, so
each slot's velocity = delta between two DIFFERENT entities = garbage. The
policy literally could not perceive enemy motion — and the death forensics
(dies to predictable GRUNTS in waves 2-3, 55% of kills) is exactly the symptom.
Harmless for the dense save-state frontier (reactive shooting), fatal for
open-early-game evasion.

Fix (commit cc93a12): MameObsBuilder computes velocity by stable node address,
threads (vx,vy) per sprite into _extract_features (prefers it over slot-key
delta; python-gym path unchanged). Verified realistic motion (~13px/step
grunts, ~0 family). C6 = warm-start rz5g3pf5 + fixed obs, pure wave-1. This is
the highest-confidence lead yet — if the wall is the velocity bug, depth should
finally climb. Eval pending.

C6 (warm-start + fixed obs) evaled mean wave 3.2 — flat. Expected: the warm-
started policy already learned to IGNORE the velocity channel (noise), so it
doesn't suddenly use it. Clean test = from-scratch WITH fixed velocity
(scratch2, runs/oiwnq38o): a fresh policy can learn to exploit real motion from
the start. Compare to garbage-velocity from-scratch (mean 2.8): scratch2 > that
⇒ velocity helps; scratch2 also ~3 ⇒ wall is deeper (architecture/difficulty).

## 2026-06-13 — CONCLUSIVE: wave-3 wall is robust to ALL interventions

scratch2 (fresh + fixed velocity) evaled mean wave 2.8 = SAME as garbage-
velocity from-scratch (2.8). Velocity fix is correct but was NOT the bottleneck.

Full continuous-play ablation (eval = mean wave from a true wave-1 boot, 2 lives):
| config | mean wave |
|---|---|
| baseline frontier head (88ir32b3) | 3.1 |
| C1 civilian reward | 3.2 |
| C2 + mid-game bridge | 3.1 |
| C3 heavy death penalty | 2.6 |
| C4 pure wave-1 + entropy | 3.0 |
| C5 + 1-up bonus | 3.2 |
| from-scratch (garbage vel) | 2.8 |
| C6 velocity-fix + warm-start | 3.2 |
| scratch2 velocity-fix + from-scratch | 2.8 |

EVERYTHING caps at wave ~3 (2.6-3.2). Reward shaping, warm-start vs scratch,
entropy, and the velocity fix all fail to move it. The ceiling is fundamental
to this setup: PPO + 945-dim distance-ranked category-slot obs + MLP, on
continuous Robotron from wave 1 with 2 lives. Death mode: grunts in waves 2-3.

Conclusion: NOT a reward/training-recipe problem. The likely culprit is the
OBS REPRESENTATION + policy (the distance-ranked slot encoding scrambles
spatial structure; an MLP can't recover the geometry needed for evasion/
wall-avoidance). Breaking it needs a representational/architectural change
(spatial grid + CNN, or attention over entities, or recurrence) — a real build,
not a knob. HELD for user decision; cheap experiments exhausted.

Death-LOCATION analysis (C5, 19,938 deaths): 63% open field, 34% one-axis-edge,
only 3% corner (LESS corner-clustered than uniform ~6%). The policy is NOT
getting cornered against walls — it dies in the open to converging grunts.
Reframes the representation argument: bottleneck is GLOBAL threat-field
reasoning (where are all threats, which open region is safe), which the
distance-ranked nearest-N slot obs obscures and a spatial grid+CNN would expose.
Strengthens option 1 (CNN); weakens a pure-recurrence fix.

## 2026-06-13 — CNN architecture did NOT break the wall (conclusive)

Built spatial-grid obs (spatial_obs.py, 11x36x24) + GridCNN policy + obs_mode
'grid' in train_mame.py (smoke-tested, committed). One 3M from-scratch run
(cnn1, runs/f2g037u5, CPU ~240fps): continuous eval mean wave **2.4** — NOT
better than MLP (2.6-3.2), slightly worse.

THE WALL IS ROBUST TO EVERYTHING TRIED:
- reward shaping (6 variants), warm-start vs from-scratch, velocity fix,
- OBS REPRESENTATION (945-slot vs 11ch spatial grid),
- POLICY ARCHITECTURE (MLP vs CNN).
All continuous-from-wave-1 runs cap at mean wave ~3 (2.4-3.2). The bottleneck is
NONE of these. Remaining untested invariants: frameskip 4, action space,
2-life budget, PPO itself, or it's the honest model-free ceiling for this task.
Death mode throughout: dies in OPEN field to slow grunts (decision quality, not
control granularity or wall-cornering) → not obviously a frameskip issue either.

Strategic juncture: the cheap + medium levers are exhausted. Real remaining
options are big (different RL algorithm, 5-10x compute scaling, or accept the
save-state specialist for deployment). HELD for user; GPU being set up to make
any further scaling cheap.

## 2026-06-13 — GPU enabled (user flagged the 4080)

nvidia-smi shows the RTX 4080 fully in WSL2 (16GB, driver 610, CUDA UMD 13.3);
only gap was a cpu-only torch (2.12.0+cpu). Installing torch==2.12.0+cu130
(exact version, cu130 matches the 13.3 driver → no sb3/model compat change).
Future runs: --device cuda (helps the compute-bound CNN ~2.5x; MLP stays
env/MAME-bound at ~600fps).

## 2026-06-13 — Long-budget scaling run on GPU (user-approved)

Definitive test of "is wave-3 a sample/optimization limit or a true ceiling."
Scaled BOTH model and data: GridCNN bumped to 32/64/128/128 conv + net_arch
[512,256]; 15M steps (5x the standard link) on the 4080. cnnL, runs/pma8n1ge,
fresh, pure wave-1, civilian+1up reward, ent 0.02. If continuous eval still
caps at ~3 well before 15M, that's strong evidence of a true model-free ceiling
for this task; if it climbs, wave-3 was a sample/optimization limit.
PLATEAU: C1-C4 all wave ~3 despite civilian reward, mid-game bridge, death
penalty, pure-wave-1, entropy. Common thread: policy dies at ~21k, just short
of the 25k first bonus life — so the lives→depth flywheel never ignites. C5
rewards banking a 1-up directly. If still flat, escalate: from-scratch
continuous (warm-start may be a local-optimum trap) or revisit obs/architecture.

After C1-C3 (9M steps) stuck at wave ~3, the deep 1-life retention states are
suspected of sabotaging continuous learning (each teaches single-life myopia).
C4 drops them (pure wave-1 full-lives) + bumps entropy. If still stuck, the
warm-start itself is a local-optimum trap → next is FROM-SCRATCH continuous
training (no frontier baggage), accepting a slower climb.

Diagnosis after C1/C2 (6M steps, depth stuck at ~3): NOT a training-time
problem — a reward-structure one. Death was -20 vs wave-clear ~250*wave (~750)
and rescue ~175, so death was trivially cheap → policy rushes, dies at wave 3-4
BEFORE banking the ~25k bonus life that starts the 1-up snowball. C3 makes
survival pay. If depth still flat, the wall is the warm-start local optimum
(→ try from-scratch or higher entropy) or the policy/obs ceiling.

Read: depth gated by the 1-up flywheel (rescue→25k bonus life→go deeper). C1
turned the flywheel on (score doubled) but depth needs the early-game survival
to become reliable. C2 keeps wave-1-heavy training (the 1→5 chain) + bridges
the wall. Success metric = `eval_continuous` mean wave, NOT frontier ATH.

## 2026-06-11 — REBOOT WIPED /tmp: save-state pool lost, link 16 cut short

Host went down ~03:30 (rebooted 09:14). `/tmp/mame_states` held the ENTIRE
wave-5..12 ladder pool (rl_reset + w5_1..58) — all gone, along with link 16's
trainer (983k/3M steps; metrics recovered from its tensorboard events:
**143,950 / wave 13, both new ATHs**).

Fixes:
- `mame_bridge.py` STATE_DIR now `~/Code/robotron-rl/mame_states/` (persistent;
  `MAME_STATE_DIR` env override). Never tmpfs again.
- `mame_gym/capture_ladder_states.py` — multi-wave recapture: saves at every
  wave entry in [min,max] up to per-wave quota in one driving pass; writes
  `ladder_state_map.json` (index→wave/score/lives). Rebuilt pool with the
  0gl51bp3 900k head from boot (3 lives → deep episodes fill several bands).
- Chain resumes as link 16b from the 899964 checkpoint (precedent: #9b).

## 2026-06-10 — Death forensics (user-requested)

Every life loss now logged with 3-packet history (mutual-destruction aware —
the killer often dies WITH the player, so suspects come from pre-death frames
and "vanished at death" is the prime-suspect flag). Enabled via
MAME_DEATH_LOG_DIR env var; aggregate with `mame_gym/death_audit.py`.

Audit of 215 deaths (vzgyfwgi policy, waves 1-6):
- **81.4% explained-direct** (lethal in contact range; composition: Grunt 66,
  Electrode 34, Brain 31, sparks 16, Spheroid 7, Hulk 6, ...)
- **12.1% explained-closing** (lethal within one frameskip-4 step of mutual
  approach — initial 15u radius ignored ~12u/step closing speed)
- **1.4% explosion-debris at the site** ($00xx records = the killer's corpse;
  mutual destruction confirmed from the forensic side)
- **5.1% unexplained** — pattern matches fast enforcer sparks (~16u/step)
  crossing the 4-frame sampling gap from >40u out. NOT invisible-enemy shaped.

Conclusion: no evidence of unmapped enemy types in waves 1-6. Tank/Quark/
TankShell (wave 6+) still pending census. Forensics stay ON for all future
links — `explained-unmapped` verdicts are the standing missing-sprite alarm.

## 2026-06-10 — CLASSIFICATION CORRECTION (user caught Prog-on-wave-2)

User spotted a death-audit record showing a "Prog" suspect on wave 2 —
impossible (Progs only exist on Brain waves 5/10/15…). Stop-the-line per user
instruction. Ground truth recovered from `dumps/ENEMY_BEHAVIOR_ANALYSIS.md`
(ROM reverse-engineering) + the Windows bot's `game_state.py`:

1. **State-words are death-handler code addresses — constant for life.**
   Live entities NEVER change SW. The census "animation variant" ranges
   ($3A45-$3A86 etc.) were death-explosion/effect records — my range
   classifier was painting corpses as live enemies and suppressing real
   kill bonuses. REVERTED to exact-match.
2. **$0390 is not Prog** (unknown effect record; dropped).
3. **Progs have no unique SW**: brain-mutated civilians show $1F1F in the
   slot pool (same as Cruise Missile); standalone Progs use $00B6 (same as
   Hulk). Progs were therefore ALWAYS visible to the policy, aliased to the
   right threat class. No obs change needed beyond reverting the ranges.
4. Brain waves are every 5th wave (5, 10, 15, …) — ladder intel.

Lesson recorded: never trust inherited type tables — verify against ROM
ground truth or controlled measurement. Both pre-existing sources
(analyze_dumps.py's $0390:Prog, the census range inference) were wrong in
ways that poisoned training.

Chain restart: link 7 (vzgyfwgi) trained on phantom-entity obs — NOT used as
warmstart. Link 8 restarts from arubzxev (last clean-obs head).

## 2026-06-10 — LIST-WALK OBS (user suggested using the commented ASM)

Downloaded Scott Tunstall's full commented disassembly (robomame.asm,
21k lines → /tmp/robomame.asm; re-fetch from seanriddle.com/robomame.asm).
It documents the game's own per-category entity linked lists — the
authoritative classification source:

| Head | List | Members |
|------|------|---------|
| $9817 | list 1 | spheroids, enforcers, quarks, sparks, TANKSHELLS |
| $981F | list 2 | family members |
| $9821 | list 3 | grunts, hulks, brains, progs, cruise missiles, tanks |
| $9823 | list 4 | electrodes |
| $981B | — | free-object list (dead entries) |

Node layout: +0/1 next, +4 display X, +5 display Y, +8/9 state word.
(Resolved a long-standing offset puzzle: our $98D4 "slot pool" view reads
real object records at +4 — pool base is $98D0.)

New obs pipeline (`robotron_server.lua` walk_lists + `mame_obs.iter_entities`):
- Category by list membership — corpse/effect-free BY CONSTRUCTION (dead
  objects unlink to the free list)
- TankShell/Quark/Tank classified without knowing their SWs (list+SW table;
  unknown SW on list 1 ⇒ TankShell) — wave 6+ ready
- Prog vs Hulk ($00B6) and Prog vs CruiseMissile ($1F1F) split by movement
  signature (Prog ~14 u/step X-dominant vs Hulk ~3 u/step / missile
  Y-dominant), keyed by node address (stable lifetime identity).
  Behavioral stakes (user-flagged): Hulks invincible-avoid vs Progs
  shoot-on-sight.
- Validated live: wave 1 = exactly 15 grunts/5 electrodes/Mom+Dad, zero
  noise; wave 5 = 15 brains + 16 civilians + spheroid. 

Link 8 v3 running from arubzxev with this obs + wave-5 pool + forensics.

## 2026-06-10 — Visual verification + Prog ground truth (user-reviewed)

User reviewed 15 random overlays (waves 1-5): ALL GOOD. Targeted captures
(visual_verify_targeted.py) added Enforcer/Spark/CruiseMissile verification —
including a frame of an Enforcer materializing inside its Spheroid. All
labels on correct sprites.

Prog hunt (user noted no spawned-enemy verification): 12k steps of wave-5
play produced ZERO progs. Root cause found in robomame.asm:
- CREATE_PROG at $1E19 sets the prog's state word to **$1F1F — identical to
  Cruise Missile** (confirmed at source line 1E46).
- $2119 = Brain-in-programming-state (2,613 node-frames observed = ~100
  programming sessions STARTED). Programming takes ~20 frames of the brain
  standing still; our policy kills the brain before CREATE_PROG completes.
  Conversions begin constantly but never finish under fire — which is
  correct play (killing the programming brain saves the civilian).
- Resolution: $1F1F typed as CruiseMissile (threat-correct for both; velocity
  features carry the motion difference). $00B6 Hulk-vs-standalone-Prog keeps
  the velocity heuristic (behavior-critical: invincible vs shootable).
- Verification artifacts: /tmp/visual_verify*/, tools permanent, rerun per
  wave band. Pairs double as YOLO dataset.

**Chain head: `models/f3y6eg9k/`.** For perspective: the native gym needed
warmstarts + wave-4 snapshot rotation + 6 chain links to reach wave 5; the
MAME gym got there in one fresh 3M run on faithful dynamics.

This file is the autonomous loop's source of truth across wakeups while the user is away (2026-05-29 → ~2026-06-05).

## Current status

**Mode:** RESUMED — chain v3 link 7 in progress (2026-06-07). Wandb post-mortem
of the corruption event revealed the link was already failing (peak ep_rew_mean
2,249 at step 655k, then plateaued at wave-4/score-23,550 for ~1.4M steps before
the corruption fired) — see `/home/strider/Code/robotron_native/current_issues.md`.
Resuming with new instrumentation: guard now records which condition fired
(`wave_jumped`, `lives_inflated`, `score_exploded`, `oob`) plus pre/post
wave/lives/score-delta context. If corruption reproduces, we'll know the cause.
Pre-flight: `/tmp/cap_*_6809.bin` snapshots rebuilt via
`/home/strider/Code/robotron_native/tools/extract_6809_snapshots.py` (they were
wiped by reboot).

**Chain head:** `models/iwouosoi/` (peak 21,675, ep_rew_mean 2,767, highest_wave 4) — first post-promotion link.

**Promoted recipe (v3 after experiment #8):** fresh training initial 3M, then chain with gamma=0.999 AND 3M-step links. HPs lr=3e-4, clip=0.2, ent_coef=0.01, **gamma=0.999**, **timesteps=3,000,000 per link**. Reward shaping = base score_delta/10 + survival + wave bonuses + death penalty + spawner_kill_bonus (50) + shooter_kill_bonus (20). Mixed snapshot rotation cap_001..020.

**Why the chain regression happened:** the python-gym-warmstart policy learned behaviors valid in the python re-implementation but not in the real ROM. Fine-tuning from it kept it stuck in a degraded attractor (~1,800 ep_rew_mean for 30+ runs). Training from scratch on native dynamics found a genuinely better policy in 3M steps.

**Paused chain head (kept as fallback only):** `models/vm9hy8gw/` (ep_rew_mean 1,481, wave 3 — last python-warmstart run before pause).

## Experiment queue & results

| # | Hypothesis | HPs | Reward | Snapshots | Result | Verdict |
|---|------------|-----|--------|-----------|--------|---------|
| 1 | snapshot rotation breaks recipe | scratch | base | cap_001 only, 300k | highest_score 7k, wave 2 | ❌ WORSE |
| 2 | HPs too aggressive | fine-tune (lr=5e-5, clip=0.1, ent=0.005) | base | mixed, 1M | ep_rew_mean 1,550 still below floor, clip_fraction 0.197 | ❌ no help |
| 3 | reward signal too thin (no spawner/shooter bonuses) | scratch | +50/spawner +20/shooter on kill | mixed, 1M | ep_rew_mean 1,582, highest_wave 3, highest_score 19,650, explained_var 0.796 (chain high) | ❌ no help |
| 4 | exploration vs exploitation (target_kl) | scratch + target_kl=0.02 | base | mixed, 1M | ep_rew_mean 1,495, highest_wave 3, highest_score 19,450, explained_var 0.833 (chain high) | ❌ no help |
| 5 | longer horizon valuation (gamma=0.999) | scratch + gamma=0.999 | base | mixed, 1M | ep_rew_mean 1,733 (highest of experiments), highest_wave **4** (met!), highest_score 18,550, clip_fraction **0.046** (3-4× lower than chain), KL **0.0066** (2-3× lower) | ⚠️ PROMISING but ep_rew_mean miss — strongest near-miss so far. Worth re-exploring with a 2nd 1M iteration or compound with another change. |
| 6 | game-dynamics gap (fresh train, no warmstart) | scratch (lr=3e-4 clip=0.2 ent=0.01) | base + spawner/shooter bonuses (still in code from #3) | mixed, 3M | ep_rew_mean **2,767**, highest_wave **4**, highest_score 21,675, ep_len_mean 3,685 (chain-best), 0 corrupted, sparklines show steady monotonic learning | ✅ **PROMOTED** (but chain links 1-3 decayed; chain re-paused 2026-05-30 link 3) |
| 7 | gamma=0.999 + spawner/shooter bonuses from #6 checkpoint | iwouosoi 3M checkpoint warmstart + gamma=0.999 | base + bonuses | mixed, 1M | ep_rew_mean **2,394**, highest_wave **4**, highest_score 21,475, ep_len_mean 2,869, death_eps 320 (low), sparkline monotonic climb | ✅ **PROMOTED** — gamma=0.999 is the sustainer (but chain v2 also decayed at 3 links, just at higher values) |
| 8 | 3M-step links instead of 1M (chain decay is a horizon issue) | 6oxntz8g checkpoint warmstart + gamma=0.999 | base + bonuses | mixed, 3M | ep_rew_mean **2,134**, highest_wave 4, highest_score **24,250** (chain high), ep_len_mean 2,462, clip_fraction 0.215, KL 0.022 | ✅ **PROMOTED** — 3M-step links hold above floor (1M-step chain v2 tripped at same starting point) |

## Promotion rules

A run is "promising" if **ep_rew_mean ≥ 1,900 AND highest_wave ≥ 4**. If experiment N is promising:
- Treat it as the new recipe. Resume normal chain monitoring with that recipe.
- The wakeup transitions to chain-mode: standard deploy-to-Xenia + restart-from-checkpoint cycle.

If experiment N is not promising:
- Move to next experiment.

If all queued experiments exhausted without promotion:
- Continue running the BEST recipe found in a chain pattern (don't grind on worst recipe).
- User will return and decide further.

## What NOT to do

- Do NOT switch back to python gym (`train_progressive.py` / `robotron2084gym/`). User explicit instruction 2026-05-26.
- Do NOT make large architectural changes without an experiment to test.
- Do NOT chain-train the recipe that triggered pause (chain HPs + base reward) — that's what just failed.

## Native chain (post-promotion)

| Link | Run ID | ep_rew_mean | highest_score | highest_wave | clip_fraction | KL |
|------|--------|-------------|---------------|--------------|---------------|------|
| 0 | iwouosoi (3M scratch) | 2,767 | 21,675 | 4 | 0.395 | 0.044 |
| 1 | kc8471u4 | 2,688 | 19,150 | 4 | 0.387 | 0.051 |
| 2 | qx19jigr | 2,254 | 18,900 | 3 ↓ | 0.397 | 0.052 |
| 3 | srnkussl | **1,707** ↓ TRIPS | 20,400 | 4 | 0.395 | 0.047 |

**Chain v1 paused at link 3 (2026-05-30 ~02:08).** Same decay shape as python-warmstart chain. gamma=0.995 not enough horizon. Going to experiment #7 → gamma=0.999.

## Chain v2 (gamma=0.999, post-promotion experiment #7)

| Link | Run ID | ep_rew_mean | highest_score | highest_wave | clip_fraction | KL |
|------|--------|-------------|---------------|--------------|---------------|------|
| 0 | tzkpezzx (gamma=0.999 from iwouosoi 3M) | 2,394 | 21,475 | 4 | 0.376 | 0.045 |
| 1 | 6oxntz8g | 2,074 | **23,400** ↑new-high | 4 | 0.283 | 0.030 |
| 2 | kiey76kd | **1,939** ↓ TRIPS | 21,650 | 4 | 0.238 | 0.031 |

**Chain v2 paused at link 2 (2026-05-30 ~02:58).** Same decay shape as v1 — gamma=0.999 shifted equilibrium up but didn't structurally fix the 1M-step chain decay. The 1M-step chain link length itself may be the problem. Going to experiment #8 → 3M-step links.

## Chain v3 (3M-step links, gamma=0.999, post-promotion experiment #8)

| Link | Run ID | ep_rew_mean | highest_score | highest_wave | clip_fraction | KL |
|------|--------|-------------|---------------|--------------|---------------|------|
| 0 | a3lye43q (3M from 6oxntz8g) | 2,134 | **24,250** ↑ | 4 | 0.215 | 0.022 |
| 1 | eufd7e0y | **3,270** ↑↑ (+53%!) | 20,925 | 4 | 0.369 | 0.050 |
| 2 | 8tjxtuii | 2,310 (dipped from peak, above floor) | 23,000 | 4 | 0.401 | 0.063 |
| 3 | 9b755rcf | 2,098 (tight, 98 above floor) | 22,175 | 4 | 0.321 | 0.041 |
| 4 | 6fr2ydbv | 2,299 (recovered, 299 above) | **24,975** chain high holds | 4 | 0.305 | 0.045 |
| 5 | oovasxbd | 2,039 (razor thin, 39 above) | 21,225 | 4 | 0.344 | 0.053 |
| 6 | nmhj01ud (KILLED at ~2.1M) | — | — | — | — | — |
| 7 | idzzb9re (instrumented, resumed from oovasxbd 2026-06-07) | **2,120** (120 above floor) | 20,200 ↓ | 4 | 0.355 | 0.045 |

**Chain v3 HALTED at link 6 (2026-05-30 ~08:16) due to corruption_eps=1 mid-run.** Per cron protocol's explicit ALERT rule. Findings written to `/home/strider/Code/robotron_native/current_issues.md`. **No auto-restart** — user investigation needed. First corruption in ~28M+ cumulative native steps since the May 26 native-gym fix. Could be rare edge case in the wave-transition path or random state combination tripping a guard threshold.

**Chain v3 link 7 (2026-06-07) — RESUMED with instrumented guard, ran CLEAN.** No
corruption reproduction in 3M additional steps. All four per-cause counters
(wave_jumped / oob / lives_inflated / score_exploded) stayed at 0. Confirms
the link 6 event was a true 1-in-~30M edge case, not a recurring failure.
However, `highest_score` decayed across recent links (24,975 → 21,225 → 20,200)
and wave-4 ceiling is locked in — the chain v3 recipe is **plateaued**.
ep_rew_mean is hovering just above the 2,000 floor. Time to design
experiment #9 to escape the wave-4 ceiling.

## Experiment queue — Phase 2 (in progress)

| # | Hypothesis | Variable | Result | Verdict |
|---|------------|----------|--------|---------|
| 9a | Wave-5+ bonus too small | +3000/+8000/+20000 (was +1000/+3000/+8000) | ep_rew 1,808 ↓, score 19,075 ↓, wave 4 | ❌ WORSE — sparse bonus distortion |
| 9b | Snapshot pool over-weights wave 1 | cap_010/015/020 only (drop cap_001/005) | ep_rew **2,228** ↑, score **23,225** ↑, wave 4 | ✅ **PROMOTED** (best metrics in 4 links, wave-4 ceiling unchanged) |
| 9c | Force every episode into the wave-3→4 transition window | 10 self-captured late-wave-3 snapshots (`/tmp/cap_late_w3_001..010_6809.bin`, wave=3 score=12k-12.5k); generated via `/home/strider/Code/robotron_native/tools/capture_wave4_snapshots.py` driving idzzb9re | ep_rew **2,940** ↑↑ (+32%), ep_len **2,950** (+34%), score 23,650, wave 4 | ✅ **PROMOTED** — biggest ep_rew jump in chain v3. Wave-4 ceiling unbroken but policy now plays the wave-3 endgame much harder. |

**Diagnostic from 9a+9b:** the wave-4 ceiling is a **capability gap, not a coverage gap**. Skewing to wave 2-3 starts (9b) improves wave 1-4 exploitation but doesn't produce wave-4→5 transitions to learn from. Bumping bonuses (9a) distorts value targets without giving the policy more wave-5 samples. To break the ceiling we likely need either: direct wave-4+ training exposure (9c — needs tooling), or a fundamentally different exploration mechanism / policy structure.

## Chain v3 promoted head (post-9d) — WAVE-4 CEILING BROKEN

**Head:** `models/h4gb05w1/` (highest_wave **5**, highest_score **36,775** +48% over chain v3 record, ep_rew_mean 2,110, ep_len_mean 1,460, ev 0.942, corrupted_eps 0). Snapshots: 10× cap_wave4 (wave=4 score=16-22k, captured AFTER 300-step settling delay so transition completes). 9d = chain link from `05ejjc3u` with wave-4 starts. Run 2026-06-08.

**Critical fix during 9d:** the first wave-4 snapshot capture (without settling delay) produced *broken* snapshots — game caught mid-wave-transition, loaded snapshots auto-advanced waves silently without spawning enemies. Player became immortal, no episodes ended in initial 9d run. Killed and re-captured with 300-step delay post-arrival. Fixed snapshots produce clean wave-4 gameplay (verified: player dies normally, score progresses).

**Chain v3 progressive starts trajectory:**
| Link | Run ID | Starts | ep_rew_mean | highest_score | highest_wave | ep_len | ev |
|------|--------|--------|-------------|---------------|--------------|--------|-----|
| 9c   | 0e12g4zj | wave-3 | 2,940 | 23,650 | 4 | 2,950 | 0.699 |
| 9c.2 | 05ejjc3u | wave-3 | 3,510 | 24,950 | 4 | 3,230 | 0.917 |
| 9d   | h4gb05w1 | wave-4 | 2,110 | **36,775** ↑↑ | **5** ↑ | 1,460 | 0.942 |
| 9d.2 | wh4rfkdb | wave-4 | 2,200 | **41,850** ↑ | 5 | 1,540 | 0.969 |
| 9d.3 | 1ezrikey | wave-4 | 2,030 | 34,850 ↓ | 5 | 1,380 | 0.978 | NOT PROMOTED — regression vs 9d.2. Wave-4 chain saturated. |

**9e attempt (wave-5 starts) failed:** wave-5 snapshots captured with 300-step delay produced stuck game state (no score change, no death, no progression in 3000+ steps even with the chain head policy). Tried 1500-step delay — still stuck. Wave-5 game state has internal state outside the pin region that doesn't reliably reproduce from snapshot. Bug not yet root-caused.

## Session 2026-06-08 — full attempts to break wave-5 ceiling (ALL FAILED)

Chain head `wh4rfkdb` (41,850 / wave 5) remains the leader after 6 consecutive
failed breakthrough attempts. Wave-5 is a structural ceiling needing different
approach.

| Exp | Run | Strategy | Score | Δ vs 41,850 | Verdict |
|-----|-----|----------|-------|-------------|---------|
| 9d.3 | 1ezrikey | wave-4 chain L3 | 34,850 | -17% | chain saturated |
| 9e   | killed   | wave-5 starts | broken | n/a | snapshot stuck-state bug |
| 9f   | u2bnabfj | wave-3+4 blend | 35,450 | -15% | sideways (ep_rew +51% but score -15%) |
| 9g   | 2o8hv1dg | late wave-4 (score≥25k) | 35,550 | -15% | same plateau |
| 9h   | 2qwtrmaw | + brain-kill bonus | 35,850 | -14% | value func at ev=0.99, saturated |
| 9i.v2 | wzoo6u66 | lives=5 | 41,250 | -1.4% | almost matched record but no break; ev=1.0 saturated |
| 9j   | vmo3iiac | 6M single run (2x duration) | 39,050 | -7% | full convergence (ev=1.0, value_loss≈0); longer training didn't break through |

**Wave-5 corruption fix attempted 2026-06-09 — FAILED.** Five patches tried
(see `/home/strider/Code/robotron_native/wave5_snapshot_bug.md` for details):
- $9859=0, $9843=0, $9885=$A55A snapshot patches — all stuck
- 2-IRQ-per-step in runner.cpp — broke wave-4 too
- PC=$D19D hook forcing $9810=2 — CPU lands in unimplemented opcodes

Bug requires the deeper save-state implementation (CPU regs + RAM, ~half-day
work). Current code reverted to baseline. Training pivoted back to wave-4
late captures via 9j; same plateau pattern.

**Fixes landed this session (will benefit any future work):**
- Snapshot-export tool: `/home/strider/Code/robotron_native/tools/capture_wave4_snapshots.py` (wave-N capture with settling delay, score gate)
- Brain-kill bonus: +100 reward per Brain (SWs 0x1DD6, 0x2119) — wave-5+ exclusive enemy
- Corruption guard threshold: `lives_inflated` raised 5→10 (1-up bonuses can legitimately push lives to ~7-9)
- Wave-transition snapshot capture pattern: 300-step settling delay works for wave 4, wave 5 needs deeper fix

**For next session / direction:**
The wave-5 ceiling is structural. Real options:
1. **Investigate the wave-5 snapshot stuck-state bug deeply** — likely a pin-range issue or unsaved state pointer. Read native emulator code, identify which RAM bytes need to be set after load.
2. **FSM-driven captures** — port `robotron_fsm.py` to native gym, use FSM (which can play deep waves) to generate diverse mid-wave-5/6/7 snapshots.
3. **Larger policy** — MLP 512x512 may be at capacity for wave-5+ enemy combinations. Try larger or recurrent net.
4. **Much longer single-run training** — value function is currently saturating in 3M steps; try 10M+ in a single run without chaining.
5. **Lower lr or higher ent_coef** — current saturation suggests policy converged to local optimum. Restart with higher exploration.

**Prior heads:**
- `models/05ejjc3u/` (9c.2 — wave-3 starts, 3,510 / 24,950, wave 4 cap)
- `models/0e12g4zj/` (9c)
- `models/xj4j627d/` (9b — wave 2-3 starts, 2,228 / 23,225)
- `models/idzzb9re/` (link 7 — full 5-snapshot, 2,120 / 20,200)

## File pointers

- `train_native.py` — modified 2026-05-29 to add spawner/shooter kill bonuses (SPAWNER_SWS, SHOOTER_SWS, `_count_spawners_shooters`). Toggleable via removing the bonus block in step().
- `models/vm9hy8gw/` — paused chain head, used as warmstart for all experiments.

## 2026-06-13 — ROOT CAUSE OF THE WAVE-3 WALL FOUND (user's insight)

User: "a RANDOM agent gets past wave 3; first real difficulty is ~wave 5." Tested
it: random agent caps at mean wave 2.3 (max 3) — SAME wall as every trained
agent (2.4-3.2). The wall is AGENT-INDEPENDENT => env/reward bug, not a ceiling.
This invalidates the entire "model-free ceiling" conclusion; reward/obs/arch/15M
scaling all failed because the reward was broken.

Traced it: the env detected deaths from the `lives` counter decreasing, but the
lives register reads spurious transient values for ~15 steps after EVERY wave
entry (e.g. lives 1->2->1 entering wave 2, dead-flag 0 the whole time). So every
wave advance fired a FALSE -20 death penalty (and post-C5 a false +250 1-up on
the up-flick). The agent was punished for advancing waves -> learned not to.
Real deaths also lagged: lives decrements ~26 steps after the $9848 dead-flag
rises.

FIX (commit pending): real death = `lives decremented AND dead-flag set` (clean
conjunction — lives-flickers have dead-flag 0; dead-flag flickers have no life
loss). No reward during death frames; -20 once per real death; dropped the
lives-based 1-up bonus. Verified: 2 deaths/episode (=2 lives), zero false
alive-frame penalties. NEXT: retrain with fixed reward — expect the wall to
finally break. Also pending: per-major-checkpoint VIDEO recording (user wants to
watch progress) + revisit lives count (only 2; arcade default is 3).

## 2026-06-13 — Reward fix necessary but NOT sufficient; deeper diagnosis

After the game_state reward fix, a fresh MLP run (28v2tmxk) mid-eval at 1.77M =
mean wave 2.9 — still walled. Key: the reward bug only affected TRAINED agents,
but a RANDOM agent also caps at mean 2.3 (user's insight), so the env is hard
for ALL agents in the early waves — untouched by the reward fix.

Diagnostics (note: the old death-forensics read the $98D4 "slot pool" which the
ASM audit shows overlaps font-render memory $98D0-$98D2 — likely GARBAGE; the
945-dim obs uses the validated typed-entity list-walk and is fine):
- Killer (re-run with CORRECT typed-entity data, at the exact game_state==0x1B
  death instant): GRUNTS at close range (8-14u) in the OPEN middle of the field.
- DIR table in robotron_server.lua is CORRECT (dir6=down+left, dir7=left, etc.);
  movement works. Earlier "broken directions" were wall/drift test artifacts.
- Player vs grunt speed ratio: UNRESOLVED (clean test kept failing on port/CPU
  contention with training). Xenia validation said 25 vs 2 u/s (12x), but a
  noisy test suggested only ~2-4x. Re-test cleanly after training frees CPU.

Open question: why do all agents die to slow grunts in the open early game?
Hypotheses to test next: (a) player speed actually low (control/frameskip),
(b) early game just needs more training on the CORRECTED reward (all prior
training was on broken reward — chain several fixed-reward links), (c) 2 lives.
TODO: audit/fix the $98D4 slot pool (kill-bonus + forensics read it).

## 2026-06-13 — fixed-reward eval + clean speed test

fixed1 (28v2tmxk, fresh MLP, corrected game_state reward, 3M): final eval mean
wave 2.5 (training highest_wave hit 6, but modal play still dies wave 2-3).
So the reward fix is CORRECT but did NOT improve the eval mean — trained STILL
≈ random. The wall is an ENV property, not reward/training.

Clean speed test (reset-based, mame_gym/diag_movement.py):
  player up/down 4.0 u/step, left/right 2.0 u/step; grunt 0.75 u/step.
  => player ~3-5x faster than grunts (enough to evade). The 2x vert/horiz is a
  unit-scale artifact (x spans ~133 units, y ~199, over a 292x240 screen ->
  uniform physical speed), NOT a movement bug. DIR mapping confirmed correct.

So: movement works, player outpaces grunts, reward fixed — yet all agents die
to grunts at wave 2-3. Env-bug candidates exhausted from memory traces.
Recording gameplay videos (videos/) for the user to watch and spot the issue
(their random-agent insight was the key unlock). Still-open: whether early game
is genuinely this hard for model-free RL with 2 lives + frameskip-4, or a subtle
issue a human will see in the video.

## 2026-06-13 — Entropy fix + verified rewards + high civilian reward

User observation ("plays basically random") -> checked policy entropy: fixed1
(ent_coef 0.02) ended at 3.81/4.16 = 92% of fully random. The value function
learned (explained_var 0.83) but the entropy bonus kept the policy from
committing. ROOT of the "trained ≈ random" puzzle.

fixed3 (ent_coef 0.005, warm-start fixed1): entropy fell to 3.25, eval mean wave
2.5 -> 2.9, scores up (max 21,400). The policy commits and plays better.

Overlay video (record_video_overlay.py, now with a reward-event stack) verifies
end-to-end on a wave-3 run: perception boxes match sprites (Hulk/Elec/Enfo
labels correct), action arrows render, and the reward stack shows the RIGHT
events firing — WAVE_CLEAR +500/+750, RESCUE +250 (the corrected civilian count
works!), spawner_kill +50, death -20. Videos in videos/.

Also fixed: _count_strategic read the $98D4 font-memory garbage -> now uses the
typed entity list; civilian rescue 75 -> 250 (user-directed). fixed4 (warm-start
fixed3, ent 0.005, high civilian reward) launched to push civilian gathering.

## 2026-06-14 — Entropy fix worked; fixed5 multi-change regressed; back to basics

fixed1(ent 0.02): policy 92% random (entropy 3.81). fixed3(ent 0.005): entropy
3.25, eval mean 2.5->2.9, policy commits — USER CONFIRMED purposeful play in the
overlay video (moves+shoots at enemies, though shots off-center/miss and walks
past civilians).

fixed5(ent 0.002 + norm_reward + civilian 250, ALL AT ONCE): entropy 2.77,
value_loss 8000->11 (mostly a normalization scale artifact), but eval REGRESSED
to mean 1.9 with 7/25 wave-1 deaths — civilian reward 250 caused suicidal
human-chasing into grunt clusters; lesson: change one variable at a time.

fixed6: warm-start fixed3 (models/28v2tmxk), proven config (ent 0.005, NO
norm_reward), civilian reward 75->150 only (moderate). OUTPUT_DIR models/z686auz7
(run-id collision now logged to avoid evaling the wrong model — fixed5's real
model was models/7g4cvw3e, not the shared 28v2tmxk = fixed3).

Open behaviors (user, from video): shot alignment (off-center, shots miss ->
candidate: on-axis aiming bonus), civilian collection (should improve with the
reward + training). Path: the setup is finally correct end-to-end; chain
proven-config links + tune civilian reward gently.

## 2026-06-14 (overnight) — fixed6 stable; fixed7 adds aiming reward

fixed6 EVAL (25 continuous wave-1 runs, stochastic): **mean wave 2.7**, median 3,
max 4, score mean 8865 / max 21900, dist {2:9, 3:14, 4:2}. RECOVERED from fixed5's
1.9 regression — moderate civilian reward (150) is stable, no suicidal chasing,
and scores are HIGHER than fixed3 (the rescues now pay off). Milestone video:
videos/milestone_fixed6.mp4 (711 frames, full game, stochastic).

But fixed3 (2.9) -> fixed6 (2.7) shows the proven config has PLATEAUED at ~2.8 —
more training of the same recipe isn't breaking it. So fixed7 introduces the one
targeted change the user motivated from the videos: an AIMING REWARD.

fixed7 = warm-start fixed6 (models/z686auz7) + aiming reward, else identical
(ent 0.005, no norm_reward, civilian 150, lr 3e-4, gamma 0.999, 12 envs, 3M).
  - _compass_dir(dx,dy): player->threat vector -> game fire dir 1..8 (unit-tested,
    all 8 octants pass).
  - _nearest_threat_dir(): nearest non-family entity within chebyshev 70.
  - reward += 0.5 (parts["aim"]) per active step when fire == nearest-threat dir.
  - Rationale: agent fires EVERY step (no no-fire action); the lever is DIRECTION.
    User saw "stays just off-center, all shots miss" — this rewards on-axis fire.
  - Smoke-test: random agent triggers aim bonus 18/120 (~the 1/8 baseline); a
    trained policy should drive it much higher. No crash.
OUTPUT_DIR: models/usib1jk3 (tensorboard collides on runs/28v2tmxk — eval the
final_model.zip from models/usib1jk3, NOT the bc-checkpoint dir).

## 2026-06-14 (overnight) — fixed7 AIMING REWARD WORKED; chaining as fixed8

fixed7 EVAL (25 continuous wave-1 runs, stochastic): **mean wave 3.0** (NEW BEST),
median 3, max 4, score mean 9983 / max 15000, dist {2:3, 3:20, 4:2}.
  vs fixed3 2.9 / fixed6 2.7. The aiming reward (+0.5/step for fire on-axis with
  the nearest in-range threat) is a clear KEEPER:
    - distribution tightened hard: only 3/25 early (wave-2) deaths vs fixed6's
      9/25; 20/25 now reliably reach wave 3. A more CONSISTENT player.
    - mean score up (9983 vs 8865) — shots connect now, so kills/wave rose.
  Wave-4 ceiling still holds (2/25), so depth past w3-4 is gated by SURVIVING the
  denser waves, not by aim. Milestone video: videos/milestone_fixed7.mp4.

fixed8 = chain fixed7 (models/usib1jk3) forward, SAME reward (aiming kept), proven
config (ent 0.005, no norm_reward, civilian 150), 3M. Pure continuation — tests
whether the now-aim-capable policy compounds with more training toward wave 5+
BEFORE adding the next lever. One-variable discipline: if fixed8 plateaus at ~3.0,
fixed9 adds a survival/evasion lever (the wave-4 ceiling implies dense-wave
survival is the bottleneck — candidate: threat-proximity penalty, mirroring the
FSM's CLOSE_MOVE spacing heuristic).

## 2026-06-14 (overnight) — fixed8 depth-plateau confirmed; fixed9 adds EVASION

fixed8 EVAL (chain fixed7, no reward change, +3M): **mean wave 3.0** (SAME depth as
fixed7), median 3, max 4, dist {2:4, 3:18, 4:3}, score mean **12678** (up from
fixed7's 9983, +27%) / max 17950. Milestone video: videos/milestone_fixed8.mp4.
  -> Two consecutive links at depth 3.0 = DEPTH PLATEAU CONFIRMED. Pure chaining
     compounds SCORE (better intra-wave kills/rescues) but NOT depth. Confirms the
     wave-4 ceiling is gated by SURVIVAL in dense waves, not aim or scoring.

fixed9 = chain fixed8 (models/m1mc8nbc) + EVASION penalty (one new variable):
  - _nearest_threat() now returns (dist, dir); feeds both aim bonus and evasion.
  - reward += -1.5 * (1 - d/20) when nearest threat within _DANGER_RADIUS=20 (a
    graduated imminent-CONTACT penalty, max at contact, 0 at the 20u edge).
  - Deliberately TIGHT (just above _KILL_RADIUS=15): a wide radius would punish
    being in crowded deep waves (the marathon goal) and steer the policy AWAY
    from depth. This only fires one step from death -> teaches last-moment dodge.
  - Smoke-test (fixed8 policy, 400 steps): danger fired 14% of steps, total -19.6
    vs +1555 positive (score 620 / rescue 300 / wave_clear 500 / survive 135) —
    gentle, learnable, NOT constant (so not a crowding/anti-depth penalty). ~half
    a death's worth per episode.
  - If fixed9 doesn't lift depth, fixed10 increases _PROX_PENALTY; if it hurts
    (timid play, lower score), revert and try scaled survival reward instead.
OUTPUT_DIR: models/zzdxb3cn.

## 2026-06-14 (overnight) — fixed9 EVASION pushed the depth ceiling; chaining as fixed10

fixed9 EVAL (chain fixed8 + evasion penalty): **mean wave 3.1** (new best), median 3,
max 4, score mean 13641 / **max 26100** (new high), dist {2:7, 3:9, **4:9**}.
  vs fixed8 {2:4, 3:18, 4:3} score 12678. The evasion penalty WORKED on depth:
    - wave-4 reaches nearly TRIPLED: 9/25 (36%) vs 3/25. Top score way up.
    - BUT distribution went bimodal: wave-2 deaths rose 4->7/25. More deep runs
      AND more early deaths — reads like an evasion policy still consolidating
      (some runs dodge into w4, others over-focus on dodging / back into walls &
      die early — the wall-backing the user flagged). Net positive: ceiling
      pushed, mean + top score up. Milestone video: videos/milestone_fixed9.mp4.

fixed10 = chain fixed9 (models/zzdxb3cn) forward, SAME reward (evasion kept), proven
config, 3M. Pure continuation — let evasion mature and hopefully collapse the
wave-2 variance into consistent w4+. If the bimodality persists after chaining,
that's a real tradeoff (penalty may cause wall-backing) -> fixed11 would soften
the penalty near walls or cap it. If depth keeps rising, keep chaining toward w5.
OUTPUT_DIR: models/ceqasoqt.

## 2026-06-14 (overnight) — fixed10 CONSOLIDATED evasion (variance collapsed); chain fixed11

fixed10 EVAL (chain fixed9, no reward change, +3M): **mean wave 3.2** (new best),
median 3, max 4, score mean **15002** (new high) / max 23875, dist {2:2, 3:15, 4:8}.
  vs fixed9 {2:7, 3:9, 4:9} score 13641. Chaining MATURED the evasion policy:
    - wave-2 deaths collapsed 7->2/25; bimodality RESOLVED. 23/25 reach w3+, 8/25
      reach w4, only 2 early deaths. A consistent, healthy distribution now.
    - mean + mean score both up. Milestone video: videos/milestone_fixed10.mp4.

Session arc: fixed6 2.7 -> fixed7(aim) 3.0 -> fixed8(chain) 3.0 -> fixed9(evasion)
3.1 -> fixed10(chain) 3.2. Pattern confirmed: LEVERS (aim, evasion) break ceilings,
CHAINING consolidates. Monotone improvement, no regressions since the fixed5 lesson.

New frontier: the WAVE-4 CEILING. 8/25 reach w4 but 0/25 ever clear it (max=4 every
link). Breaking into w5 is the next target.

fixed11 = chain fixed10 (models/ceqasoqt) forward, SAME config, 3M. One more
consolidation pass — does continued maturation push any run into w5, or is w4 a
hard wall needing a new lever? If it plateaus at 3.2 with no w5, NEXT STEP is
wave-4 death forensics (the death-logging harness from task #41) to see what
specifically kills the agent in w4 — then motivate a targeted lever instead of
guessing (candidate: multi-threat "surrounded" danger, but that risks the
crowding-backfire, so forensics first).
OUTPUT_DIR: models/qelkncca.

## 2026-06-14 (overnight) — fixed11 PLATEAU at 3.2; pivot to wave-4 death forensics

fixed11 EVAL (chain fixed10, no change, +3M): **mean wave 3.2** (SAME as fixed10),
median 3, max 4, score mean 14003 / max 24000, dist {2:2, 3:16, 4:7}.
  vs fixed10 3.2 {2:2, 3:15, 4:8} score 15002. PLATEAU CONFIRMED — two consecutive
  chains at 3.2, max=4 every link since fixed7, 0/25 ever reach wave 5. Chaining is
  exhausted; the wave-4 wall is hard. Milestone video: videos/milestone_fixed11.mp4.

NEXT: wave-4 death forensics (don't guess the next lever). The env already logs
every death to MAME_DEATH_LOG_DIR as JSONL (suspects w/ names+dist+vanished flag,
tiered verdict direct/closing/debris/unmapped/unexplained). Running a forensics
eval of fixed11 to bucket deaths by wave and tally WHAT kills the agent at w4 (and
w3) -> motivate a targeted lever (e.g. if enforcer/tank FIRE dominates, add a
projectile-dodge signal; if hulks, a hulk-avoid term; if walls, a wall term).

## 2026-06-14 (overnight) — WAVE-4 FORENSICS: deaths are "surrounded"; fixed12 adds encirclement

Forensics eval (fixed11, 40 runs, 120 deaths logged):
  - 110/120 deaths are explained-DIRECT (contact). Almost NO projectile deaths.
  - Prime killers: Grunt 52/120 (43%), Hulk 26 (22%), Enforcer 11, Electrode 10,
    Spheroid 9, EnfBullet/Spark 7. Contact with grunts+hulks = 65% of deaths.
  - SURROUNDED is the mechanism, escalating with wave — avg lethal threats within
    30u AT DEATH: w2=2.6, w3=4.2, w4=5.7 (only 3/27 w4 deaths had a single threat;
    most 5-10). Hulks within 30u in 30/49 w3 deaths (they herd/box the player).
  => The nearest-only _PROX_PENALTY can't see the box: it dodges the closest
     threat straight into the others. THIS is why chaining plateaued at 3.2.

fixed12 = chain fixed11 (models/qelkncca) + ENCIRCLEMENT penalty (one new variable):
  - _threat_field() now also returns n_encircle_sides = # distinct compass octants
    holding a threat within _ENCIRCLE_RADIUS=35.
  - reward += -0.3 * (n_sides - 2) when n_sides >= 3 (0 at <=2 sides).
  - Direction-aware, NOT density-based: a single escapable cluster on one side is
    free (won't steer away from deep waves); only true encirclement (3+ sides, no
    exit) is penalized. Teaches keeping an escape route / moving to open space.
  - Smoke-test (fixed11 policy): boxed fired 4% at w1-2 (correctly sparse when not
    surrounded), total -7.5 vs +2237 positive — grows in dense waves by design.
  - If it lifts depth past w4 -> chain to mature. If not -> raise _ENCIRCLE_PENALTY
    or address the Hulk-aim misdirection (aim reward draws agent toward unkillable
    hulks) as fixed13.
OUTPUT_DIR: models/1etzyhnw.

## 2026-06-14 (overnight) — fixed12 encirclement REGRESSED (reverted); fixed13 = hulk-aim fix

fixed12 EVAL (chain fixed11 + encirclement penalty): **mean wave 2.9** — REGRESSION
from 3.2. dist {2:8, 3:11, 4:6} (wave-2 deaths 2->8), score 12681.
  => NEGATIVE RESULT. My forensic read was partly wrong: the agent dies surrounded,
     but "avoid being surrounded" is the wrong lesson — in dense waves you're always
     somewhat surrounded, so the penalty taught FREEZE/FLEE instead of moving through
     gaps (freezing = death). No silver lining (fewer deep runs AND more early
     deaths), unlike fixed9's productive bimodality -> NOT chained. REVERTED.
  Video: videos/milestone_fixed12_REGRESSED.mp4. The real surrounded-death fix is
  better MOTION, which a state penalty can't express (future: motion shaping / obs).

*** CURRENT BEST MODEL: fixed11 = models/qelkncca (mean wave 3.2, max score 24000). ***
Reward family: score/10 + survive(0.3*wave) + kill/rescue bonuses + aim(+0.5 on-axis)
+ danger(graduated contact penalty, 20u) + Option-A wave bonuses. ent_coef 0.005.

fixed13 = warm-start fixed11 (models/qelkncca, NOT the regressed fixed12) + HULK-AIM
FIX (one new variable vs the 3.2 baseline):
  - _threat_field() now returns nearest_ANY (for the contact penalty, hulks incl.)
    and nearest_SHOOTABLE (for the aim bonus, hulks EXCLUDED).
  - Hulks are indestructible (forensics: 22% of deaths) — rewarding aim at them is
    wasted and lures the agent toward contact. Now aim only rewards pointing at
    killable threats; hulks stay in the danger penalty (must dodge).
  - Low backfire risk: removes a misdirection, adds NO timidity (unlike encircle).
  - Smoke-test clean (boxed absent, aim/danger fire, reached w3, no crash).
  - If fixed13 <= 3.2, STOP experimenting and leave fixed11 as the night's result;
    the plateau likely needs a motion/obs change, not more reward shaping.
OUTPUT_DIR: models/92002rzy.

## 2026-06-14 (overnight) — fixed13 HULK-AIM FIX broke the wave-4 ceiling (first wave 5!)

fixed13 EVAL (fixed11 baseline + hulk-aim exclusion): **mean wave 3.1**, median 3,
**max wave 5 (FIRST EVER)**, score mean 13570 / **max 32900 (new all-time high)**,
dist {2:5, 3:14, 4:5, **5:1**}.
  vs fixed11 {2:2, 3:16, 4:7} max-wave 4, max-score 24000. The hulk-aim fix
  CRACKED THE CEILING: max wave 4->5 (first w5 in 200+ eval runs across 8 links),
  top score 24k->32.9k. Mean flat (3.1 vs 3.2 = eval noise), but for the MARATHON
  goal depth-ceiling > mean. Same shape as fixed9's productive bimodal intro that
  fixed10 consolidated -> chaining warranted. Video:
  videos/milestone_fixed13_FIRST_WAVE5.mp4.

  Interpretation: removing the aim reward on indestructible hulks stopped the agent
  from committing fire+position to enemies it can't kill, freeing it to clear
  killable threats and slip past hulks -> deeper waves.

fixed14 = chain fixed13 (models/92002rzy) forward, SAME config (hulk-aim kept), 3M.
Pure continuation to consolidate the wave-5 capability (mirrors fixed9->fixed10).
STOP CONDITION (revised): if fixed14 doesn't consolidate (mean <=3.1 AND no clearer
w5 presence), stop reward-experiments; the residual plateau needs a MOTION/OBS
change (the surrounded-death problem fixed12 couldn't solve with a state penalty),
a call to make with the user. Current best-by-mean: fixed11 (3.2). Best-by-depth:
fixed13 (w5, 32.9k).
OUTPUT_DIR: models/5t8lu5hl.

## 2026-06-14 (overnight) — fixed14 CONSOLIDATED wave-5 (new BEST overall)

fixed14 EVAL (chain fixed13, hulk-aim kept, +3M): **mean wave 3.2** (back to best),
median 3, max 5, score mean **15472 (new high)** / max **33850 (new all-time high)**,
dist {2:6, 3:11, 4:6, **5:2**}.
  vs fixed13 3.1 {2:5,3:14,4:5,5:1}. CONSOLIDATION WIN (fixed9->fixed10 pattern):
  wave-5 reaches DOUBLED 1->2/25, mean back to 3.2, score record. The wave-5
  capability is real, not a fluke. *** fixed14 = models/5t8lu5hl is now BEST overall
  (3.2 mean + consistent wave-5 + record score). *** Video: milestone_fixed14_BEST.mp4.

fixed15 = chain fixed14 (models/5t8lu5hl), SAME config, 3M. Still improving (not
plateaued) -> one more consolidation pass to push wave-5 presence / reach wave 6.
STOP: if fixed15 plateaus (no mean or wave-5 gain vs fixed14), that's the natural
end for the hulk-aim config (cf. fixed10->fixed11 plateau) — leave fixed14 as the
result; deeper progress then needs a motion/obs change (a call for the user).
OUTPUT_DIR: models/121g9nl1.

## 2026-06-14 — fixed15 PLATEAU; STOPPING reward-shaping arc. SESSION SUMMARY

fixed15 EVAL (chain fixed14, +3M): **mean wave 3.2** (= fixed14), dist {2:4, 3:13,
4:7, 5:1}, score mean 14331 / max 33950. PLATEAU (fixed14 3.2 -> fixed15 3.2,
wave-5 2->1 = noise). Stop condition met. Video: videos/milestone_fixed15.mp4.

### Overnight session arc (mean wave / notable):
  fixed6  2.7  (proven config, plateau baseline)
  fixed7  3.0  AIM REWARD (+0.5 on-axis at nearest shootable) — tightened dist
  fixed8  3.0  chain — +27% score, depth flat (lever vs chain pattern emerges)
  fixed9  3.1  EVASION PENALTY (graduated 20u contact) — wave-4 reaches 3->9/25
  fixed10 3.2  chain — variance collapsed, wave-2 deaths 7->2
  fixed11 3.2  chain — PLATEAU (depth gated by survival, not aim/score)
  [forensics: 110/120 deaths CONTACT; grunts 43% + hulks 22%; die SURROUNDED,
   avg threats-within-30u at death w2=2.6 w3=4.2 w4=5.7]
  fixed12 2.9  ENCIRCLEMENT PENALTY — REGRESSED (taught freeze/flee), REVERTED
  fixed13 3.1  HULK-AIM FIX (don't reward aim at indestructible hulks) — FIRST
               EVER wave 5, top score 32900 (broke the wave-4 ceiling!)
  fixed14 3.2  chain — CONSOLIDATED: wave-5 1->2/25, score mean 15472 (BEST)
  fixed15 3.2  chain — plateau -> STOP

### DELIVERABLE / BEST MODEL: fixed14 = models/5t8lu5hl
  mean wave 3.2, reaches wave 5 (~2/25), score mean 15472 / max 33850. Reward:
  score/10 + survive(0.3*wave) + spawner/shooter/brain-kill + rescue(150) +
  aim(+0.5, hulks excluded) + danger(graduated 20u contact) + Option-A wave
  bonuses. ent_coef 0.005, lr 3e-4, gamma 0.999. Milestone videos in videos/
  (fixed6..fixed15, incl. the_FIRST_WAVE5 and _BEST).

### Why stopping: reward shaping is mined out — 3 wins (aim, evasion, hulk-aim) +
1 clean negative (encirclement) brought 2.7 -> 3.2 AND broke the wave-4 ceiling
into wave 5. Now plateaued. Per [[feedback_plateau_patience]] the user prefers
patience/architecture over more reward/HP pokes at a healthy plateau. The residual
gate is the "surrounded" death (a state penalty can't fix it — fixed12 proved
that); it needs better MOTION, which points to an OBSERVATION or POLICY change,
not another reward term. That's a design call for the user.

### Recommended next directions (for the user to choose):
  1. OBS: the 945-dim slot obs may not give the policy enough local spatial
     structure to navigate gaps when surrounded. Try the (11,36,24) GRID obs +
     GridCNN (already in train_mame.py via --obs-mode grid) — a CNN sees the
     field layout / escape routes directly.
  2. Longer horizon / recurrence (frame stack or LSTM policy) so the agent can
     plan motion through a swarm rather than react step-by-step.
  3. Curriculum: seed some episodes from wave-4/5 save-states (reset_pool) so the
     policy gets more gradient on the dense-wave survival it rarely reaches from
     a wave-1 boot.

## 2026-06-14 — GRID OBS experiment launched (user-requested next direction)

User chose the grid-obs path (recommendation #1). The (11,36,24) SpatialGridObsBuilder
is an ABSOLUTE full-playfield grid (24col x 36row over the 665x492 field, ~28x14px
cells, counts-accumulate per cell) — shows ALL enemies everywhere, unlike the slot
obs which truncated to nearest-N per category (suspected cause of "dies surrounded").

grid1 = FRESH (grid obs can't warm-start from slot models — different obs space),
5M steps, GridCNN, ent_coef 0.01 (fresh-exploration), same reward shaping (aim/
evasion/hulk-aim all apply, computed from packet). models/12dau2we.

GOTCHA fixed: first launched on --device cpu -> CNN compute oversubscribed the 24
cores (trainer 1054% CPU + 12 MAMEs = 2250%), throughput crawled, first launch also
hit a MAME respawn-storm (27 MAMEs) on a slow boot. FIX: --device cuda (RTX 4080).
Trainer CPU 1054%->76%, total 2250%->1276%, GPU 29% util / 2.2GB. Throughput **744
fps** (faster than slot's ~500 — GPU freed the cores), ETA ~1.9h for 5M. ep_rew_mean
1570 at 98k steps (fresh policy already getting dense-shaping reward).
NOTE: SB3 warns "should be MlpPolicy on GPU" — spurious; the grid path is MlpPolicy
+ custom GridCNN feature extractor, which does use the GPU. Eval with obs_mode grid.

## 2026-06-14 — grid1 EVAL: 2.6 fresh (under-trained); chaining grid2

grid1 EVAL (FRESH 5M grid obs, obs_mode grid): **mean wave 2.6**, max 4, score mean
6533 / max 18850, dist {2:12, 3:11, 4:2}. Final ep_rew_mean 2012 (climbed 1570->2012
ALL RUN, NO plateau = under-trained).
  vs slot fixed14 3.2/w5 — but UNFAIR: fixed14 = ~10 chained runs (30M+ steps); grid1
  = ONE fresh 5M from scratch. Reaching w4 fresh in 5M is a fair start; the reward
  trajectory says it's nowhere near converged. Video: videos/grid1_FRESH.mp4.

grid2 = chain grid1 (models/12dau2we) +5M, GPU, obs grid, ent 0.01 (same — isolate
"more training"). Tests whether the grid policy's trajectory catches up toward the
slot baseline. If grid2 climbs well (toward 3.0+), grid is promising -> keep chaining
(and consider dropping ent to 0.005 for committed play, like the slot lineage). If
grid2 stalls well below slot, the grid/CNN may need more steps than it's worth vs the
proven slot model. Total grid budget after grid2 = 10M (still < slot's 30M+).

## 2026-06-14 — grid2 PLATEAU at 2.5; grid3 tests the ent_coef confound

grid2 EVAL (chain grid1, +5M = 10M total grid, ent 0.01): **mean wave 2.5** (median 2),
max 4, score mean 6043, dist {2:13, 3:11, 4:1}. FLAT vs grid1's 2.6 — the grid policy
PLATEAUED at ~2.6 (ep_rew also hovered ~1900-2000, never passed grid1's 2012).
Skipped grid2 video (behaviorally ~= grid1).

vs slot fixed14 3.2/w5 — grid is below. BUT one strong confound remains: ent_coef.
Grid ran at 0.01; the slot lineage PROVED 0.01 is too random (fixed1 92% random) and
0.005 gave committed play (a top-3 jump of the session). So grid's 2.6 plateau may be
under-committed play, not an obs limit.

grid3 = chain grid2 (models/cpzcsnz2) + **ent_coef 0.005** (proven committed-play
setting), 5M, GPU, obs grid. DECISIVE TEST:
  - jumps toward 3.0+  => grid obs IS competitive; plateau was just entropy. Pursue.
  - stays ~2.6         => grid/CNN genuinely underperforms slot here; ship slot
                          fixed14 (3.2, wave 5) as the deliverable, grid not worth it.

## 2026-06-15 — GRID VERDICT: underperforms slot even with ent fix. STOP grid; ship slot.

grid3 EVAL (chain grid2 + ent_coef 0.005): **mean wave 2.7**, median 3, max 4, score
mean 7420 / max 15575, dist {2:11, 3:10, 4:4}. Video: videos/grid3_BEST_GRID.mp4.

### Grid arc (all 12 envs, GridCNN on RTX 4080, ~650 fps):
  grid1  2.6  fresh 5M,  ent 0.01  {2:12,3:11,4:2}
  grid2  2.5  chain 10M, ent 0.01  {2:13,3:11,4:1}  (flat — plateau)
  grid3  2.7  chain 15M, ent 0.005 {2:11,3:10,4:4}  (ent fix: small lift, +score, +w4)

### VERDICT: the 24x36 grid obs UNDERPERFORMS the slot obs for this task.
  - Plateaus at ~2.6-2.7 across 15M steps vs slot fixed14's 3.2 (and wave 5).
  - The ent_coef 0.005 fix CONFIRMED entropy was part of the limit (2.5->2.7, w4 1->4,
    score +23%) — but did NOT close the gap to slot. So it's not just entropy.
  - Most likely cause: coarse cell resolution (~28x14px, ~1 sprite/cell) can't give
    the policy precise enemy positions for dodging, whereas the 945-dim slot obs
    feeds EXACT nearest-N distances. Full-field coverage didn't beat exact-local.

### DECISION: STOP the grid exploration. *** SHIP slot fixed14 = models/5t8lu5hl ***
  (mean wave 3.2, reaches wave 5, score max 33850) as the best deliverable.

### Open option for the user (NOT launched autonomously — multi-hour speculative):
  Finer grid 48x72 (one-line GRID_W/GRID_H change in spatial_obs.py) — tests the
  resolution hypothesis directly. Bigger CNN, slower, fresh from scratch (~4-6h of
  runs for a comparable read). Worth it ONLY if the user wants to chase the grid
  approach; otherwise the slot model + the other recommendations (recurrence/LSTM,
  wave-4/5 curriculum seeding) are better bets for breaking past wave 3.2 mean.

## 2026-06-15 — FINER GRID 48x72 (user-requested resolution test)

User chose to test the resolution hypothesis. spatial_obs.py GRID_W,GRID_H 24,36 ->
48,72: cells ~14x7px (~1 sprite each, was ~28x14). GridCNN auto-adapts (dummy-forward
n_flat); smoke-tested obs (11,72,48) + forward OK.

grid4 = FRESH 48x72, 5M, ent 0.01, GPU — matches grid1's config EXACTLY except grid
size, so grid4 vs grid1 (2.6) is a clean RESOLUTION-only comparison. models/5ivs3hae.
Healthy: 12 MAMEs, trainer CPU 86% (GPU offload OK), GPU 15%/2.7GB.
  - If grid4 > grid1 (toward 3.0+): resolution WAS the grid limit -> chain + ent 0.005.
  - If grid4 ~= grid1 (~2.6): finer res doesn't help; grid obs is genuinely worse than
    slot for this task -> ship slot fixed14, done with grid.

## 2026-06-15 — CURRICULUM SEEDING prepped (queued after grid4) — race-to-100 lever

User granted broad overnight autonomy for the race to 100. Highest-value lever found:
mame_states/robotron has **322 deep-wave save-states** (cataloged by wave -> /tmp/
state_waves.json): wave5=77, w6=74, w7=36, w8-14~18, w15=18, w16-23~39, w24=22, w28=12,
up to w30. The slot policy plateaus at 3.2 because it NEVER trains past wave 3-4 from a
wave-1 boot; seeding episodes from these states gives direct gradient on dense-wave
survival. Eval stays continuous-from-wave-1 (the real metric; Xbox has no save states).

Curriculum reset_pool v1 (mame_gym/curriculum_pool_v1.txt, 100 entries): 35% wave-1
boot (keep continuous skill) + 50% frontier waves 5-7 (bridge past the plateau) + 15%
deep (w15/24/28). Sampled uniformly per episode.

READY-TO-LAUNCH (after grid4 frees the 12 MAMEs) — warm-start slot fixed14, ONLY new
variable = reset_pool (curriculum):
  .venv/bin/python3 train_mame.py \
    --bc-checkpoint models/5t8lu5hl/final_model.zip \
    --vec-normalize models/5t8lu5hl/vec_normalize.pkl \
    --num-envs 12 --timesteps 5000000 --obs-mode slot \
    --ent-coef 0.005 --lr 3e-4 --gamma 0.999 --clip-range 0.2 \
    --base-port 9920 --frameskip 4 --device cpu \
    --reset-pool "$(cat mame_gym/curriculum_pool_v1.txt)"
  Hypothesis: training on waves 5-7+ lets the policy clear them in continuous play ->
  breaks the wave-4 ceiling / lifts mean wave. If it works, progress the curriculum
  deeper (shift the pool toward w8-15) in subsequent runs = the path toward 100.

## 2026-06-15 — CURRICULUM HARDENED against wave-specialism (user feedback)

User: "don't fall back into the old state — as soon as we pass once we marked it done.
It didn't fully train on a level, just got a lucky level once. The model needs to go
1-100." => guard against wave-specialism. Revised design:

RULES (binding):
1. MASTERY-GATED progression, never first-touch. Advance the curriculum deeper ONLY
   when the CONTINUOUS-from-wave-1 eval RELIABLY reaches+clears the current frontier
   — threshold: >=50% of 25 continuous runs reach the frontier wave AND median climbs.
   Reaching a wave once = luck, does NOT count and does NOT advance anything.
2. SUCCESS METRIC = continuous 1->N distribution (MEDIAN / %-reaching-wave-N climbing),
   NOT max wave and NOT seeded survival. (Note: slot fixed14's "wave 5" was 1/25 runs
   = exactly the lucky-once signature; median was 3. Real progress = median rising.)
3. Seeds are a TRAINING AID ONLY (expose deep-wave combat the policy can't reach yet);
   eval is always a true wave-1 boot, no save-states (matches Xbox hardware).
4. PRIMARY diet stays continuous traversal: pool v2 = 55% wave-1 boot + 45% frontier
   w5-7 (varied across 187 states so it generalizes, not memorizes one board). Dropped
   the far-deep w15/24/28 islands — they'd teach disconnected survival the model can't
   bridge to (the specialism trap).
5. Curriculum progression = move the frontier band up only after mastery (e.g. add w8-10
   seeds once continuous median hits ~6-7), building a real 1->100 staircase.

LAUNCH (after grid4) uses mame_gym/curriculum_pool_v2.txt (NOT v1):
  --reset-pool "$(cat mame_gym/curriculum_pool_v2.txt)"  (else identical to the v1 cmd)

## 2026-06-15 — grid4 (48x72) = 2.7: resolution rejected. GRID CONCLUSIVELY DONE.

grid4 EVAL (fresh 48x72, ent 0.01): mean wave 2.7, median 3, max 4, score 6769,
dist {2:9, 3:14, 4:2}. ~= grid1's 2.6 (24x36). Finer resolution did NOT help.
FINAL GRID VERDICT: 24x36=2.6, 24x36+ent0.005=2.7, 48x72=2.7 — all plateau ~2.6-2.7
regardless of resolution OR entropy, vs slot 3.2. The grid/CNN obs genuinely
underperforms the slot obs for this task. Closing the grid direction. (Skipped grid4
video — conclusive negative, ~= grid1 behaviorally.) Slot fixed14 stays the model.

=> Proceeding to CURRICULUM SEEDING (the race-to-100 lever) on slot fixed14.

## 2026-06-15 — curric1 did NOT break the plateau (honest read); 3 approaches converge

curric1 EVAL (slot fixed14 + curriculum v2, continuous from wave 1): mean 3.2,
**median 3**, max 6, score mean 14906 / max 48125, dist {2:4, 3:15, 4:5, 6:1}.
  HONEST VERDICT (per anti-specialism rules): NO plateau break. Median flat at 3;
  reaches-wave-4+ = 6/25 vs fixed14's 8/25 (slightly LOWER, within noise). The max-6
  and record score are LUCKY-ONCE outliers (1/25) — exactly what the user warned not
  to chase — NOT progress. Deep-wave training gave a marginally better floor (wave-2
  deaths 6->4) but did NOT transfer to reliably going deeper. Video: videos/curric1.mp4.

*** CONVERGENT EVIDENCE — a real capability ceiling, not a quick-fix issue: ***
  Reward shaping (fixed14):  median 3, mean 3.2
  Grid obs (grid1-4, 2 res): median 2-3, mean 2.6-2.7
  Curriculum (curric1):      median 3, mean 3.2
  THREE distinct levers (reward / observation / data-distribution) all top out at
  median-3 / mean-3.2. Strong signal the PPO+MLP+slot-obs setup has a fundamental
  ceiling. The remaining untried lever is ALGORITHMIC: recurrence.

NEXT BET: RecurrentPPO (sb3_contrib 2.7.0 confirmed available) + MlpLstmPolicy. A
single-frame MLP can't integrate enemy TRAJECTORIES over time; an LSTM can track
converging threats across steps — directly targets the "dies surrounded" failure
(63% of deaths in the open field to converging grunts). Requires code: train_mame.py
--recurrent path + eval_continuous.py LSTM-state handling. Implement + smoke-test
carefully (bug risk), then launch fresh. fixed14 remains the deliverable.

## 2026-06-15 — RECURRENT (LSTM) experiment launched — architecture pivot

Implemented RecurrentPPO support (train_mame.py --recurrent: MlpLstmPolicy, 256 hidden,
n_steps=128; eval_continuous.py "rnn" mode threads+resets LSTM state per episode,
non-recurrent paths untouched). Smoke-tested end-to-end: 4k-step train + 3-ep recurrent
eval both clean (plumbing verified; state resets per episode).

rnn1 = FRESH RecurrentPPO/LSTM, slot obs, 5M, ent 0.005, GPU. models/pw3xm9by.
Rationale: 3 non-recurrent levers all plateaued at median-3; an LSTM integrates enemy
trajectories over time (targets "dies surrounded by converging grunts"). 
CAVEATS (flagged for the user): fps ~223 (LSTM BPTT is heavy) => ~6h for 5M; fresh LSTM
trains slower than MLP, so a single 5M run may be INCONCLUSIVE (may not yet match
fixed14's 3.2) — it's the first data point on whether recurrence has legs, likely needs
multiple runs / more steps to mature. Will eval intermediate checkpoints for an early
read. fixed14 stays the deliverable; run is reversible if the user prefers another bet
(bigger net / frame-stack / death-rung curriculum).

## 2026-06-15 — rnn1 intermediate read (800k/5M): on-track, unremarkable so far

rnn1 @ 800k checkpoint (recurrent eval, 15 runs): mean 2.4, median 2, max 3,
dist {2:9, 3:6}. Fresh LSTM at 16% trained — learning competitively (reliably reaches
w2-3) but NO early breakout; trajectory ~ the fresh grid runs (which hit 2.6 by 5M).
ep_rew climbing 1160->1830. LSTM's temporal edge (if real) would emerge as the
recurrent state matures with more steps. Full 5M run (~5h left at 229fps) is the real
read. Will re-check at ~2.5M. fixed14 (3.2) still the deliverable.

## 2026-06-15 — rnn1 trajectory CLIMBING: 800k m2 -> 1.7M m3

rnn1 @ 1.7M (recurrent eval, 15 runs): mean 2.6, **median 3**, max 3, dist {2:6, 3:9}.
Trajectory: 800k=median2/mean2.4 -> 1.7M=median3/mean2.6. The LSTM is steadily
climbing and NOT yet plateaued at 35% trained — approaching fixed14 (median3/mean3.2).
ep_rew 1160->1980. Key question for the full 5M run: does it push PAST 3.2 (recurrence
validated) or plateau at ~3.2 like the 3 non-recurrent levers? Encouraging trend; ~4h
to full completion. Next intermediate eval ~3M.

## 2026-06-15 — rnn1 trajectory continues climbing: 2.6M mean 2.8 / median 3 / reaching w4

rnn1 @ 2.6M (recurrent eval, 20 runs): mean 2.8, median 3, max 4, dist {2:8, 3:8, 4:4}.
Trajectory (mean): 800k=2.4 -> 1.7M=2.6 -> 2.6M=2.8. Median steady at 3, now reaching
wave 4. NOT plateaued at 54% — a genuine unbroken climb (contrast the flat non-recurrent
plateaus). ep_rew 1160->2260. On track to ~match fixed14 (3.2) by 5M; whether it
EXCEEDS the 3.2 wall is open and a fresh LSTM likely needs >5M to show its ceiling.
~2.7h to full completion. fixed14 stays the deliverable.

## 2026-06-15 — rnn1 3.5M = mean 2.6: recurrence PLATEAUS at median-3 too

rnn1 @ 3.5M (recurrent eval, 20 runs): mean 2.6, median 3, max 4, dist {2:9, 3:9, 4:2}.
DIPPED from 2.6M's 2.8. Full trajectory (mean): 800k 2.4 -> 1.7M 2.6 -> 2.6M 2.8 ->
3.5M 2.6. Oscillating ~2.6-2.8 / median 3 — PLATEAUING, not climbing through (the 2.8
was a favorable 20-run sample within noise). Slightly BELOW fixed14's 3.2.

*** KEY FINDING: FOUR approaches now converge at median-3 / mean ~2.6-3.2: ***
  reward shaping (3.2), grid obs (2.6-2.7), curriculum (3.2), recurrence/LSTM (~2.6-2.8).
The median-3 wall is robust across reward, observation, data-distribution, AND
architecture (MLP vs LSTM). This is NOT a tuning issue — it's a deep ceiling. Likely
points to something more fundamental: task difficulty (wave 3-4 w/ 2 lives is genuinely
hard), control resolution (action space / frameskip 4), or a need for far more training
/ a different method class. The next move is a fundamental rethink — a DIRECTION CALL
for the user, not another incremental lever.

Letting rnn1 finish (72% done) for the definitive 5M number, but the verdict is clear:
recurrence at this scale (256 hidden, 5M) does NOT break the plateau. fixed14 (3.2)
remains the best deliverable.

## 2026-06-15 — rnn1 FINAL (5M) = mean 2.6 / median 3. RECURRENCE VERDICT: no breakthrough.

rnn1 final 5M (recurrent eval, 25 runs): mean 2.6, median 3, max 4, dist {1:2,2:8,3:12,4:3}.
Below fixed14's 3.2, slightly worse floor (2 wave-1 deaths). DEFINITIVE: LSTM at this
scale (256 hidden, 5M fresh) does NOT beat the MLP — lands in the same median-3 band.

#### ====================  OVERNIGHT SYNTHESIS (for the user)  ====================
FOUR fundamentally different approaches all converge at MEDIAN WAVE 3 (mean 2.6-3.2):
  - Reward shaping (slot fixed14):     mean 3.2, median 3, reaches w5  <-- BEST
  - Observation/grid (24x36 & 48x72):  mean 2.6-2.7, median 3
  - Curriculum seeding (deep states):  mean 3.2, median 3 (max-6 was lucky-once)
  - Recurrence (LSTM):                 mean 2.6, median 3
The median-3 wall is ROBUST across reward, representation, data-distribution, AND
architecture. This is not a tuning gap — it is a fundamental ceiling.

BEST DELIVERABLE: slot fixed14 = models/5t8lu5hl (mean 3.2, reaches wave 5).

LIKELY REAL BOTTLENECK (untested, fundamental — a DIRECTION CALL for the user):
  1. CONTROL RESOLUTION / frameskip. Agent acts every 4 frames; grunts close ~12u/step.
     In dense waves it may simply LACK the reaction granularity to dodge (death forensics
     = contact deaths while surrounded). Try frameskip 2 (or 1). TOP recommendation —
     cheap, directly targets the failure. NOTE: frameskip affects the sim->Xbox input-rate
     mapping, so it's a deployment-relevant decision (hence the user's call).
  2. LIVES / task framing. Eval uses 2 lives. Real Xbox life dynamics (extra lives via
     score) change how deep a run goes — worth matching the target's economy.
  3. MUCH longer training (100M+). The 4-approach convergence argues somewhat against
     pure-steps being the fix, but not conclusively.
  4. Accept median-3 as the current ceiling for PPO-class methods here; the leap to 1->100
     may need expert-level play (offline RL from human/FSM demos, MCTS-style search, etc.).
RECOMMENDATION: do NOT throw another incremental lever at the same wall. Discuss the
fundamental direction (lean: frameskip experiment first). Stopping the autonomous loop
here — this is a decision point that needs the user.

## 2026-06-15 — ROOT CAUSE BREAKTHROUGH: the MAME GYM is the bottleneck, not RL

User: simple state machines reach level 8-9 on XENIA (the XBLA game = same ROM as MAME).
Why is RL stuck at median 3? FSM-oracle investigation:

  SAME crude potential-field FSM (flee threats+walls, shoot nearest), two gyms:
    PYTHON gym:  level 25, 22, 8   (median ~22; most runs hit the 40k-step cap ALIVE)
    MAME gym:    wave 4            (median 4; dies ~once/wave)
  => 6x+ gap with IDENTICAL agent code = it is the ENVIRONMENT, not RL, not FSM quality.
  (Sophisticated user-FSM also gets 23 on python; RL historically reached L12 on python.)

RULED OUT as the MAME cause: false-death (deaths are real, gs=0x1b); player speed
(~9.5px/step, ~6x a grunt); lives (2 confirmed correct, extras earned); frameskip
(fs=1 still caps at 3); FSM quality (same crude FSM gets 25 on python).

CONCLUSION: every agent caps at wave 3-4 on the MAME gym specifically; the same agents
go far deeper on python and the same ROM gives 8-9 on Xenia. The MAME gym makes the game
far harder than the real XBLA ROM. ALL prior median-3 results (reward/grid/curriculum/
LSTM) were measuring this ENV ceiling, not agent skill.

NEXT: (1) run a known-good FSM (chooseOutputs) ON MAME — 8-9 => MAME faithful/agents weak;
~4 => MAME defect. (2) compare wave-1 enemy count/type/speed MAME vs python vs Xenia.
Prime suspects: DIP/difficulty config or rl_reset.sta encoding a hard difficulty; action
injection. MAME launch uses cfg/nvram DIPs (cwd=rompath parent).

## 2026-06-15 — CORRECTED ROOT CAUSE: MAME is FINE; the good FSM hits wave 16; RL just fails to learn

Ported the user's ACTUAL FSM (robotron_fsm.chooseOutputs) onto the MAME env (coord map
game-units->665x492 px; classify_sw name map; always-move/fire handling). Result:
  chooseOutputs on MAME: wave 9,16,16,16,16,16  => median 16, max 16 (reproducible).

=> BOTH earlier conclusions were WRONG:
   - NOT a fundamental ceiling (a rule agent reaches wave 16).
   - NOT a broken MAME env (it allows wave 16; my crude potential-field FSM at wave 4
     was simply a weak agent — flee+shoot w/o threat prioritization / family handling).
   The python-gym comparison was a red herring (user's reimpl may be too easy).

REAL FINDING: MAME is faithful and allows deep play. RL at median 3 is genuinely
UNDERPERFORMING vs an achievable wave-16 bar on the SAME env. All the median-3 results
(reward/grid/curriculum/LSTM) were RL failing to learn, not an env limit.

WHY RL fails (to investigate, in priority):
  1. OBS GAP — the FSM reads EXACT entity positions+types; RL gets the 945-dim
     category-ranked slot obs. Test: feed chooseOutputs ONLY the decoded 945-dim obs;
     if it still hits ~16, obs is sufficient and training is the problem; if it
     degrades, the obs is the bottleneck. (Strong suspect.)
  2. IMITATION LEARNING is the unlock — we now have a wave-16 EXPERT on the exact
     training env. BC/DAgger from chooseOutputs (project already has a BC pipeline:
     collect_brain3_demos.py / train_multihead.py) should jump the policy toward 16,
     vs fighting PPO-from-scratch to median 3.
  3. Reward local optimum / exploration — less likely the main cause given (1)/(2).

Tools added: mame_gym/fsm_choose_on_mame.py (FSM expert on MAME — a ready teacher /
eval oracle), mame_gym/fsm_oracle.py, fsm_python_crude.py.

## 2026-06-15 — OBS-GAP TEST: obs is SUFFICIENT. RL's failure is TRAINING, not obs/env.

Ran chooseOutputs on MAME fed ONLY the entities the 945-dim obs conveys (per-category
nearest-N + 12 catch-all, max 41 slots — same selection as MameObsBuilder):
  OBS-LIMITED:  wave median 12 (9,12,16,9,12,12)
  FULL sprites: wave median 7.5-16 (harness-dependent; high variance on 6 runs)
  RL policy:    median 3
Obs-limited >= full and BOTH >> RL's 3 => the 945-dim obs CONVEYS ENOUGH for wave-12
play. The 41-slot truncation is NOT the bottleneck.

### DEFINITIVE DIAGNOSIS (after correcting two wrong calls):
  - ENV is fine: good FSM reaches wave 9-16 on MAME.
  - OBS is sufficient: same FSM reaches median 12 using ONLY the 945-dim obs view.
  - => RL median-3 is a TRAINING / OPTIMIZATION failure. PPO-from-scratch never learned
       the mapping (obs -> good action) that demonstrably EXISTS and reaches wave 12+.

### THE UNLOCK — IMITATION LEARNING from the FSM:
  A good (obs->action) mapping exists and is computable (the FSM). So:
  1. Generate BC demos: run the OBS-LIMITED FSM, record (945-dim obs, FSM action) pairs.
     (Obs-limited teacher = decisions from the SAME info the policy receives, so the
      policy can actually reproduce them.) Project has a BC pipeline already
      (collect_brain3_demos.py / train_multihead.py) — point it at chooseOutputs.
  2. BC-train the MLP policy to clone the FSM -> should jump toward wave 12+.
  3. RL fine-tune (PPO) from the BC policy to exceed the FSM.
  This replaces "fight PPO-from-scratch to median 3" with "clone a wave-12 expert,
  then improve it." The FSM (fsm_choose_on_mame.py / fsm_obsgap_on_mame.py) is also a
  permanent eval ORACLE and the marathon-to-100 teacher.

## 2026-06-15 — FSM wired as BC TEACHER (imitation-learning pipeline)

Built the imitation pipeline (the unlock from the obs-gap finding):
  - mame_gym/collect_fsm_demos.py — runs the obs-limited chooseOutputs FSM on MAME,
    records (945-obs, FSM-action[move,fire]) to demos/fsm_mame_demos.npz. Diversity:
    broad reset pool (mame_gym/bc_collect_pool.txt = 40% wave-1 + deep states w5-26)
    + epsilon=0.1 DAgger exploration. Validated: balanced action histograms, no collapse.
  - train_bc_mame.py — BC-trains MlpPolicy([512,512], MultiDiscrete[8,8], IDENTICAL to
    train_mame.py) via SB3 evaluate_actions NLL; VecNorm stats fitted on demos. Outputs
    models/bc_fsm/{final_model.zip, vec_normalize.pkl} = drop-in for train_mame.py
    --bc-checkpoint / --vec-normalize.
  Existing train_bc.py is for the PYTHON gym + Discrete-64 (incompatible) — left intact.
Plan: collect 300k pairs -> BC train -> eval continuous (expect a big jump from median 3
toward FSM's wave 12) -> PPO fine-tune from the BC policy to exceed the FSM toward 1->100.

## 2026-06-15 — BC clone underperforms (median 2): covariate shift. Need DAgger.

BC training: 300k demos, 25 epochs. val move_acc plateaus ~60%, fire_acc ~57%.
BC policy eval (models/bc_fsm): stoch median 2 {1:4,2:16,3:5}; DET all-25 = wave 2.
Below from-scratch PPO (3), far below the FSM teacher (12). It SCORES high (28k max) =>
learned FIRING but not survival/dodging (move_acc too low to evade).

DIAGNOSIS: classic BC covariate shift. 60% per-action accuracy -> ~40% wrong moves ->
policy drifts to states absent from FSM-demos -> compounding errors -> dies at wave 2.
Naive BC (FSM trajectories + epsilon) didn't cover the policy's own error states.
Likely accuracy ceiling causes: (a) covariate shift; (b) the 945-obs encodes
relative-pos/dist/angle while the FSM decides from ABSOLUTE pixel positions -> harder
mapping; (c) my obs_limited slot replication may not be byte-identical to
MameObsBuilder's selection (train/label mismatch).

FIX (priority): DAGGER — run the CURRENT policy, label its OWN visited states with the
FSM action, aggregate, retrain; iterate 3-5x. Directly fixes covariate shift (teaches
recovery from the policy's mistakes). Alternatives: feed the policy absolute positions
(match the FSM's representation); RL-fine-tune-from-BC (risky: random value head can
wipe the BC policy). Pipeline tools (collect_fsm_demos/train_bc_mame) ready to extend.

## 2026-06-15 — Slot-match 100%; BC=generalization gap (not capacity); DAgger running

- SLOT-MATCH check (mame_gym/obs_decode.py decodes 945-obs -> FSM entities): FSM-on-decoded-obs
  vs stored demo labels = 100% move & fire agreement. Labels were correct; FSM IS a
  deterministic function of the obs. So the 60% BC accuracy is NOT a label bug.
- Bigger net [1024,1024,512]: train nll 1.18->0.48 but val acc only 60->64% AND eval got
  WORSE (median 1, overfit). => not capacity/underfitting; it's GENERALIZATION / covariate
  shift (MLP fits FSM trajectories, mispredicts on the policy's own visited states).
- => DAGGER (dagger.py): train BC -> run policy (visits its own states) -> label every
  visited obs with decode-FSM -> aggregate -> retrain; iterate. Collection uses the diverse
  pool (covers deep states w/ expert labels); EVAL forces wave-1 boots (true continuous
  metric — fixed a bug where the diverse pool inflated the eval to median 15).
  iter 0 (BC baseline, fixed eval) = median 2. Running 6 iters; best -> models/dagger_best/.

## 2026-06-16 — DAgger lifts BC to median 3 (plateau); RL fine-tune from DAgger launched

DAgger (6 iters, dagger.py, [512,512], 60k/iter, eval=wave-1 boots):
  iter0(BC) m2/1.9  -> iter1 m2/2.7{4:5} -> iter2 m2/2.0 -> iter3 m3/3.0{2:7,4:7} BEST
  -> iter4 m3/2.9{3:13} -> iter5 m3/2.5 -> iter6 m2/2.3.  best=median 3 @ iter3.
  => DAgger fixed covariate shift (BC m2 -> m3 with 7/15 reaching w4) but PLATEAUED at
     median 3 — the SAME wall as from-scratch PPO. KEY: even imitation from a wave-12
     expert caps at median 3. The MLP clones the FSM to only ~64%/action; Robotron is
     unforgiving so the 36% errors kill it ~w3-4. Representation gap, not covariate shift.

RL FINE-TUNE from DAgger-best (models/dagger_best -> models/nn17yanj): PPO doesn't need to
MATCH the FSM, only SURVIVE. DAgger policy = far better init than random (good firing,
reaches w4 ~half the time) -> PPO may optimize survival from that basin and escape the
from-scratch local optimum. 12 envs, 3M, ent 0.005. (Risk: random value head may dent the
policy early; watch for recovery.) Will eval intermediate checkpoints.
If this also plateaus at 3 -> the MLP-on-slot-obs architecture is the limit; next lever is
an ENTITY-RELATIONAL policy (attention over the entity slots) to match the FSM's per-threat
reasoning, which a flattened-MLP can't represent.

## 2026-06-16 — ARCHITECTURE BREAKTHROUGH: entity-attention smashes the BC ceiling

RL fine-tune from DAgger @1.1M = median 2.5 (value-head dent; below the median-3 init) —
flat-MLP not breaking through. So built the entity-relational lever:
  mame_gym/slot_policy.py SlotAttnExtractor — encodes the 945-obs as 41 entity slots +
  player token, self-attention (TransformerEncoder, masked by valid flag), masked mean/max
  pool. Permutation-aware, relational — matches the FSM's per-threat reasoning.
BC with attention (train_bc_mame.py ... attn): val move_acc 64% (MLP ceiling) -> **78%+
by epoch 9 and still climbing**. The flat MLP's 64% wall was the architecture, not the
data. Higher clone fidelity should yield much deeper play.
NEXT (loop): eval attention-BC; if higher -> DAgger + RL fine-tune all with the attention
extractor. The MLP fine-tune (models/nn17yanj) finishes for the record but is the old arch.

## 2026-06-16 — attention-BC: 80% clone fidelity but wave 1 in play (covariate shift); attn-DAgger running

attention-BC final: val move 80.3% / fire 77.4% (vs MLP 64%). Loaded-policy check confirms
79%/76% on demos (no loading bug). BUT continuous eval: DET all wave 1 (argmax death-loop),
STOCH eps all wave 1. So higher BC fidelity did NOT help play — the more expressive attn
policy is MORE brittle off-distribution. => covariate shift dominates; BC-accuracy doesn't
predict play. DAgger (train on the policy's OWN states) is required for ANY architecture.

LAUNCHED: attention-DAgger (dagger.py ... attn) — DAgger loop with the SlotAttnExtractor +
stochastic eval. Hypothesis: attn's higher FSM-fit capacity (79%) + DAgger's covariate-shift
fix should beat MLP-DAgger's median-3 plateau. Runs concurrent with the MLP RL fine-tune
(nn17yanj, different port). Loop armed; watchers on both. Tools: mame_gym/slot_policy.py.

## 2026-06-16 — killed MLP RL-fine-tune (low value) to accelerate attn-DAgger
MLP fine-tune from DAgger (nn17yanj): reached only 1.2M/3M (throttled by contention), 1.1M
eval was median 2.5 — not breaking the median-3 wall (old MLP arch, as expected). Killed it
to free the full machine for attn-DAgger (the breakthrough path). attn-DAgger so far:
iter0 mean 1.0 -> iter1 mean 1.4 (climbing). Now runs unthrottled.

## 2026-06-16 — attn-DAgger plateau (median 2). COMPLETE SCORECARD + strategic reframe.

attn-DAgger final: iter0 m1 -> peaked median 2 (mean 2.1) -> plateaued. WORSE than
MLP-DAgger (m3). Attention's higher BC fidelity (80%) did NOT help play (more brittle,
argmax-locks at wave 1). Dead end.

### FULL SCORECARD — continuous-from-wave-1 median wave (FSM teacher = wave 12-16):
  RL from scratch ........... median 3 (mean 3.2)   <- best, tied
  reward shaping ............ median 3
  grid obs (24x36, 48x72) ... median 3 (mean 2.6-2.7)
  curriculum seeding ........ median 3
  recurrence / LSTM ......... median 3 (mean 2.6)
  MLP behavior cloning ...... median 2
  MLP DAgger ................ median 3 (best imitation)   <- best learned, tied
  attention BC .............. median 1
  attention DAgger .......... median 2
  RL fine-tune from DAgger .. median ~2.5 (killed early; RERUN now = models/aquui4cd)
  *** FSM (rule-based) ...... wave 12-16 ***

### ROBUST CONCLUSION: ~10 distinct learned-policy approaches ALL cap at median 2-3,
while the rule-based FSM reaches 12-16 on the SAME env+obs. This is not a tuning or
architecture gap — it's fundamental: Robotron rewards PRECISE, deterministic play; an
NN approximates the FSM at ~80%/action and the errors are fatal (and argmax can lock).
Neither RL (discover) nor imitation (copy) closes it.

### STRATEGIC OPTIONS (for the user — genuine fork after exhausting standard levers):
  A. The FSM IS the working player (wave 12-16) and runs on the obs available on Xbox
     (memory reading, per ~/win/code). Path to 1->100 = IMPROVE THE FSM's rules, not RL.
     Likely fastest. (Note: FSM caps ~8-9 on Xenia per user; improving it is the lever.)
  B. A higher-fidelity imitation method to actually MATCH the FSM (>95%/action): e.g.
     much larger demo sets + longer DAgger, or a policy that outputs the FSM's exact
     decision structure. Uncertain.
  C. Accept median-3 as the PPO-class ceiling here; 1->100 may need search/planning
     (MCTS-style) or a hybrid FSM+NN.
  LAST learned-policy lever in flight: RL fine-tune from MLP-DAgger (median 3), full 3M,
  clean machine (models/aquui4cd) — if it also caps at 3, the case for a pivot is complete.

## 2026-06-16 — FSM forensics: its ceiling (wave 16) is genuine deep-wave density

FSM (chooseOutputs) on MAME: median 16 (5/6 runs die identically at w16 — deterministic).
77 deaths logged. Deaths-by-wave spread 3-16 (thins out deep). Deep deaths (w>=14, n=20):
ALL contact (explained-direct); prime killers Brain(alt) 10 + Grunt 10; ~half had NO
lethal enemy within 30u at the death frame (fast projectile / cross-frame spark) and ~half
were surrounded (9-12 threats within 30u). => the FSM's wave-16 wall is real: deep-wave
Brain/Grunt density + fast projectiles overwhelm even precise rule-based play. Not a quick
rule fix.

STATE OF THE PROJECT (comprehensive):
  - 11 learned-policy approaches: all median 2-3.
  - FSM (best player): wave 16 (own ceiling = deep-wave density).
  - 1->100 is far beyond BOTH currently.
Holding autonomous experimentation here — the next move is a genuine strategic/direction
decision for the user (improve FSM's deep-wave evasion / new learning paradigm / hybrid /
accept ceiling). Last learned run (aquui4cd) finishing for the record. Loop alive.

## 2026-06-16 — FINAL learned-policy data point: RL-from-DAgger (3M) = median 2. SPACE EXHAUSTED.

RL fine-tune from MLP-DAgger, full 3M (aquui4cd): median 2, mean 2.0 {1:5,2:16,3:4}. The
value-function reset degraded the median-3 DAgger init and it never recovered. Confirms
BC->RL doesn't help here. 11/11 learned approaches capped at median 2-3; FSM = wave 16.

HOLDING: learned-policy space is exhausted and the FSM's ceiling is characterized
(deep-wave density). All remaining moves are STRATEGIC and need the user:
  (A) improve FSM deep-wave survival (positioning + projectile dodge) — best for the goal;
  (B) search/planning (MCTS) or model-based — bigger bet;
  (C) hybrid FSM+NN; (D) accept current ceiling.
No more NN variants will be launched (they all hit the wall). Loop alive at a holding
cadence awaiting direction. Best models: FSM oracle (mame_gym/fsm_choose_on_mame.py, w16);
best learned = from-scratch PPO / MLP-DAgger (median 3, models/dagger_best).

## 2026-06-16 — FSM frameskip-2 = max 15 (no better than fs4's 16). Both ceilings robust. STOP.

Tested the forensic hypothesis (deep deaths = fast projectiles crossing the fs4 gap): ran
chooseOutputs at frameskip 2. Result max 15 / median 9 — NOT deeper, slightly worse/less
consistent. => FSM's ~wave-16 ceiling is robust to timing; deep deaths are genuine
difficulty (density + projectiles), not a sampling gap.

AUTONOMOUS SPACE FULLY EXHAUSTED:
  - 11 learned-policy approaches -> all median 2-3.
  - FSM (best player) -> wave 16, robust ceiling (deep-wave density; frameskip doesn't help).
  - 1->100 needs a fundamentally different approach (search/planning, or a substantially
    better hand-designed player) = a real research effort + strategic decision.
PAUSING the autonomous loop: no remaining lever makes progress without the user's strategic
direction. Everything documented here. Best deliverables: FSM oracle (fsm_choose_on_mame.py,
w16) and from-scratch PPO / MLP-DAgger (median 3, models/dagger_best). Resume with a chosen
direction (improve FSM / search-planning / hybrid / accept ceiling).

## 2026-06-16 — CRITICAL DIAGNOSTIC: eval pipeline is CORRECT; wall = catastrophic covariate shift + under-done DAgger

User pushed back (rightly): imitation from a wave-16 expert should trivially beat wave 3.
Ran the test I'd missed — LIVE policy-vs-FSM agreement:
  - POLICY drives (off-distribution): move 0.44 / fire 0.37, dies wave 1.
  - FSM drives (on-distribution, same eval code): move 0.87 / fire 0.78, env reaches w7+.
=> The eval pipeline is NOT bugged (policy reproduces its ~80% training accuracy on the
FSM's trajectory). The wall is CATASTROPHIC covariate shift: one off action -> unseen state
-> spiral to 44% -> death even on trivial wave 1.

REVISED CONCLUSION (corrects the "fundamental wall" claim): NOT a fundamental limit.
  - DAgger (the fix) DID reach median 3 but PLATEAUED after only 6 iters / 360k states.
    Literature uses 10-50 iters. We UNDER-DID it.
  - Per-action fidelity ~80-87%; Robotron needs ~95%+ at rare critical (dodge) states.
ACTION: aggressive MLP-DAgger (dagger.py 15 100000 = 15 iters x 100k, ~1.5M DAgger states,
deterministic eval) to test the under-trained dimension. Also on the table: long RL run
(30M+ steps; prior runs were only 3M vs Atari's 50M+), and oversampling critical/near-death
states in the imitation data.

## 2026-06-16 — Research synthesis + exploring rewarding paths (user: "explore all paths")

RESEARCH (full synthesis in chat): two ceilings — imitation fidelity (~80-87%, < the ~95%
unforgiving Robotron needs) AND imitating a ceilinged FSM (DAgger can't exceed wave-16). Our
3M RL budget is 15-60x below the Atari floor (under-trained — matches the climbing aggressive
DAgger). Decisive unused asset = the perfect fast simulator.
RANKED LEVERS: (1) Expert Iteration / search-as-teacher over the ROM (manufactures a teacher
STRONGER than the FSM, distill) — the ceiling-breaker; de-risk with a planner-vs-FSM test.
(2) JSRL + Kickstarting on PPO (cheap, lets RL exceed FSM, cures covariate shift on-policy).
(3) Free: frameskip 1-2, DART, action-masking. (4) Implicit-BC / ACT to raise fidelity.
AVOID: GAIL/AIRL (mode collapse), MuZero (we have the sim), diffusion.

NOW RUNNING IN PARALLEL:
  - aggressive DAgger (dagger_long, iter 7+): climbing mean 3.7, 5/15 reach w5 — confirms
    under-trained imitation; bounded by FSM. (best -> models/dagger_mlp_long_best)
  - JSRL (models/5evyvkjc): built mame_gym/jumpstart_wrapper.py + train_mame --jsrl-h0;
    FSM guides first 500 steps (annealing), PPO continues -> can EXCEED the FSM.
NEXT BUILDS: kickstarting (aux CE to live FSM); Expert-Iteration de-risk (CEM/MPPI or
short-horizon search over the ROM via bridge save/restore — does it beat FSM wave 16?);
frameskip-2 variant. Loop restarted per user; exploring the high-reward paths.

## 2026-06-16 — DAgger broke the wall (median 4, oscillating 3-4); JSRL wrapper hung (fragile)

aggressive DAgger (1.3M states): iters 8-9 median 4 (15/15!), iter 10 median 3 — settles
oscillating 3-4. PAST the original median-3 wall; best -> models/dagger_mlp_long_best.
Confirms user's under-trained diagnosis: more DAgger data DID break through (vs 6-iter run
that plateaued at 3). Bounded ~3-4 by FSM fidelity, as expected.

JSRL (jumpstart_wrapper) HUNG: the in-wrapper 500-step FSM-prefix on 12 SubprocVecEnv envs
simultaneously wedged a MAME (trainer 0% CPU, frozen). The wrapper-prefix approach is too
fragile. Killed. (Fix options if revisited: much smaller h0, fewer envs, or annealed
save-state resets instead of in-wrapper FSM fast-forward.)

NEXT (robust): long RL fine-tune from dagger_mlp_long_best (median 4) — 10M steps (vs prior
3M; research says we were 15-60x under the Atari floor) to test "not training long enough"
for RL from a strong init. Launch when DAgger frees the machine. Then: Expert-Iteration
de-risk (planner-vs-FSM) as the ceiling-breaker.

## 2026-06-16 — WALL BROKEN: aggressive DAgger reached median 4-5 (scores 51k!); long RL launched

aggressive DAgger FINAL (15 iters, 1.8M states): in-process iter15 eval = median 8 (14/15
reach w8) but the robust 25-run eval_continuous = median 4-5, mean 4.4, 12/25 reach w5,
**score mean 51,272 / max 60,150** (vs old ~15-33k). The wall (median 3) is DEFINITIVELY
broken — under-trained imitation, exactly as the user diagnosed. The policy is now
combat-strong (3x score). Best -> models/dagger_mlp_long_best.

LAUNCHED: long RL fine-tune from the median-4.4 DAgger init (models/hvoheja1), 10M steps
(vs prior 3M), ent 0.005 — tests "more RL steps from a strong init" + whether RL pushes
survival deeper / EXCEEDS the FSM. Will eval intermediate checkpoints.
Other queued paths: extend DAgger further (add aggregate save/resume to dagger.py first),
kickstarting, Expert-Iteration de-risk, frameskip-2.

## 2026-06-16 — long RL from DAgger FAILED (degraded); extending DAgger instead

Long RL (10M) from median-4.4 DAgger init: peaked @2.4M (mean 4.3, 9/20 reach w6, 104k
score) then DEGRADED — @3.1M mean 3.6, @3.9M median 3 (24/25 die w3, score 17k).
Catastrophic forgetting: RL optimization drifted the policy off the DAgger competence
(needs kickstarting to anchor — not implemented). Killed. DAgger (median 4.4) stays best.

PIVOT to the robust working path: EXTENDED DAgger (dagger.py 30 iters x 100k; added
aggregate save/resume so it's resumable -> demos/dagger_agg_mlp_x.npz). Pushes pure
imitation toward FSM-level (it was still climbing at 15 iters). Bounded by FSM (~12-16) but
robust + the best lever so far. After: Expert-Iteration (search > FSM) for the path to 100;
and if RL-from-DAgger is revisited, ADD KICKSTARTING (aux CE to live FSM) to prevent the drift.
*** Current best deliverable: models/dagger_mlp_long_best (median 4-5, score ~51k). ***

## 2026-06-16 — Extended DAgger CONFIRMED median 5 (score 141k!); imitation scaling holds

Extended DAgger (30 iters, ~2.7M states). In-process evals spiked to median 8-9 (15-run
over-read), but INDEPENDENT 25-run eval of dagger_mlp_x_best = median 5, mean 5.4, max 8,
9/25 reach wave 7-8, **score mean 63,515 / max 141,875** (new ATH).
IMITATION SCALING (robust 25-run medians): 360k->m3, 1.8M->m4.4, 2.7M->m5; scores 33k->141k.
=> more DAgger data keeps lifting it (diminishing, fidelity-bounded ~87%). Wall (m3)
DECISIVELY broken; now median 5 reaching w7-8. *** Best deliverable: models/dagger_mlp_x_best
(median 5, ATH score 141k). *** Resumable aggregate: demos/dagger_agg_mlp_x.npz.
PATH PAST ~5-6: Expert Iteration (search-as-teacher > FSM) — build next. Also: Implicit-BC
for higher fidelity; kickstarting for non-degrading RL-from-DAgger.

## 2026-06-17 — MEDIAN WAVE 8 (independently confirmed)! Imitation scaling is the answer.

Extended DAgger (30 iters, 3.3M states) FINAL — independent 25-run det evals:
  iter 24: median 5   iter 26: median 7 (score 110k)   iter 30: MEDIAN 8 (24/25 at w8!,
  mean 8.0, score mean 118,766 / max 137,700). The LAST iter is the BEST = STILL CLIMBING.
  (In-process best-selection picked iter 24 by noisy 15-run median — misleading; the true
   best is the final iter 30. Lesson: select best by independent eval, not 15-run.)
ROBUST IMITATION SCALING: median 3 (360k) -> 4.4 (1.8M) -> 5 (2.4M) -> 7 (2.9M) -> 8 (3.3M).
Pure DAgger, just trained longer, broke the "fundamental" median-3 wall and reached the
FSM's Xenia level (8-9). User's under-trained diagnosis fully vindicated.
*** BEST DELIVERABLE: models/dagger_BEST_m8 (median 8, 24/25 reach w8, ATH score 137,700). ***
NEXT: it was still climbing -> RESUME DAgger from demos/dagger_agg_mlp_x.npz for +20 iters
to push toward FSM-on-MAME ceiling (12-16). Then Expert Iteration to exceed the FSM.

## 2026-06-18 — DAgger x3 resume (+20 iters, 3.6M->4.8M): MEDIAN HOLDS 8, SCORE ~2x

WSL OOM-crashed mid-x3-resume on 2026-06-17 (concat held raw aggregate + a full-size
normalized copy = ~27GB sustained at 3.6M). FIXED dagger.py: normalize per-batch on GPU
(no `on` copy; verified max diff 4.8e-7) + np.asarray load. Resumed from the surviving
3.6M aggregate (demos/dagger_agg_mlp_x3.npz), 20 iters x 60k collect -> 4.8M end. Ran
~4h, peak RAM ~31GB (per-iter np.concatenate transient), corruption/OOM = 0. One MAME
instance wedged once and auto-relaunched.

In-process eval climbed 6->8->9->...->iter20 MEDIAN 12 ({9:1,12:14}) -- but that was
INFLATED (MAME state carried over from collection inflates the in-process metric).
*** INDEPENDENT 25-run STOCHASTIC wave-1 eval of models/dagger_x3_best (=iter 20):
    wave median=8 mean=7.8 max=13  dist {3:1,4:2,6:2,7:5,8:8,9:4,11:1,12:1,13:1}
    SCORE mean=121,831  max=274,050. ***
(Det eval is degenerate: deterministic policy + deterministic MAME => 1 trajectory
repeated, w7/114k. Use STOCHASTIC eval for the real distribution. Lesson reaffirmed:
in-process eval over-reads; the headline "median 12" was not real, true median is 8.)

vs prior best dagger_BEST_m8 (median 8, score mean 63.5k / max 141.9k): SAME median wave,
but ~1.9x score mean AND ~1.9x max (new native ATH 274,050) + deeper upside (max w13 vs w8).
*** NEW BEST DELIVERABLE: models/dagger_x3_best (median 8, score mean 122k / max 274k). ***

CEILING REACHED: pure DAgger is FSM-fidelity-bounded at ~median 8. More DAgger refines
score but won't lift median (and the agg is now at the ~5M concat-safe RAM cliff -- a
further resume needs the concat doubling fixed first). PATH PAST median 8 = Expert
Iteration (search-as-teacher > FSM) -- needs building. Decision point for the user.

## 2026-06-18 — GOAL set: wave 1->100 (MAME->Xenia->Xbox). Expert Iteration begun.

User set north-star: RL model playing Robotron wave 1->100, deploy MAME->Xenia->real Xbox.
Current best = median wave 8 (dagger_x3_best); wave 100 needs search-driven play past the
FSM ceiling. Chose ExIt; picked "native in-mem save/restore + sparse search".

NATIVE save/restore PRIMITIVE built (robotron_native/core/runner.cpp rcore_save/load_state
+ robotron_core.py wrappers + core/test_savestate.py): 356KB memcpy, **5.4us save/load**,
deterministic-replay byte-identical. Search-cost bench: 81-action x4-frame ~3.1h/60k-collect,
1-frame ~0.8h -> sparse search is cheap. BUT **native gym is currently frozen** (no enemy
spawn, player stuck, boot jumps into zeroed RAM) -- pre-existing, NOT from this change (my
edits are 3 never-called fns; only other uncommitted change is an inert PROBE_IRQ debug
block). Native ExIt blocked until native gameplay is fixed.

PIVOTED to prototype ExIt on the WORKING MAME gym. MAME savestate via bridge = 2-3ms (fast
enough). Built mame_gym/search_oracle.py (search teacher) + validate_exit_search.py.
KEY ALGORITHM LESSON: naive greedy K-frame search (hold candidate move, score by short-term
score) LOST to the FSM (wave 2 vs 3). The correct ExIt teacher is POLICY IMPROVEMENT:
first frame = candidate move + FSM fire, then FSM as the rollout policy for H frames, score
by SURVIVAL (1000/frame) + score gained, death penalty. That version BEATS the FSM:
*** A/B start=0, 400 steps: FSM wave2/score3,475 vs SEARCH wave4/score9,525 (+2 waves, +6k). ***
(Single deterministic start -- promising but needs broader validation: more starts, longer
episodes.) NEXT: broaden validation, then swap search_oracle into dagger.py collect() labeler
and run a distillation iteration to lift the policy past median 8.

## 2026-06-18 — ExIt distillation WORKS at DUP=1 (search labels lift median 6->8)

Built exit_dagger.py (ExIt distillation: policy drives -> search_oracle labels -> distill,
reusing dagger.py train_bc/eval). FSM base subsampled to 1.5M from dagger_agg_x3 so search
labels matter; search labels accumulate in demos/exit_search_<tag>.npz.

v1 (DUP=20 upweight) REGRESSED: iter0 base median 7 -> iter1 median 3 ({2:3,3:6,4:3,5:3}).
Killed. Ran a controlled DUP sweep (sweep_dup.py) reusing v1's saved 12k search labels +
same 1.5M base:
  DUP=0 (base): median 6 mean 6.6 {3:2,6:10,8:9}
  *** DUP=1:    median 8 mean 7.9 {6:1,7:1,8:19}  <- 19/21 reach w8, big tighten ***
  DUP=5:        median 5 ({5:20,8:1})  (overfit collapse)
  DUP=20:       median 4 ({4:21})      (worse collapse)
LESSON: search labels are HIGH-VALUE PER SAMPLE (12k = 0.8% of data lifted base 6->8 and
tightened the dist) but must be BLENDED 1:1 — upweighting overfits the narrow search set and
collapses generalization. v1's regression was the upweight, not the labels.

LAUNCHED v2: exit_dagger.py 6 12000 12 9978 1500000 v2, DUP=1, resuming from the 12k labels
(demos/exit_search_v2.npz). Watching whether accumulating search labels (12k/iter x6) climbs
PAST median 8 toward the wave-100 goal. search labeling ~360ms/state (H=12) is the throughput
limiter; if the climb stalls on data quantity, next lever = parallelize search-labeling across
MAME instances. Best -> models/exit_v2_best/.

## 2026-06-18 — ExIt-distillation single-instance UNDERPERFORMS; need parallel labeling

v2 per-iter in-process medians were noisy (iter0=6, iter1=4) — the n=15 in-process eval can't
resolve the small effect of a feasible label count. INDEPENDENT 25-run stochastic eval of
exit_v2_best: **median 5, mean 5.3, max 8, score mean 64,394 / max 131,450** — clearly WORSE
than dagger_x3_best (median 8, max 13, score 122k/274k). *** dagger_x3_best REMAINS BEST. ***
DIAGNOSIS (base-size vs label-count dilemma): small base (1.5M, ~median 5-6) lets search labels
matter but is weak; full base (4.8M, median 8) is strong but drowns a feasible 12-24k search
labels at DUP=1 (~0.5%). Resolution = MANY MORE search labels, which single-instance throughput
(~360ms/state) can't supply. UNBLOCK = parallelize search-labeling across N MAME instances
(~16x), then distill search labels at scale onto the full median-8 base. Building that next.
(Search teacher itself is sound — beats FSM +2 waves in A/B; the bottleneck is label THROUGHPUT,
not label quality.)

## 2026-06-18 — Parallel ExIt at SCALE (240k labels) STILL underperforms; thread closing

Built parallel_search_collect.py (12 MAME workers) -> collected 240k search labels at 34.5/s
(11.5x single-instance) -> demos/psearch_p1.npz. Distilled (distill_psearch.py) onto a 2M base
+ 240k search (DUP=1). INDEPENDENT 25-run stochastic eval of models/distill_p1:
  median=5 mean=5.2 max=9 score 63k/136k  -- IDENTICAL to exit_v2_best (24k labels, median 5).
*** Scaling search labels 10x (24k->240k) changed NOTHING. ExIt-distillation is NOT
    label-quantity-bound, and it sits well below FSM-distilled dagger_x3_best (median 8). ***
LIKELY CAUSE: search labels are a noisier / less-learnable distillation TARGET than the FSM's
smooth deterministic potential field — the 1-ply search wins per-state A/B but doesn't make a
better thing to imitate. (Running a clean 2M-base-only DUP=0 control to confirm search adds ~0.)
CONCLUSION: ExIt-via-distillation with a 1-ply search teacher does not beat plain FSM-DAgger.
dagger_x3_best (median 8, score 122k/274k) REMAINS BEST.

CLEAN CONTROL settles it (all independent 25-run stochastic, same 2M FSM base):
  distill_p1base (2M FSM, NO search):  median 7 mean 6.6 max 13 score 94k/274k
  distill_p1     (2M FSM + 240k srch): median 5 mean 5.2 max 9  score 63k/136k  <- WORSE
*** Search labels HURT distillation (median 7->5, score 94k->63k). The 1-ply search teacher
    wins the per-state A/B but is a NOISIER, less-learnable behavioral-cloning TARGET than the
    FSM's smooth potential field. ExIt-via-distillation is a DEAD END. ***
FSM-distillation scales cleanly (2M->med7, 4.8M->med8) -> imitation-on-FSM is the working BC
lever, capped at ~median 8 (teacher level).
*** KEY STRATEGIC TAKEAWAY: behavioral cloning is ceiling-bound at teacher level. Exceeding
    median 8 toward wave-100 requires RL (reward maximization), NOT more imitation. ***
NEXT: RL-from-DAgger WITH kickstarting — warmstart PPO from dagger_x3_best + auxiliary CE-to-FSM
loss to prevent the catastrophic forgetting that degraded the earlier plain RL-from-DAgger
(2026-06-16). Needs implementing the aux loss in the PPO update (train_mame.py). Also still
open: fix the frozen native gym for the Xenia/Xbox deployment leg.

## 2026-06-18 — PIVOT to RL: RL-from-DAgger + JSRL (run ixfe78rh)

ExIt-distillation closed (BC is teacher-ceiling-bound). To EXCEED median 8 -> RL. train_mame.py
already supports warmstart (PPO.load bc_checkpoint) + JSRL (--jsrl-h0, FSM-guided annealing
episode prefix via jumpstart_wrapper.py) — a ready anti-collapse mechanism (vs the unimplemented
aux-CE idea). LAUNCHED (run models/ixfe78rh):
  train_mame.py --bc-checkpoint models/dagger_x3_best/final_model
    --vec-normalize .../vec_normalize.pkl --lr 1e-4 --ent-coef 0.005 --target-kl 0.03
    --gamma 0.999 --jsrl-h0 40 --num-envs 8 --timesteps 2000000 --reset-pool <bc_collect_pool>
Conservative anti-drift config (low lr + target_kl 0.03 + JSRL) to avoid the prior RL-from-DAgger
collapse (which peaked mean 4.3 then degraded). WATCH FOR: ep_rew_mean / wave climbing above the
dagger_x3_best level, NOT collapsing. Judge final by INDEPENDENT 25-run stochastic eval vs
dagger_x3_best (median 8). If it collapses again -> need true aux-CE kickstarting or different RL.

RESULT (independent 25-run stochastic): median 7 mean 7.6 max 12, score mean 122,365 / max 257k.
*** STABLE — NO COLLAPSE. *** Matched dagger_x3_best (median 8, score mean 121,831) — score mean
IDENTICAL, reached wave 12 — and crucially did NOT degrade (vs prior plain RL-from-DAgger that
fell to mean 4.3). The JSRL + low-lr(1e-4) + target_kl(0.03) recipe SOLVES the catastrophic-
forgetting blocker. It held at 7-8 in 2M steps but didn't yet climb past 8 -> the lever now is
CHAINING (more timesteps), as the python-gym PPO chain reached L12/286k over many M steps.
NEXT: chain link 2 from ixfe78rh, exploration-tilted (ent_coef up, target_kl ~0.04, JSRL kept)
to climb past 8 while JSRL anchors against collapse.

## 2026-06-18 — RL chain link2 (g5sav2ny, +3M): STABLE but does NOT climb past 8

Link2 = ixfe78rh + 3M more steps, exploration-tilted (ent_coef 0.01, target_kl 0.04, jsrl-h0 30).
INDEPENDENT 20-run stochastic eval: median 7 mean 6.9 max 13 score 107k/240k.
RL TRAJECTORY: dagger_x3_best(BC) median8/122k -> link1(+2M) median7/122k -> link2(+3M) median7/107k.
*** RL-from-DAgger+JSRL is STABLE (no collapse — the recipe works) but does NOT exceed the BC
    median-8 level in 5M steps; the exploration-tilt slightly HURT (mean/score down). ***
dagger_x3_best (median 8, score 122k/274k) REMAINS the firm BEST across ALL session levers
(BC / ExIt / RL). The median-8 plateau is robust.
HYPOTHESIS for the RL plateau: the MAME env reward may lack strong DEEP-WAVE incentive — the
python-gym chain reached L12 via Option-A step-function wave bonuses (L5+ +1k, L7+ +3k, L10+ +8k).
If the MAME reward's deep-wave gradient is weak, RL has no pull past wave 8. NEXT: audit the MAME
reward (mame_robotron_env.py step()) for deep-wave bonuses; if weak, strengthen and rerun RL.
Operational note: kill MAME with `pkill -x mame` (by NAME) — `pkill -f <pattern>` matches the
launching shell's own cmdline and SUICIDES it (caused a long launch-failure detour today).

## 2026-06-19 — FRONTIER CURRICULUM WORKS: new best (mean 8.1, ATH score 296,950)

Harvested 24 wave-8 states by driving dagger_x3_best (capture_wave5_states.py ... 24 8 5000) ->
w5_5000-5024. Launched frontier-curriculum RL (run models/yiwzpqq7): warmstart dagger_x3_best,
reset-pool the wave-8 states (+ one wave-1 anchor), lr1e-4/ent0.01/target_kl0.04/jsrl-h0 20,
3M steps, --auto-capture-min-wave 9 (harvest the next frontier).
INDEPENDENT 20-run stochastic eval (continuous-from-wave-1):
  median 8.0  mean 8.1  max 13  score mean 138,325 / max 296,950
  dist {2:1,3:1,4:1,6:2,7:4,8:4,9:2,12:3,13:2}  <- 5/20 reach wave 12-13.
*** FIRST lever this session to BEAT dagger_x3_best: same median 8 but mean 7.8->8.1, score
    122k->138k (+13%), NEW ATH 296,950 (was 274k), and MORE deep-wave reaches. ***
=> dense practice at the stuck frontier pushes the upside deeper. NEW BEST = models/yiwzpqq7.
LEVER VALIDATED -> ITERATE: harvest wave-9/10 states (drive yiwzpqq7, which now reaches w12-13),
then frontier-curriculum round 2 from the deeper frontier. This curriculum-bootstrapping
(harvest frontier -> train there -> go deeper -> repeat) is the live path toward wave 100.

ROUND 2 (run hzdfhq7q, wave-9-dominant pool: 20x wave9 + 3x wave8 + 1x wave1) REGRESSED:
median 7 mean 6.9 score 113k/286k (vs round1 mean 8.1). dist {2:1,3:3,6:2,7:6,8:2,9:5,12:1} —
4 runs die at wave 2-3 => the policy OVER-SPECIALIZED on the deep frontier and FORGOT the
wave-1-7 chain (too few wave-1 starts to retain it). *** yiwzpqq7 (mean 8.1) REMAINS BEST. ***
LESSON: the frontier curriculum must keep the FULL chain in the reset-pool (heavy wave-1
retention + frontier), not drill the frontier alone. Round3 = BALANCED pool (lots of wave-1 +
wave-8 + wave-9), warmstart yiwzpqq7. The eval metric is continuous-FROM-WAVE-1, so early-game
competence must be preserved while extending the frontier.

## 2026-06-19 — Frontier curriculum round 3 (rkx7xmi4): REGRESSED, not promoted
- Warmstart yiwzpqq7, BALANCED pool 10×w1 + 7×w8(5005-5023) + 6×w9(6000-6020), lr1e-4 ent0.008 tkl0.035 gamma0.999 jsrl-h0 25, 3M steps, auto-capture-min-wave10.
- Independent 20-run continuous eval (port 9968): mean=7.0 median=7 max=12, score max=267,600 mean=108,839, dist {3:2,4:2,5:1,6:2,7:6,8:2,9:3,12:2}.
- vs best yiwzpqq7 (mean 8.1 median 8 max 13, score mean 138k ATH 296,950) → WORSE on every axis. yiwzpqq7 remains best.
- TWO failures now at the wave-9 frontier (round 2 median 7, round 3 mean 7.0). Shared factor: wave-9 states in pool + warmstart from yiwzpqq7. Round 1 (wave-8 pool) is the only curriculum win.
- HYPOTHESIS: wave-9 starts (harvested from a weaker policy) place the agent in low-recoverability positions → gradient noise that erodes base competence even with heavy wave-1 anchoring.
- ROUND 4 PLAN: warmstart yiwzpqq7, DROP wave-9, consolidate wave-8 (10×w1 + 12×w8 5001-5023), re-harvest FRESH wave-9 from this stronger policy via --auto-capture-min-wave 9. Isolates wave-9-pool vs yiwzpqq7-base as the regression cause.

## 2026-06-19 — Round 4 (sy1mga9v): wave-8 consolidation, no wave-9 — partial recovery, NOT promoted
- Warmstart yiwzpqq7, pool 10×w1 + 12×w8 (5001-5023), NO wave-9, --auto-capture-min-wave 9 (harvested fresh w9 6001-6020).
- Independent 20-run eval (port 9968): mean=7.6 median=7.5 max=11, score max=241,525 mean=124,015, dist {3:2,5:1,6:1,7:6,8:2,9:5,10:1,11:2}.
- vs yiwzpqq7 (mean 8.1): still BELOW, but ABOVE round 3 (7.0). => wave-9 pool contamination IS real (7.0->7.6 by dropping it), BUT further RL from yiwzpqq7 degrades even on the proven wave-8 pool.
- DIAGNOSIS (3 regressions R2/R3/R4 share warmstart-from-yiwzpqq7): yiwzpqq7 is an OVER-CONVERGED local optimum; any further RL drifts it down. The ONE win (R1) warmstarted from dagger_x3_best (headroom). RL fine-tuning saturates after one frontier round per base.
- ROUND 5 PLAN: warmstart dagger_x3_best (proven-improvable base) + pool {heavy w1, w8, FRESH w9}. Decisive: if w9 from a fresh base helps -> new best + deeper; if it regresses too -> RL-curriculum lever EXHAUSTED at 8.1, pivot to architectural lever (recurrent/LSTM policy for deep-wave temporal reasoning).

## 2026-06-19 — Round 5 (94t3tc9g): dagger_x3_best + w1+w8+w9 — REGRESSED (6.3), not promoted
- Warmstart dagger_x3_best, pool 10×w1 + 6×w8(5003-5023) + 5×w9(6003-6019), 3M.
- Independent 20-run eval: mean=6.3 median=7 max=10, score max=174,725 mean=90,244, dist {3:3,4:2,5:3,6:1,7:5,8:3,10:3}.
- CONCLUSION (5 rounds): EVERY pool with wave-9 states regressed (R2 7.0, R3 7.0, R5 6.3), across BOTH bases. Wave-9 ENTRY states = unrecoverable starts -> near-instant deaths -> destructive gradients. Frontier-by-deeper-states DEAD END past wave-8.
- Only winning config: R1 dagger_x3_best + WAVE-8-ONLY -> yiwzpqq7 8.1 (REMAINS BEST, ATH 296,950).
- OPERATIONAL: avoid pkill -f / pgrep -f with a pattern that appears in my own command (self-match -> exit 144 kills the launching shell). Use pkill -x mame, or the bracket trick ps -eo pid,args | grep '[e]val_continuous'.
- ROUND 6 (decisive RL-convergence test): dagger_x3_best + 10×w1 + 12×w8 (NO w9), 5M steps. If >8.1 -> longer-training is the knob; if ~8 -> RL exhausted, pivot to recurrent/LSTM.

## 2026-06-19 — Round 6 (dajy9vg0): dagger+w8, 5M steps — DID NOT break plateau (7.4), not promoted
- Warmstart dagger_x3_best, pool 10×w1 + 12×w8 (NO w9), 5M steps (vs 3M).
- Independent 20-run eval: mean=7.4 median=7 max=12, score max=272,700 mean=130,281, dist {2:1,3:2,5:1,6:1,7:8,8:1,9:1,10:1,11:2,12:2}.
- 5M came in BELOW the 3M version (yiwzpqq7 8.1) — MORE steps over-trained & regressed. yiwzpqq7 REMAINS BEST (ATH 296,950).
- **RL-CURRICULUM LEVER EXHAUSTED (6 rounds conclusive):** RL fine-tuning (945-dim slot obs + MLP[512,512], frameskip-4, Option-A reward) saturates at mean ~8. More steps, deeper states, re-warmstart all FAIL to exceed 8.1. Churning RL variants is negative-EV.
- PIVOT: diagnose the wave-8 wall root cause (obs-capacity vs policy-memory) before choosing next lever (expand obs vs recurrent/LSTM). Failure signature "dies surrounded by converging grunts".

## 2026-06-19 — Grid-CNN from-scratch (9iwa3zzv): mean wave 3.0 — from-scratch is the wrong approach, KILLED
- Architectural pivot: --obs-mode grid (SpatialGridObsBuilder (11,72,48) + GridCNN), from scratch, wave-1 pool, GPU (RTX4080, ~454fps).
- Trained to 6.7M/12M. Training "highest_wave" reached 6 but that's a sticky all-time MAX.
- DIAGNOSTIC eval (10-run stochastic continuous, 6.7M ckpt): mean=3.0 median=3 max=4, score mean 9,062. FAR below yiwzpqq7 (8.1, 138k).
- ROOT CAUSE: from-scratch RL walls at ~wave 3 (matches spatial_obs.py docstring "walled at wave ~3"). The slot policy only reached 8 via DAgger-on-FSM BC WARMSTART; pure from-scratch slot RL also walls ~3. So grid-from-scratch walling at 3 is expected.
- KILLED at 6.7M (mean-3 trajectory wouldn't reach 8.1 by 12M; ~3.5 GPU-hr better spent on the right approach).
- NEXT LEVER: BC-warmstart the grid CNN — collect FSM demos in GRID obs, BC-pretrain GridCNN PPO, then RL (mirrors dagger_x3_best->yiwzpqq7). Tests the grid's TRUE ceiling (richer obs) vs slot's 8.1.

## 2026-06-19 — Grid BC warmstart + RL fine-tune (the grid recipe)
- collect_grid_demos.py: 480k FSM grid demos (12 workers, ~2024/s, float16 compressed shards demos/grid_v1_shards, ~87MB). FSM labels packet-based (obs-agnostic).
- bc_grid.py: BC-pretrained GridCNN (MlpPolicy+GridCNN feat-extractor, net [512,256], matches train_mame grid branch). 12 epochs, NLL 2.84->0.19 (strong FSM fit). -> models/grid_bc_v1.
- grid_bc_v1 standalone eval (12-run stochastic): mean=2.5 median=2.5 max=3, score 10,227. WEAK — single-pass BC compounds errors (slot's dagger_x3_best used 3 DAgger iters to reach median 8). Expected; BC->RL is the plan.
- RL fine-tune from grid_bc_v1: run models/xrd7uq48, --obs-mode grid --device cuda, lr3e-4 ent0.01 gamma0.999 reset-pool 0, 12M, auto-capture-min-wave 5. Same HPs as from-scratch grid (9iwa3zzv) so improvement is attributable to BC init. KEY TEST: does RL-from-BC beat from-scratch (mean 3 @6.7M) and slot's 8.1? If BC init too weak, add DAgger iters on grid next.

## 2026-06-19 — Grid BC->RL (xrd7uq48): stuck at mean 2.4, KILLED. Grid obs = dead end at feasible compute.
- RL fine-tune from grid_bc_v1 (lr3e-4 ent0.01): 600k eval mean 2.1, 2.3M eval mean 2.4 {2:6,3:4}. NOT climbing; below from-scratch trajectory.
- Grid climbs ~0.1 mean-wave per 1M steps (from-scratch 2.5->3.0 over 4M). Reaching 8.1 from 2.5 needs ~50M+ steps — INFEASIBLE per run.
- ROOT CAUSE: slot obs IS the FSM's native input (nearest-N per category) -> slot BC trivially ~8. Grid forces CNN to re-derive threat features -> far less sample-efficient. Full-grid replacement is a dead end.
- NEXT LEVER (hybrid): keep 945-dim slot obs (strong BC ~8) + APPEND a compact global-threat summary (enemy density in angular sectors / safe-direction) -> addresses 63%-open-field-death root cause; RL can exploit the extra features to push PAST 8. BC reproduces FSM (~8, ignoring global feats since FSM doesn't use them); RL discovers their value via reward.

## 2026-06-19 — HYBRID obs (945 slot + 12 global) pipeline built + launched
- hybrid_obs.py: HybridObsBuilder = 945 slot obs + 8 directional threat-pressures + 4 wall-openness = 957-dim. Validated on MAME (sectors in [0,1], wall feats complementary). Wired obs_mode="hybrid" into env + train_mame (MlpPolicy path) + eval_continuous.
- collect_hybrid_demos.py: 1.2M FSM hybrid demos (~2674/s, float32 shards demos/hybrid_v1_shards). bc_hybrid.py: MlpPolicy [512,512] + VecNorm(norm_obs), 14 epochs NLL 2.39->0.84 -> models/hybrid_bc_v1.
- hybrid_bc_v1 standalone eval (12-run): mean ~3.3 median 3 max 6 dist {2:5,3:2,4:4,6:1}. Better than grid BC (2.5), capability to w6, but single-pass (slot's median-8 used 3 DAgger iters). 6MB MlpPolicy (vs grid 61MB CNN).
- RL fine-tune from hybrid_bc_v1: run models/j1es2u1m, --obs-mode hybrid, lr3e-4 ent0.01 gamma0.999 cpu reset-pool 0, 12M, auto-capture-min-wave 5, NO jsrl (slot-obs-only). KEY: hybrid is MLP-friendly (slot-native) so RL should behave like the PROVEN slot RL (unlike grid CNN which failed); global feats let it exploit open-field awareness to beat 8.1. Eval checkpoints (verbose=0, no tables).

## 2026-06-19 — Hybrid BC->RL (j1es2u1m): stuck ~3.4, KILLED. RL-doesn't-climb is now ironclad.
- 2M eval mean 3.2, 3.2M eval mean 3.4 {3:8,4:3,5:1} — flat vs BC 3.3. Same as grid.
- DEFINITIVE (slot/grid/hybrid): RL barely climbs over its BC start on this gym (slot strong-BC 8.0->8.1; grid/hybrid weak-BC stay flat). Results are BC-CAPPED at FSM fidelity (~8).
- => To beat 8.1, do NOT rely on RL-from-weak-BC. NEXT (efficient): WEIGHT-TRANSPLANT dagger_x3_best (slot, median 8) into a hybrid policy (copy all layers; first-layer W[:, :945]=slot, W[:, 945:957]=0; VecNorm = slot 945 stats + 12 global stats from hybrid demos). Hybrid then STARTS at 8 (global feats contribute 0) and RL can learn to use the 12 global feats to exceed 8. Cleanest test of the hybrid thesis from a strong base, no BC rebuild.

## 2026-06-19 — Slot->Hybrid TRANSPLANT validated; RL fine-tune launched (the clean hybrid test)
- transplant_slot_to_hybrid.py: grafted dagger_x3_best (slot, median 8) into a 957-dim hybrid policy (10 params copied, 2 first-layer padded 945->957 with global cols ZEROED, 0 skipped). VecNorm = slot 945 stats + 12 global stats from hybrid demos. -> models/hybrid_transplant.
- Transplant validation eval (12-run stochastic hybrid): mean 8.9 median 9 dist {7:5,9:4,11:1,12:1,13:1} — reproduces strong slot play (global feats zero-weighted, no interference). CONFIRMS surgery correct. Strong hybrid base achieved WITHOUT BC rebuild.
- RL fine-tune from transplant: run models/17ti39jd, --obs-mode hybrid, lr1e-4 ent0.01 tkl0.035 gamma0.999 cpu, pool 10×w1+12×w8, 6M, auto-capture-min-wave 9, no jsrl. Mirrors winning R1 recipe + global features. KEY TEST: can RL learn to use the 12 global-threat features to push past 8.1? Watch for regression (yiwzpqq7-restarts regressed) vs climb. Eval checkpoints (verbose=0).

## 2026-06-19 — Transplant RL (17ti39jd): REGRESSED 8.9->6.9, KILLED. RL-regresses-strong-bases is definitive.
- 1.7M eval mean 6.9 median 7 {3:1,5:3,6:2,7:5,9:2,10:2} — DOWN from transplant baseline 8.9. Same pattern as yiwzpqq7-restarts (R2/R3/R4).
- DEFINITIVE SESSION CONCLUSION: RL fine-tuning on this MAME gym REGRESSES strong bases (yiwzpqq7-restarts, transplant) and does NOT lift weak ones (grid/hybrid BC). The ONLY marginal RL win ever was R1 (dagger_x3_best 8.0->yiwzpqq7 8.1, +0.1). RL is NOT a viable lever past the FSM-fidelity ceiling (~8) on ANY obs (slot/grid/hybrid).
- The ceiling is the TEACHER: BC clones the obs-limited FSM (~wave 8). To exceed wave 8 needs a BETTER teacher (full-info FSM or search) cloned onto an obs rich enough to support it (hybrid). ExIt search-distillation already failed earlier. Full-info-FSM + hybrid + DAgger is the one principled untried lever.
- ARTIFACTS preserved: models/hybrid_transplant (mean 8.9 hybrid-native policy = dagger_x3_best re-expressed, slot ceiling, global feats zeroed) — ready base for a better-teacher experiment. Full hybrid pipeline built: hybrid_obs.py, collect_hybrid_demos.py, bc_hybrid.py, transplant_slot_to_hybrid.py.
- BEST DELIVERABLE REMAINS models/yiwzpqq7 (slot, mean 8.1, ATH 296,950).

## 2026-06-19 — PIVOTAL REFRAME: the bottleneck is BC FIDELITY, not obs/teacher/RL.
FSM teacher evals (deterministic, fsm_choose/fsm_obsgap on MAME, 8 runs):
- obs-limited FSM (the nearest-N selection BC clones): mean 11.4 median 12 {9:3,12:4,16:1}
- full-info FSM (all entities): mean 15.1 median 16 {9:1,16:7} (16 = step cap, survives)
Policy evals:
- dagger_x3_best (BC): stochastic ~8, DETERMINISTIC mean 7.3 median 7 {7:11,11:1}
- yiwzpqq7 (RL): stochastic 8.1, DETERMINISTIC mean 4.8 {4:6,5:5,8:1} (RL made argmax WORSE — sampling-dependent)
KEY: the slot obs is SUFFICIENT (FSM hits 11.4 on the very nearest-N the obs conveys), but our cloned policy only reaches ~8 — a ~3-4 wave BC-FIDELITY GAP. This is the long-standing "DAgger plateaued at 8" ceiling, now quantified: ~3 waves of headroom remain on the EXISTING obs if BC could match the FSM. RL is counterproductive (doesn't climb; hurts determinism). 
=> Levers to close 8->11 (all on existing slot obs, cheap): (a) bigger/deeper policy net (current [512,512] may be capacity-limited for deep-wave precision); (b) more DAgger iters with deep-wave (8-12) emphasis; (c) train/eval DETERMINISTIC (yiwzpqq7's RL-stochasticity is a trap). full-info FSM (15.1) is a further ceiling but needs a richer obs to clone.
BEST DELIVERABLE: models/yiwzpqq7 (stochastic 8.1) — but dagger_x3_best is the cleaner base for closing the BC gap.

## 2026-06-19 — bc_bignet = NEW BEST (det 8.1 / stoch 8.8); BC gap cause = deep-wave data coverage
- bc_bignet ([1024,1024,512] BC on 4.8M dagger_agg_x3, 14 ep): DETERMINISTIC mean 8.1 med 8 {7:1,8:10,10:1}; STOCHASTIC mean 8.8 med 8 {6:2,7:3,8:2,9:1,10:1,12:2,13:1}.
- NEW BEST: beats yiwzpqq7 on both (stoch 8.8>8.1; det 8.1>>yiwzpqq7 det 4.8). Clean BC policy, robust deterministically. Bigger net helped only +0.8 -> capacity is a MINOR factor.
- BC gap ROOT CAUSE found: bc_collect_pool.txt = 100 states, 40 wave-1 + 60 mid (wave2-5), ZERO wave-8+. So the aggregate undersamples deep-wave (8-12) states -> policy never got dense deep supervision, even though obs-limited FSM plays there (11.4). 
- NEXT: collect FSM demos RESET FROM wave-8/9 states (5001-5024, 6001-6020) for dense deep-wave supervision; add to aggregate; re-BC bignet. Target: close 8->11 gap.

## 2026-06-19 — bc_deep (naive deep-augment): REGRESSED. bc_bignet stays best.
- bc_deep (3.5M agg subsample + 1.2M deep-wave demos, bignet): DET mean 6.0 {4:7,7:7,13:1}; STOCH mean 7.8 {4:3,7:3,8:4,9:2,10:1,12:2} score max 270k. WORSE than bc_bignet (det 8.1/stoch 8.8) — introduced wave-4 early deaths (bc_bignet det was {8:10}, none early).
- CAUSE: lost early coverage (4.8M->3.5M) + deep over-weighted (25% of mix) -> early-wave fragility. The policy CAN play deep (w13/w12 runs) but dies early too often.
- REFINEMENT: FULL aggregate (preserve early) + MODEST deep supplement (~8%, ~400k). Run bc_deep2.
- BEST: models/bc_bignet (det 8.1, stoch 8.8) — clean bigger-net BC, the session's banked improvement over yiwzpqq7.

## 2026-06-19 — DEEP-AUGMENTATION FAILED (both variants); bc_bignet is the session's best.
- bc_deep2 (4.5M early + 400k deep, ~8%): DET mean 6.7 {3:1,7:14} / STOCH mean 7.7 {2:1,3:1,5:1,7:5,8:2,9:1,10:1,12:3}. Worse than bc_bignet on both. Modest deep made it consistent-at-7 (capped LOWER than bignet's 8).
- VERDICT: adding wave-8/9-teleport-start FSM demos HURTS (naive 25% AND modest 8% both regressed). Likely the save-state-start obs distribution differs from naturally-reached deep states + pollutes mid-wave behavior. Deep-data lever DEAD.
- FINAL BEST: models/bc_bignet ([1024,1024,512] BC on 4.8M dagger_agg_x3) — DET 8.1 {7:1,8:10,10:1}, STOCH 8.8 (to w13). Beats yiwzpqq7 (det 4.8/stoch 8.1) — much more robust deterministically.
- OPEN: bc_bignet dies VERY consistently at wave 8 ({8:10} det) while obs-limited FSM reaches 11.4 on the same obs. Specific reproducible wave-8 failure -> next diagnostic = death-forensics at wave 8 (why does the policy die there but the FSM survives?). Other lever: clone full-info FSM (15.1) but needs richer trainable obs (grid failed).

## 2026-06-19 — Wave-8 death FORENSICS on bc_bignet (deaths_bignet, 95 deaths, 100% explained-direct):
- Killer by wave: w5 Brain(alt) 29/29; w6 Grunt 10/Enforcer 1; w7 Quark 27/29; w8 Grunt 20/22 (Hulk 2).
- Wave-8: nearest-suspect distance mostly d=4 (point-blank, 11/22) and d=11 — the "surrounded/cornered by converging grunts" failure (env comment: "real fix is better MOTION").
- DIAGNOSIS: the BC-fidelity gap is DIFFUSE threat-avoidance — the policy approximates the FSM's evasion but its motion/positioning is slightly worse, so it gets caught by Brains(w5)/Quarks(w7)/Grunts(w8) the precise FSM evades. No single fixable mechanic. This is the BC approximation quality limit (DAgger plateaued at 8; bigger net +0.8; deep-data hurt).
- Candidate fresh levers (uncertain, multi-step): (a) FRAME-STACKING / temporal obs so the policy sees enemy trajectories for anticipatory evasion (better MOTION); (b) clone the full-info FSM (15.1) on a richer trainable obs (grid failed). 
- bc_bignet (det 8.1 / stoch 8.8) is near the practical BC ceiling for the current slot-obs+MLP approach.

## 2026-06-20 — BREAKTHROUGH: BC-ANCHORED RL WORKS. RL improves past wave 8.
- train_bc_anchored.py v3 (CHUNKED PPO/BC alternation + SEPARATE BC optimizer; start=bc_bignet): ck_458k DETERMINISTIC 20-run eval mean=9.8 median=8.5 max=12 {8:10,9:1,12:9} score mean 159,714. vs bc_bignet det 8.1 -> +1.7 waves, NO regression (min wave 8), 9/20 runs reach wave 12.
- This REFUTES "RL is dead": earlier RL regressions were (a) too-weak anchor (v1: 64 BC steps vs PPO 2560 -> dipped 6.9), (b) BC in on_rollout_end breaking PPO on-policy ratio (v2 -> 4.4), (c) undertrained naive PPO. v3 fixes all: anchor holds the floor at 8, RL pushes ~half the runs to 12.
- NEW BEST: models/bcanchor_v3/checkpoints/ck_458k (det 9.8) — pending continued-climb check on later checkpoints + stochastic eval.
- KEY RECIPE: PPO fine-tune of a strong BC policy MUST anchor to the BC/FSM demos (separate optimizer, chunked phases) to avoid catastrophic forgetting; then RL genuinely climbs. lr1e-4 ent0.005 ppo_chunk32768 bc_steps150 bc_lr1e-4.

## 2026-06-20 — Anchored-RL iteration: 9.8 is a ONE-SHOT peak (can't iterate RL on RL).
- v4 (from 9.8 peak, over-stable lr5e-5 anchor250): det 8.0 {7:1,8:19} — too anchored, pinned at BC level, no climb.
- v5 (from 9.8 peak, lr1e-4 anchor200): COLLAPSED det 2.2 {2:19,7:1} — RL on the already-RL-tuned peak destroyed it (bc_nll stayed 0.135 but policy collapsed -> demo-NLL != play quality).
- LESSON (consistent w/ session): anchored RL climbs ONCE from a CLEAN BC base (v3: bc_bignet 8.1 -> 9.8 peak @458k) but RL-on-RL-tuned-policy collapses (v5, like yiwzpqq7-restarts). The +1.7 wave gain is a ONE-SHOT climb, captured by early-stopping at the peak.
- DEFINITIVE BEST: models/bcanchor_best (v3 ck_458k, det 9.8 / max wave 12). Proves RL improves past wave 8.
- To exceed 9.8: need a BETTER BC BASE (bc_bignet 8.1 -> higher) then one-shot anchored RL; OR the from-scratch grid run (xb39hf1u, still cooking). Micro-tuning RL-from-peak is a dead end.

## 2026-06-20 — Re-distillation attempt: lossy clone (6.7 bimodal), iterate-past-9.8 needs cleaner clone.
- collect_policy_rollouts.py: 1M bcanchor_best DET rollouts (eps=0.08 for state diversity) -> demos/redistill_v1_shards.
- bc_redistill.py: [1024,1024,512] BC clone, NLL->0.28. Eval det 20-run: mean 6.7 median 8.5 {3:9,7:1,10:10} — BIMODAL: captured DEEP capability (10/20 reach wave 10) but LOST early-wave (9/20 die wave 3). Clone 6.7 < bc_bignet 8.1 base, so redistill-then-anchor base is WORSE, not better.
- CAUSE (likely): eps=0.08 random steps during collection contaminated early-wave labels. FIX for next time: re-collect eps=0 (pure det) OR BLEND FSM early-wave demos (dagger_agg) + redistill DEEP demos so clone is good early AND deep. Deferred (diminishing returns, late session).
- STATUS: bcanchor_best (det 9.8) remains the BANKED BEST + proof RL works. from-scratch grid (xb39hf1u) still cooking (wave 5 @5.6M). Iterate-past-9.8 via cleaner re-distillation is the clear next lever.

## 2026-06-20 — Re-distillation SHELVED: clones are lossy (≤7), can't iterate past 9.8.
- Pure-redistill clone (eps0.08): det 6.7 {3:9,10:10} — deep but early-broken.
- Blended clone (3.5M FSM-early + 1M 9.8-deep): det 6.9 {5:1,7:19} — early-fixed but FSM dominated -> capped at 7, lost deep.
- BOTH clones < bc_bignet (8.1). CONCLUSION: BC cannot cleanly clone the 9.8 policy (its edge over bc_bignet is RL-specific, not distillable). So iterate-past-9.8 via re-distillation is a DEAD END, as is RL-on-RL (v5 collapse). 9.8 is the ONE-SHOT ceiling of anchored RL from the bc_bignet base.
- To exceed 9.8 (future): (a) a better CLEAN BC base than bc_bignet 8.1 (the BC-fidelity problem — bigger net gave +0.8, deep-data failed); (b) the from-scratch grid run (xb39hf1u) reaching high on its own; (c) much larger single-shot anchored-RL run; (d) recurrent policy. 
- DEFINITIVE SESSION DELIVERABLE: models/bcanchor_best (det 9.8 / max wave 12) — RL improves the BC policy past wave 8. RL is NOT dead (the session's earlier conclusion was wrong; it was bugs+undertraining).

## 2026-06-21 — From-scratch grid RL (xb39hf1u) CONCLUSIVE at 40M: mean 2.6.
- 40.4M-step checkpoint, 12-run stochastic continuous eval: mean=2.6 median=3 {2:5,3:7}. Training highest_wave touched 6 (single outlier episode), but real mean caps ~2.6.
- CONCLUSION (the "can RL learn from nothing?" test): pure from-scratch RL on grid obs learns (0->~wave 3) but is massively sample-inefficient and plateaus ~mean 2.6 even at 40M steps — far below BC-warmstart+anchored-RL (bcanchor_best det 9.8). The BC warmstart is ESSENTIAL; from-scratch is not viable at feasible compute.
- Run continues to 50M per "don't kill early" but result won't change. BEST=models/bcanchor_best (det 9.8).

## 2026-06-21 — Disk-fill recovery (C->D move) + bigger anchored-RL launched (bcanchor_big1)
- Disk filled mid-session; repo moved C->D (now /dev/sdd, 827G free). The grid run xb39hf1u (tfevents writing until 05:38) was the process killed — but it was an ALREADY-CONCLUDED dead end (from-scratch grid, mean 2.6 @40M). Nothing of value lost.
- Post-move verification: .venv+CUDA (torch 2.12) OK; MAME ROM+binary OK; gym boots/resets/steps live (945-dim obs, scoring). Key models intact: bcanchor_best (9.8), bc_bignet (8.1), yiwzpqq7, bcanchor_v3.
- LEVER CHOSEN (user): bigger single-shot anchored-RL — scale the ONLY recipe that beat the wave-8 ceiling (v3: bc_bignet 8.1 -> 9.8 one-shot peak @458k). Hold the proven RL:BC anchor ratio constant, double env diversity + per-cycle experience, checkpoint denser to not miss the peak.
- RUN models/bcanchor_big1: ANCHOR_BASE=bc_bignet, NENV=16 (was 8), PPO_CHUNK=65536 (was 32768; still 2 rollouts/cycle at 16 envs), BC_STEPS=300 (was 150; doubled to preserve ratio), lr1e-4 ent0.005 bc_lr1e-4, wave-1 reset pool [0], 6M steps, CK_EVERY=100k. port 9942. log logs/bcanchor_big1.log.
- First cycle healthy: done=65,536 bc_nll=0.183 (v3 productive region ~0.135-0.19). 16 MAME up, 16GB RAM (under 48GB cap), GPU active.
- EVAL PROTOCOL for checkpoints (how 9.8 was measured): `.venv/bin/python mame_gym/eval_continuous.py models/bcanchor_big1/checkpoints/ck_<N>k.zip models/bcanchor_big1/checkpoints/vec_<N>k.pkl 20 <port> det slot`. Promote to bcanchor_best only if DET 20-run mean > 9.8 with no early-wave (<8) collapse.
- GOAL: chain toward playing wave 1->100. Autonomous loop monitors big1, evals each checkpoint, promotes winners, picks next lever (recurrent/LSTM or better BC base) if big1 plateaus.

## 2026-06-21 — NEW BEST: bcanchor_big1 ck_524k det 20-run mean=10.1 (was 9.8). Bigger anchored-RL works.
- Checkpoint sweep (det continuous, wave-1, full lives): ck_393k 12-run mean 7.7 (early-perturbation dip below bc_bignet 8.1, as v3 did); ck_524k 12-run mean 10.0 -> 20-run CONFIRM mean=10.1 median=10 min=8 max=12 dist={8:6,10:5,11:4,12:5} score_mean=169,726 max=217,575.
- PROMOTED ck_524k -> models/bcanchor_best (old 9.8 backed up as final_model_9p8.zip / vec_normalize_9p8.pkl). NEW ATH det mean 10.1 (+0.3 waves over v3's 9.8), and a TIGHTER floor: every run >=wave 8, 9/20 reach wave 11-12 (v3 was {8:10,9:1,12:9}).
- VALIDATES the "bigger single-shot anchored-RL" thesis: scaling envs 8->16 + per-cycle experience 2x (PPO 65536/BC 300, ratio preserved) lifted the one-shot anchored-RL peak 9.8 -> 10.1. bc_nll kept descending (0.183->0.144 through 720k), so later big1 checkpoints (ck_655k+) are still being evaluated for a higher peak.
- NEXT (loop-driven): finish evaluating big1's later checkpoints for the running peak; on run-finish, warmstart a big2 anchored run from the new bcanchor_best base to test whether the climb compounds toward deeper waves.

## 2026-06-22 — big1 keeps CLIMBING then OSCILLATES: ck_786k det 20-run mean=10.4 = NEW BEST (ATH score 294,625).
- ck_786k 12-run 10.7 -> 20-run CONFIRM mean=10.4 median=12 min=8 max=14 dist={8:9,12:9,14:2} score_mean=198,460 max=294,625. PROMOTED -> models/bcanchor_best (supersedes 10.1). Score 294,625 ~ matches old ATH 296,950 (yiwzpqq7) but on a det wave-1 chain, not save-state-started.
- Distribution is now BIMODAL: ~half the runs still hit the wave-8 wall (min 8), the other half break through to wave 12-14. The bigger anchored run is genuinely pushing the upper mode deeper (v3 capped at 12; big1 reaches 14).
- BUT checkpoint-to-checkpoint OSCILLATION: ck_917k (next ck) collapsed to mean 7.0 {7:12} while bc_nll stayed flat at 0.140. Anchored-RL play quality is noisy and NOT tracked by demo-NLL (consistent with the v5 collapse lesson). => keep the running MAX across checkpoints (ck_786k=10.4 banked); eval every new checkpoint, promote only confirmed improvements. The peak is captured by checkpointing, not by the final model.
- Loop continues: eval ck_1048k+ keeping running max; on big1 finish, launch big2 warmstarted from bcanchor_best (10.4).

## 2026-06-22 — big1 WRAPPED at 1.9M (peak ck_786k=10.4 confirmed); big2 launched from the 10.4 base.
- big1 checkpoint sweep (det 12-run) past the peak: 917k=7.0, 1179k=9.8, 1310k=7.0, 1441k=6.3, 1572k=8.2, 1703k=7.4. ck_786k (10.4) UNBEATEN over 7 consecutive checkpoints. Anchored-RL play quality oscillates wide (6.3-10.4) with bc_nll flat ~0.12-0.13 — the peak is a checkpoint to be captured, not a convergence point. 786k is big1's peak; verdict matches v3's single-peak shape but later (786k vs 458k) and higher (10.4 vs 9.8) — the bigger run genuinely raised the ceiling.
- Stopped big1 early (peak already banked as bcanchor_best) rather than burn compute to 6M. Cleanup gotcha: `pgrep -f train_bc_anchored` SELF-MATCHES the shell (PID changes each call — looks like a lingering proc but isn't); verify with bracket trick `ps -eo pid,args | grep '[t]rain_bc_anchored'`. Orphaned MAME needed `pkill -9 -x mame`.
- big2 = anchored-RL warmstarted from bcanchor_best (ck_786k, 10.4), SAME recipe (NENV=16, PPO 65536, BC 300, lr1e-4 ent0.005, FSM-demo anchor dagger_agg_x3, wave-1 pool), 6M, CK_EVERY=100k, port 9952. log logs/bcanchor_big2.log.
- KEY TEST + RISK: the BC anchor is still the ~wave-8-fidelity FSM demos, but the base is now 10.4. Either RL drives big2 past 10.4 (anchor = stabilizer) OR the FSM-demo anchor drags the 10.4 base back toward ~8 (anchor caps at teacher fidelity). Both outcomes are informative for the next lever. Loop monitors big2 (checkpoints models/bcanchor_big2/checkpoints, evald logs/big2_evald.txt, BAR=10.4).

## 2026-06-22 — VERDICT: anchored-RL does NOT compound (big2 regressed to FSM-teacher ~8). bcanchor_best=10.4 is the deliverable.
- big2 det 12-run sweep (warmstart from the 10.4 base): ck_393k=9.9, ck_524k=8.0, ck_655k=8.3, ck_786k=7.7 (the step where big1 PEAKED at 10.4). MONOTONIC DECLINE from the 10.4 warmstart down to ~8, max-wave capping at 9 (big1's good checkpoints reached 12-14). The FSM-demo anchor pulled the strong base back to teacher fidelity (~8).
- DEFINITIVE: the "bigger anchored-RL" lever is a ONE-SHOT lift — bc_bignet 8.1 -> big1 ck_786k 10.4 (and v3 9.8). Re-chaining anchored-RL FROM the lifted base does NOT compound; it regresses to the ~wave-8 FSM-teacher ceiling because the anchor IS those ~wave-8 demos. So chaining big3/big4 from each new best is futile (confirmed, not assumed).
- NET SESSION RESULT: det mean wave 9.8 -> 10.4 (+0.6), ATH score 294,625 (det wave-1 chain, ~matches old save-state ATH 296,950). bcanchor_best = models/bcanchor_best (big1 ck_786k). big2 stopped at ~800k (dead end), MAME cleaned.
- DECISION: do NOT churn more anchored-RL variants (lever explored across v3/big1/big2; user pref = architecture over RL-hyperparam churn). PIVOT to the RECURRENT/LSTM lever — principled fix for the wave-8 wall (death forensics = "cornered by converging grunts" = needs anticipatory evasion / better MOTION, which temporal memory provides; the slot obs is a positional snapshot with no velocity).
- RECURRENT SCOPE (sb3_contrib 2.7.0 RecurrentPPO available; eval_continuous.py already supports "rnn"): (a) collect FSM demos PRESERVING episode order (sequences, not shuffled) — adapt mame_gym/collect_fsm_demos.py to write per-episode obs/action/done sequences; (b) bc_recurrent.py = BC-pretrain a MlpLstmPolicy (945-dim obs) on those sequences via truncated-BPTT; (c) sanity-eval the recurrent BC base with eval_continuous ... det slot rnn; (d) IF the recurrent BC base is >= the non-recurrent BC (~8), anchored-RL it (recurrent variant of train_bc_anchored). Gate each step; do NOT launch the multi-hour recurrent train until the BC base sanity-evals reasonably. Build incrementally, validate each piece.
## 2026-06-22 — Recurrent/LSTM build: 1.2M seq demos collected, recurrent BC training launched.
- collect_fsm_demos_seq.py (12 workers, eps0.05): 1.2M episode-ordered slot demos+dones -> demos/seq_v1_shards (6.8 min, 2933/s). Smoke-validated shard format (obs f16, actions i8 0-7, dones bool, clean episode splits).
- bc_recurrent.py: BC-pretrain RecurrentPPO MlpLstmPolicy (lstm_hidden 256, net [256,256]) on L=32 chunks (hidden reset per chunk; ~2s motion context). 37,138 chunks. Bug found+fixed in smoke (_StubEnv must subclass gym.Env). Smoke 1-epoch bc_nll 4.16(rand)->2.58 = learning. Full run = 10 epochs batch64 lr3e-4 -> models/rbc_v1 (reuses bc_bignet vec_normalize). Launched, epoch1 bc_nll=2.72.
- NEXT (loop): on BC finish, sanity-eval `eval_continuous ... det slot rnn`. If recurrent BC base ~>=7-8 (near bc_bignet 8.1) the LSTM clone works -> build recurrent anchored-RL (push past 10.4 via anticipatory motion). If <6, clone lossy -> diagnose seq_len/lr/epochs first.

## 2026-06-22 — Recurrent BC v1 (L=32 chunk-reset) FAILED eval (mean 2.0, dies wave 2); ROOT CAUSE + fix.
- rbc_v1 first attempt: BC on L=32 chunks with hidden RESET every 32 steps. bc_nll 2.72->1.24, but det 12-run sanity eval = mean 2.0, ALL runs die wave 2 (uniform -> not under-fit, a systematic failure).
- ROOT CAUSE: TRAIN/INFERENCE hidden-state MISMATCH. Trained with hidden reset every 32 steps, but eval_continuous(rnn) carries the hidden state across the WHOLE episode (~1500 steps). After ~32 steps the LSTM hidden state drifts into a regime never seen in training -> policy collapses ~wave 2.
- FIX (bc_recurrent.py rewritten): train on WHOLE EPISODES from a zero init state (== inference). Episodes are LONG (~1213 mean, up to 2048 cap; FSM survives deep), so length-bucket episodes (sort by len, pad per-batch) and use the FAST cuDNN path via all-zero episode_starts + zero init (each row processed full from zero = reset-at-step-0, vectorized). Masked loss over real steps. Now uses 899k of 1.2M steps (741 episodes).
- WATCH: only 741 distinct episode trajectories — small for an LSTM. If the corrected base still evals weak, collect more seq demos (more episodes) before blaming the architecture. rbc_v1 retraining (20 epochs), eval pending.

## 2026-06-22 — Recurrent lever DIAGNOSED: not the architecture, the SINGLE-PASS data. (Decisive control.)
- Corrected whole-episode rbc_v1 STILL evaled mean 2.0 {2:12} (stoch 2.2). Eval rnn threading verified correct; not a det-loop (stoch also ~2.2).
- DECISIVE CONTROL: a non-recurrent MLP[512,512] BC trained on the SAME 1.2M single-pass seq demos evaled det mean 2.2 {2:10,3:1,4:1} — IDENTICAL to the recurrent 2.0. => recurrence is NOT broken; SINGLE-PASS FSM BC is just weak (matches session history: grid single-pass 2.5, hybrid 3.3). bc_bignet=8.1 only because it used the DAgger aggregate dagger_agg_x3 (4.8M iteratively expert-corrected pairs), which the one-shot seq collection lacks.
- IMPLICATION: to test recurrence properly needs DAgger-QUALITY *sequence* data (rollout recurrent policy -> FSM-relabel -> aggregate sequences -> re-BC, x3). Big build. AND the upside is capped by the obs-limited FSM teacher (~11.4) — we're already at 10.4, near that ceiling. The bigger prize past ~11 is a BETTER TEACHER (full-info FSM ~15.1) on a richer obs.
- ARTIFACTS kept: collect_fsm_demos_seq.py, bc_recurrent.py (whole-episode BPTT, validated mechanics), demos/seq_v1_shards (1.2M). Ready for a recurrent-DAgger build if pursued.
- STATUS: bcanchor_best (det 10.4, ATH 294,625) remains the session deliverable. Recurrent single-pass shelved pending a strategy decision (recurrent-DAgger vs better-teacher vs bank).

## 2026-06-22 — STRATEGY PIVOT (user): evolve the FSM TEACHER (raise the pipeline ceiling).
- Insight (user): the teacher caps the whole BC->anchored-RL chain; "our teacher is primitive." We're at 10.4, near the obs-limited FSM's own play (11.4). Pure imitation can't exceed the teacher; RL-from-scratch walls at 3; anchored-RL is one-shot 10.4. So RAISE THE CEILING: optimize the FSM itself.
- The FSM's ~19 fight-vs-flee distance thresholds (ADJACENT, CLOSE_FIRE/MOVE_* per enemy type, count limits, BORDER_ADJUST) are HAND-SET, never tuned -> low-dim continuous black-box opt, perfect for evolution (far cheaper than RL for this search).
- evolve_fsm.py: (mu=8,lambda=24) evolution strategy, seeded from the CURRENT defaults (~11.4), sigma 0.15->0.03 annealed. Fitness = mean waves over K=6 continuous wave-1 MAME games (+score/1e6 tiebreak). 12 persistent MAME workers via task/result queues. Evolve the OBS-LIMITED FSM (nearest-N, the exact behavior BC clones) so gains stay clonable into the existing 945-dim obs. Smoke-validated (workers play, fitness returns, JSON checkpoint).
- Launched: 30 gens -> logs/evolve_v1.log + models/fsm_evolved_v1.json (best params + history, checkpointed each gen). BASELINE to beat: obs-limited FSM mean 11.4.
- PLAN if it beats 11.4: evolved FSM becomes the new teacher -> re-collect DAgger demos with it -> re-BC (bignet) -> anchored-RL from a higher base -> the whole 10.4 chain lifts. CAVEAT: obs-limited perception still caps it somewhere, but 11.4 is hand-tuned, likely well below the obs-limited optimum. (full-info FSM 15.1 is a separate higher target needing richer obs.)

## 2026-06-22/23 — CRITICAL: env was near-deterministic (evals measured ~2 games). RNG-reseed fix added.
- User caught it: validate_fsm 20 "games" were really ~2 distinct games byte-identical-repeating (wave 12 & wave 17, alternating). reset_pool=[0] loads ONE save state; boot presses start at a FIXED frame so Robotron's RNG is frozen in rl_reset -> identical games.
- ROOT CAUSE of FSM-evolution "win": OVERFIT to those 2 games. evolved_v1 best_wave 16 (det) but on GENUINELY DIVERSE games only mean 8.5 — vs hand-tuned DEFAULT 8.8 on the same diverse games. Evolution did NOT beat the default; it was a determinism artifact.
- THE FIX (user's idea: poke the RNG counter, not re-boot): Robotron RNG state = 3-byte LFSR at $9884-$9886 (robomame.asm $D6CD: "LDB $84, DP=$98 -> $9884"; earlier poke to $0084-86 was the WRONG address). Two gotchas: (1) correct address $9884-86; (2) machine:load() is ASYNC — must poke AFTER the load settles (in the loading-complete path), not in the reset dispatch (deferred load clobbers it). 
- robotron_server.lua: opt-in MAME_RL_RESEED=1 -> on wave-1 reset, LCG-step a seed (decorrelated per instance by PORT) and poke $9884-86 after settle. Verified: 10 resets -> 10 DISTINCT games (waves {4,7,8,9,12}). OFF by default -> all existing behavior/results bit-identical (gated).
- IMPLICATIONS: (a) historical metrics (FSM 11.4, bcanchor_best 10.4) were measured on the ~2-game deterministic env -> INFLATED; true diverse-game performance is lower (FSM ~8.8). (b) Going forward, eval AND train with MAME_RL_RESEED=1 for valid/robust results. (c) FSM-evolution, if revisited, MUST use reseed (diverse fitness) — but default FSM is already ~8.8 so gains look marginal.
- NEXT: re-evaluate bcanchor_best with reseed=1 for its TRUE diverse number; decide whether the deterministic-overfit also inflated the policy (likely the 10.4 is really lower on diverse games). Then re-baseline the whole pipeline on the reseeded (honest) env.

## 2026-06-23 — HONEST BASELINE (reseed): bcanchor_best = mean 8.3 (was inflated "10.4"). Pipeline sits ~8.
- bcanchor_best det 20-run WITH MAME_RL_RESEED=1: mean=8.3 median=8 min=6 max=14 dist={6:2,7:7,8:4,9:4,12:2,14:1}, score_max=310,525 (a REAL new high, beats the old deterministic 294,625). So "10.4" was ~2 waves inflated by the 2-game cycle. TRUE diverse-game performance = 8.3.
- Honest picture: obs-limited FSM teacher ~8.8, best policy 8.3 -> the policy is already AT/just-below the honest obs-limited teacher ceiling. The old "3-wave BC-fidelity gap" (8.1 BC vs 11.4 FSM) was an artifact — 11.4 was also inflated; real FSM ~8.8, so the gap is ~0.5.
- KEY INSIGHT for the next lever: ALL prior training (bc_bignet, anchored-RL) ran on the DETERMINISTIC env (~2 wave-1 games) -> the policy overfit to those, which explains the inconsistency on diverse games (median 8 but max 14 — it CAN play deep, just not reliably). Training WITH reseed (diverse wave-1 every episode) should raise the MEDIAN toward the capability ceiling by forcing robustness.
- RECOMMENDATION / NEXT: re-run the proven anchored-RL recipe from bc_bignet WITH MAME_RL_RESEED=1 (one variable: diverse training env). Honest target: lift median 8 -> 11-12 (consistency), not a new ceiling. This is the cleanest high-value experiment on the now-honest env. (FSM-threshold-evolution = marginal/dead; recurrent needs DAgger; richer-obs+full-info-teacher = the bigger separate build for a true higher ceiling.)

## 2026-06-23 — Reseed-training launched (bcanchor_reseed1): proven anchored-RL recipe on the HONEST diverse env.
- Honest baseline complete: policy bcanchor_best 8.3/med8; FSM teacher 8.95/med7.5 (max 22, score 440k — high variance).
- run models/bcanchor_reseed1: ANCHOR_BASE=bc_bignet, MAME_RL_RESEED=1 (ONLY new variable = diverse wave-1 every episode), NENV=16 PPO65536 BC300 lr1e-4 ent0.005, 6M, CK 100k, port 9942. log logs/bcanchor_reseed1.log.
- TEST: does training on diverse games lift the honest MEDIAN (8 -> 11-12) by forcing robustness? All prior runs trained on the ~2-game deterministic env (overfit -> inconsistent: median 8 but max 14). Eval every checkpoint WITH reseed (det 20-run) vs the honest 8.3 bar; promote only real improvements.

## 2026-06-23 — Reseed-training experiment CONCLUDED: parity with baseline

**Question:** does training anchored-RL on DIVERSE wave-1 games (MAME_RL_RESEED=1, the determinism fix) lift the honest median past the 8.3 baseline that was measured on the old ~2-deterministic-game eval?

**Answer: no — it converges to the same ~8.3.** Two checkpoints, honest reseed evals:
- `bcanchor_reseed1/ck_3014k` (3M): two 20-game samples 8.8/9.0 and 7.8/7.0 → avg ~8.3/8.0.
- `bcanchor_reseed1b/final` (6M total, resumed from ck_3014k after a wedged-MAME death at 3M): **40-game mean 8.4 median 8.0** (min 3, max 13, score_max 270k).
- mid-run `ck_1376k` 40-game: 8.1/8.0, but one game hit **w17 / 353k** — capability ceiling is well above the median; consistency is the wall.

vs deployed `bcanchor_best`: 8.3/8.0 (min 6, max 14). **Statistically identical.** No promotion.

**Takeaway:** the anchored-RL recipe is anchor-bound — it plateaus at the obs-limited FSM-demo teacher's level (~8.3-8.95) regardless of game diversity. Diverse games make the *number trustworthy*, not *higher*. To break past ~8.3 the **teacher/obs ceiling must be raised** (richer obs + re-DAgger, or a genuinely better teacher), not more RL on the same anchor. That is a new-architecture decision — flagged for user, not auto-launched.

**Eval noise note:** 20-game reseed evals carry ±0.5-1.0 on the mean (saw 8.8/7.8/6.8 from models all within noise of 8.3). Use 40+ games to distinguish models from here on.

## 2026-06-23 — Honest FSM ceiling measurement (reseed, 40 games): TWO gaps found

| Player | mean | median | min | max |
|---|---|---|---|---|
| Full-info FSM (all entities) | 12.9 | 12.0 | 7 | 24 |
| Obs-limited FSM (41-slot trunc = policy's obs) | 10.4 | 10.0 | 2 | 24 |
| Current policy bcanchor_best | 8.3 | 8.0 | 6 | 14 |

**Two separable gaps, in cost order:**
- **Gap A (training/cloning, ~2.1 waves): obs-limited teacher 10.4 → policy 8.3.** The policy does NOT even match its own obs-limited teacher on the SAME obs. Known cause: `dagger_agg_x3.npz` was collected 2026-06-18 PRE-RESEED (≈2 deterministic trajectories), so the BC anchor never saw diverse states → overfits, generalizes to 8.3. **Recoverable with NO architecture change: re-collect DAgger demos WITH reseed → re-BC → re-anchor.** Cheapest, highest-value next step.
- **Gap B (obs truncation, ~2.5 waves): full-info 12.9 → obs-limited 10.4.** Feeding all entities + velocity instead of nearest-41 buys ~2.5 waves at the teacher level. Needs richer obs (infra mostly exists in hybrid_obs/spatial_obs). Do AFTER Gap A.

Truncation also wrecks worst-case consistency: obs-limited min=2 (blind to threats outside 41 slots) vs full-info min=7.

**Revised plan:** (1) re-DAgger with reseed on current obs → target ~10.4. (2) then richer obs (velocity + more entities) → target ~12.9. (3) evolve FSM on the chosen obs, reseed fitness. DAgger-straight on old data was the trap; the data, not the architecture, is the first bottleneck.

## 2026-06-24 — Re-DAgger + diverse-anchor + slot-sweep: BOTH prior hypotheses overturned

Acted on the 2026-06-23 plan. Two experiments, two surprises.

**(1) Re-DAgger WITH reseed did NOT fix Gap A.** Rebuilt the seed diverse: `collect_fsm_demos.py` under `MAME_RL_RESEED=1` → 700k pairs across **467 episodes** (vs the old anchor's ~2 deterministic games; old seed preserved at `demos/fsm_mame_demos.prereseed.bak.npz`). Then `dagger.py 6 60000` reseed → `demos/dagger_agg_reseed_dagger.npz` (1.06M states). Pure DAgger/BC eval **plateaued at median 2-3** (iter0 1.9 → iter6 3.0). Then anchored-RL from `bcanchor_best` with `ANCHOR_AGG=dagger_agg_reseed_dagger.npz` (added env-var hook to `train_bc_anchored.py`, default unchanged), reseed, 3M → `bcanchor_divanchor`. Honest 40-game evals: **ck_229k 8.4/8.0, ck_688k 8.4/8.0, ck_1146k 8.1/8.0** — all parity with the 8.3 baseline. **Diverse data does not break 8.3.** The recipe is anchor-bound at the *teacher's* level, and diversity wasn't the cap.

**(2) Gap B (obs truncation) is NOT REAL — it was sampling noise.** `fsm_slotsweep_on_mame.py`: same position-only `chooseOutputs`, three slot budgets, SAME 15 reseed games:

| slot budget | mean | median | max |
|---|---|---|---|
| cur-41 (current obs) | 10.5 | 9 | 24 |
| exp-79 (2× slots) | 9.7 | 10 | 12 |
| full (ALL sprites) | 8.7 | 9 | 14 |

**Full info does NOT beat the 41-slot obs** (it's even lower — flooding nearest-threat logic with distant entities mildly hurts). The 2026-06-23 "full-info 12.9 vs obs-limited 10.4" gap was **two different 40-game draws' noise**, not truncation. **Richer obs / more slots will not help.** The 945-dim obs is sufficient to play at the FSM's ~10 ceiling.

**Corrected model of the gap:** there is ONE gap, and it is cloning/training. The obs supports ~10 (FSM on the exact 41-slot obs). The policy gets 8.3. Pure MLP BC of the FSM only reaches median **3** — the MLP fails to clone the FSM's per-frame reactive mapping; anchored-RL recovers most of it (3→8.3) but stalls ~1.7 waves short of the teacher. **Lever = a better CLONE of the FSM on the existing obs** (architecture: the slot-structured obs is a poor fit for a flatten-then-MLP; `slot_policy.SlotAttnExtractor` + `dagger.py EXTRACTOR=attn` already exist). Beyond ~10 needs a better-than-FSM teacher (search/ExIt), not richer obs. Next: attn-extractor DAgger to test whether attention closes 8.3→~10.

## 2026-06-24 — FSM evolution with reseed fitness: VALIDATED teacher gain (10.4 → 11.18)

Acting on the corrected model (cloning is the only gap; teacher ceiling is the wall), re-ran FSM threshold evolution with HONEST reseed fitness — the earlier `fsm_evolved_v1` overfit because it evolved on pre-reseed determinism. `evolve_fsm.py 20 24 8 6 12 9920 reseed_v1` under `MAME_RL_RESEED=1` (CMA-style ES, 19 thresholds, 6 reseed games/individual fitness, 12 workers).

**Convergence:** population top8_mean_wave climbed 9.38 → 12.4 over 15 gens (sigma 0.138 → 0.043, converged); per-gen best plateaued ~13.0-13.7. Winner saved `models/fsm_evolved_reseed_v1.json` (best_params, 19 dims; 6-game best_wave 13.67).

**Honest 40-game reseed validation (`validate_fsm.py models/fsm_evolved_reseed_v1.json 40`):**

| player | mean | median | min | max |
|---|---|---|---|---|
| **evolved FSM (reseed_v1)** | **11.18** | **11.5** | 3 | 21 |
| baseline obs-limited FSM | ~10.4 | ~10 | 2 | 24 |
| best trained policy (bcanchor_best) | 8.3 | 8.0 | 6 | 14 |

The 6-game fitness (13.67) was ~2.5 waves optimistic vs the honest 40-game (11.18) — KGAMES=6 lets the ES chase lucky seeds — but validation confirms a **real +1 to +1.5 wave teacher gain**. The evolved FSM is now the **best player on MAME, ~3 waves above the trained policy.**

**Takeaway:** evolution with reseed fitness genuinely raises the teacher ceiling. The overfit gap says KGAMES should be larger (≥10) for a cleaner honest result. Next options: (a) re-evolve with KGAMES≥10 (warm-start from reseed_v1) to push the honest ceiling higher with less seed-luck; (b) propagate the better teacher into a deployable policy (re-collect demos with evolved FSM → DAgger → anchored-RL), expecting ~+1 wave over the current 8.3; (c) a fundamentally stronger teacher (lookahead search / ExIt) for a bigger ceiling jump. The evolved FSM itself is deployable now and is the current SOTA.

## 2026-06-24 — FSM evolution v2 (warm-start + KGAMES=10): new SOTA teacher 12.72

Refined the evolution to cut winner's-curse: warm-started the ES mean from `fsm_evolved_reseed_v1` (added `EVOLVE_WARM` env hook) and raised fitness games 6→10; also now saves the converged `mean_params` (robust to argmax bias) alongside `best_params`. `evolve_fsm.py 15 24 8 10 12 9920 reseed_v2`, `MAME_RL_RESEED=1`. Converged top8_mean 10.7→12.9, all-time best_wave 14.40 (10-game).

**40-game reseed validation (both candidates):**

| teacher | mean | median | min | max |
|---|---|---|---|---|
| **v2 best_params (NEW SOTA)** | **12.72** | **12.0** | 7 | 23 |
| v2 mean_params | 12.55 | 12.5 | 1 | 22 |
| v1 best (reseed_v1) | 11.18 | 11.5 | 3 | 21 |
| baseline FSM | ~10.4 | ~10 | 2 | 24 |
| trained policy bcanchor_best | 8.3 | 8.0 | 6 | 14 |

`best_params` wins on mean AND worst-case (min 7 — never collapses; the mean has one wave-1 game). Optimism gap shrank to 14.40→12.72 (vs v1's 13.67→11.18), confirming KGAMES=10 + warm-start. **New best teacher = `models/fsm_evolved_reseed_v2.json` best_params, 12.72/12.0.** Evolution gains accelerating (10.4→11.18→12.72), so a v3 (warm from v2, KGAMES≥12) is launched to find the threshold-tuning plateau. Beyond that, deeper waves likely need a lookahead-search/ExIt teacher, not threshold tuning.

## 2026-06-24 — Search/ExIt teacher: shelved (short-horizon greedy plays wave 3-4)

Wrote `mame_gym/search_validate.py` — policy-improvement search on the evolved v2 FSM base (keep FSM fire, search 8 moves via H-frame save/restore rollout continuing with the evolved FSM, score = SURVIVE_W·survived + score_gained − death_penalty). Smoke test (N=3, H=16, v2 base, reseed): **wave 3-4** — far BELOW the 12.72 FSM it's built on.

**Diagnosis (design flaw, not bug):** at H=16 (×frameskip 4 ≈ 1 s) almost no move dies within the horizon, so all 8 candidates score ≈ equal survival and the argmax collapses to **greedy score-maximization** → chase enemies into contact → die early. (Collection in `parallel_search_collect.py` reaches deep waves only because env.step there is driven by the POLICY; the search output is merely the recorded label.) The save/restore pattern itself is fine (env.step terminal check at env line 327 is consistent across the restore). Fix would need a much longer horizon (H≥60, ~4× slower, ~20 min/game) or a survival-weighted value that differentiates positioning — a research effort, not a quick win. **Shelved.** Evolved FSM v2 (12.72) remains SOTA teacher.

**Loop pivot:** propagate the v2 teacher into a deployable policy (collect demos with evolved v2 FSM under reseed → BC → anchored-RL). Better anchor than the ~10 baseline that produced 8.3; target a policy above 8.3.

## 2026-06-25/26 — Teacher propagation VALIDATED: v2-teacher anchor lifts policy 8.3 -> ~9

Propagated the evolved v2 FSM (12.72) into a policy: collected 700k reseed demos LABELED by the v2 FSM (`collect_fsm_demos.py` + new `EVOLVE_PARAMS` env hook; baseline reseed seed preserved at `demos/fsm_mame_demos.baseline_reseed.npz`), then ran anchored-RL from `bcanchor_best` with `ANCHOR_AGG=fsm_mame_demos.npz` (the v2 seed), reseed, 3M -> `bcanchor_v2anchor`. Skipped DAgger (earlier: adds ~nothing, median 2->3; and its relabeler uses default params which would dilute v2).

**Honest 40-game reseed evals (within-run progression = real, not noise):**

| checkpoint | mean | median | min | max |
|---|---|---|---|---|
| ck_229k (early) | 8.2 | 8.0 | 2 | 14 |
| ck_2752k (late) | 8.7 | **9.0** | 3 | 15 |
| ck_2981k (final) | **9.6** | 8.0 | 4 | 17 |
| baseline bcanchor_best | 8.3 | 8.0 | 6 | 14 |

The policy climbs 8.2 -> 9+ as anchoring to the v2 teacher strengthens — the FIRST break past the long-standing ~8.3 RL plateau. Modest (~+0.5-1 wave) because cloning loses ~3-4 waves off the 12.72 teacher, but it confirms the core thesis: **a better teacher propagates to a better policy.** Promoted ck_2752k -> `models/bcanchor_v2anchor_best/` (8.7/9.0); old `bcanchor_best` kept as fallback.

**Implication / next:** the whole chain is teacher-bound. To push the policy further, push the teacher past 12.72. Threshold-evolution has plateaued (~12.7); the remaining ceiling-raiser is a lookahead/search teacher (needs the long-horizon fix from the shelved search experiment) or a richer FSM structure.

## 2026-06-27 — v2anchor extension to 6M: REGRESSED (overfit); + wave-cycle insight (ASM-confirmed)

Extended bcanchor_v2anchor another 3M (warm from the 9.6 model, same v2 anchor) -> bcanchor_v2anchor2. bc_nll fell 0.37->0.18 (over-fit to v2 demo states). Final ck_2981k 40-game: **mean 8.0 median 8.5** — REGRESSED vs the 3M v2anchor (ck_2981k 9.6, ck_2752k 8.7/9.0). More anchoring overfits and hurts own-trajectory play. **3M is the sweet spot; `bcanchor_v2anchor_best` (8.7/9.0) remains best policy.**

**WAVE-CYCLE INSIGHT (ASM-confirmed, robomame.asm $2B7C 2BA9-2BB6):** wave config tables are indexed by wave reduced to 1..40 via `while(B>40) B-=20`. After wave 40 the game CYCLES with period 20 through configs 21-40 (waves 41-60 ≡ 21-40, 61-80 ≡ 21-40, ...). Wave counter ($BDED) keeps incrementing 1..255 (no reset at 40), so eval deep-wave counts are correct. **Implication:** the 1->100 goal collapses to mastering ~the 21-40 cycle; the relative-position obs makes 41-100 generalize for free. Bottleneck = REACHING waves 21-40 (policy ~9, teacher ~13 both short). Next promising direction: deep-wave curriculum — capture save states from the evolved FSM (reaches w16-23) across waves 10-23 and RL with resets into that band to teach the cycle patterns directly.

## 2026-06-27 — IN PROGRESS: deep-wave curriculum (curric1)

Informed by the ASM wave-cycle (configs cycle period-20 after wave 40 → master ~21-40 generalizes to 100). Bottleneck = policy dies ~9, never practices deep waves. Plan: capture deep states → reset training INTO them.
- STEP 1 done: `mame_gym/capture_ladder_fsm.py` (new, FSM-driven) captured 62 save-states waves 10-17 (idx 30001-30062). NOTE: killed mid-run so `ladder_fsm_state_map.json` not written — idx→wave map is in `logs/capture_ladder_fsm.txt`. Verified states load on a fresh port.
- STEP 2 done: `mame_gym/curriculum_pool_fsm.txt` = 40×idx0 (wave-1 reseed boots) + the 62 deep idxs.
- STEP 3 RUNNING: `train_bc_anchored.py … curric1` (PID was 597297, logs/anchored_curric1.txt) — warm from `bcanchor_v2anchor_best` (8.7/9.0), `ANCHOR_AGG=fsm_mame_demos.npz` (v2 seed), `CURRICULUM_POOL` env (new hook), reseed, 3M. ck_229k(early)=8.1/8.0 (inconclusive). Eval LATE checkpoints (~1.5-3M, 40-game reseed) — promote if >9.0. Hypothesis: deep practice → reaches deeper in continuous play.
- If session died: training is detached (survives); resume by evaling models/bcanchor_curric1/checkpoints/ late ckpts. Best policy so far = `models/bcanchor_v2anchor_best/` (8.7/9.0); SOTA teacher = `models/fsm_evolved_reseed_v2.json` (12.72).

## 2026-06-27 — Permanent eval protocol built (mame_gym/eval_protocol.py) + life-economy reframe

Acting on the SUGGESTIONS.md review (eval samples too small; promotions kept reversing). Built `mame_gym/eval_protocol.py`:
- Reserved HELD-OUT port (9970): reseed RNG is a deterministic LCG keyed by PORT (rng_seed=PORT*2749+1, advanced once/reset), so a fresh server replays the SAME seed sequence every run => two models on the same port are PAIRED by construction; seeds disjoint from training ports (9940-47) and evolution ports (9920-31).
- N>=100 continuous wave-1 games; survival curve P(reach w>=k); bootstrap 95% CIs on mean+median; PAIRED mode (--vs) reports per-seed wave delta + win/tie/loss + CI verdict.
- LIFE ECONOMY (flicker-free, score-based): extra lives = score//25000 (CMOS extra_man_every default, robomame.asm $CC00); reports life-gen/wave vs deaths/wave.

**Life-economy reframe (confirmed):** smoke on bcanchor_v2anchor_best gave deaths/wave 0.78 > life-gen/wave 0.59 — the policy BLEEDS lives. Wave depth alone is the wrong target; sustainability gate = life-gen/wave >= deaths/wave. Future promotions: >=100 paired held-out games, gated on continuous wave-1 survival + paired CI, NOT deep-reset performance.

Next: when curric1 finishes (~3M), first promotion-grade test = eval_protocol.py curric1_final --vs bcanchor_v2anchor_best at N=100.

## 2026-06-27 — curric1 (deep-wave curriculum): NO PROMOTION (paired N=100 inconclusive) + life-economy is the wall

First promotion-grade decision under eval_protocol.py. curric1_final --vs bcanchor_v2anchor_best, 100 paired held-out games (port 9970):
- A curric1: mean 8.94 [8.39-9.49] median 9.0 P(w>=10)=0.34 P(w>=15)=0.03 max16 life-gen/wave 0.57 deaths/wave 0.80
- B bcanchor_v2anchor_best: mean 9.41 [8.82-10.01] median 9.0 P(w>=10)=0.44 P(w>=15)=0.06 max20 life-gen/wave 0.59 deaths/wave 0.80
- PAIRED B-A: +0.47 [-0.37..+1.33], A wins 46 / ties 12 / B wins 42 -> INCONCLUSIVE (CI spans 0).

Deep-wave curriculum did NOT transfer (P(w>=15) only 0.03). NO PROMOTION. bcanchor_v2anchor_best (true N=100 = 9.41) remains best policy. Confirms SUGGESTIONS #3.

BIG PICTURE: life economy is the wall (SUGGESTIONS #6). Every controller BLEEDS lives: both policies life-gen/wave ~0.59 < deaths/wave ~0.80; FSM smoke 0.71 < 0.86. Wave 100 needs fewer deaths/wave and/or more score/wave, not just depth. Next: FSM-residual controller targeted at the FSM's death situations.

## 2026-06-27 — FSM death forensics (50 games, 536 deaths): QUARK is the #1 killer (32%)

Evolved-FSM-v2 with MAME_DEATH_LOG_DIR; death_audit + JSONL. 536 deaths, 96% explained-direct, 0 unexplained. Killer composition: Quark 172 (32%), Hulk 98 (18%), EnfBullet 94 (18%), Grunt 89 (17%), Brain 41 (8%). Only 1% mutual-destruction. Per-wave spikes: wave 7 = 124 deaths (23%, of which 108 QUARKS); wave 9 = 66; wave 12 = 61.

Actionable: FSM has a Quark blind spot. Fixing Quark avoidance could cut ~32% of deaths -> raise FSM ceiling >12.72 AND propagate to policy. Cheaper than a learned residual. Pivoting residual thread to FIRST try a targeted FSM Quark-handling fix (SUGGESTIONS #4), validated via eval_protocol paired vs current FSM.

## 2026-06-27 — Quark-flee threshold fix: WASH (consistency up, deep tail down) -> motivates learned residual

CLOSE_MOVE_QUARK=100 (flee Quarks 2x) vs =50 (baseline), paired N=60 via eval_protocol:
- A quark100: mean 12.10 [11.1-13.2] min7 max22 P(w>=8)=0.88 P(w>=15)=0.18 P(w>=20)=0.08
- B quark50:  mean 11.72 [10.7-12.9] min2 max25 P(w>=8)=0.78 P(w>=15)=0.25 P(w>=20)=0.02
- PAIRED B-A: -0.38 [-1.77..+1.02], A wins 30 / ties 5 / B wins 25 -> INCONCLUSIVE.

Net wave tied. Fleeing Quarks earlier CUTS catastrophic early deaths (min 2->7, P(w>=8) up) but SACRIFICES deep tail (Quarks unkilled -> more Tanks). Context trade-off a single threshold can't resolve -> the Quark decision is context-dependent (engage when safe, flee when cornered) -> learned RESIDUAL is the right tool. Safe FSM code change (CLOSE_MOVE_QUARK, default None) stays in. Next: FSM-residual controller overriding FSM movement in Quark/Hulk death-risk states.

## 2026-06-27 — IN PROGRESS: FSM-residual controller (residual1)

Built residual_env.py (ResidualWrapper: FSM proposes move+fire; policy action MultiDiscrete([2,8])=(gate,override_move); gate=1 uses override, fire always FSM; reward -= 0.2 per override) + train_residual.py (PPO [512,512] fresh). Smoke-verified.
RUNNING: train_residual.py 3000000 9940 residual1 (PID 656312, EVOLVE_PARAMS=v2, reseed, NENV=8). Goal: cut deaths/wave (Quark=32%) keeping FSM 12.72 -> beat pure FSM on eval_protocol paired.
TODO: add residual-mode to eval_protocol (wrap env in ResidualWrapper) to judge checkpoints vs pure FSM. If session dies: training detached. Best policy bcanchor_v2anchor_best (9.41); SOTA teacher fsm_evolved_reseed_v2 (12.72).

## 2026-06-27 — Residual1 (penalty 0.2): FAILED (7.83 < FSM 11.72). Penalty too weak.

Residual final N=60 held-out port: mean 7.83 [7.12-8.48] median 8 P(w>=5)=0.77 P(w>=10)=0.18. vs same-seed FSM 11.72 and best policy 9.41. UNDERPERFORMS both — never recovered FSM-level (trend 5.3->7.6->8.6->7.8 stalled). Root cause: OVERRIDE_PENALTY=0.2 negligible vs survival reward (~0.3*wave ~2.4/step @w8), so PPO overrides freely and never defaults to keep-FSM. Fix: penalty=1.0. If residual2 fails too -> pivot to supervised death-risk-model gating (536 FSM death records) or deploy FSM (best player 11.72/12.72).

## 2026-06-28 — Residual2 (penalty 2.0) early: still below FSM (ck_200k=7.25). From-scratch residual struggling.

residual2 ck_200k N=12: mean 7.25 min2 — faster convergence than residual1 but still far below FSM 12.72, still early catastrophes. Both residual attempts trend below-FSM. DIAGNOSIS: from-scratch PPO can't cleanly learn a gate (keep FSM ~99%, override only rare death-risk states); training exploration keeps it below the teacher. PRINCIPLED FIXES not yet tried: (1) gate PRETRAINED to never-override (BC the FSM through the residual action space, then PPO learns only useful overrides — Codex's exact recipe). (2) supervised death-risk model gating. (3) deploy FSM as-is (best player 11.72-12.72). Letting residual2 finish but expectation = below-FSM.

## 2026-06-28 — Residual3 (gate-pretrain): FIXED the start, ck_200k=11.42 (FSM-level)
PRETRAIN_GATE BC to [gate=0,fsm_move] on 300k demos BEFORE PPO (penalty 0.5). ck_200k N=12: mean 11.42 median 11.5 min7 P(w>=15)=0.25 vs residual1 5.3 / residual2 7.3 (crashed). Pretrain = missing piece; starts FSM-level. KEY TEST: beat FSM 12.72 by 3M? N=60 paired. PID 693058.

## 2026-06-28 — Residual3 final 9.35 (below FSM). CONCLUSION: no RL method beats the FSM.
residual3 final N=60: mean 9.35 [8.60-10.13] min1 max17. Gate-pretrain gave FSM-level start (11.42@200k) but PPO degraded to 9.35 by 3M.
SESSION CONCLUSION: every RL approach plateaus ~9-9.4, NONE beats evolved FSM (11.72-12.72): anchored 8.3, diverse-anchor 8.4, v2-anchor 9.41, curriculum 8.9, extension 8.0, residual x3 5.3/7.3/9.35. DEPLOYABLE BEST = evolved FSM v2 (fsm_evolved_reseed_v2.json). To exceed: better TEACHER (value-guided search) or structural FSM (Quark=32% deaths); wall=life economy (life-gen/wave 0.6 < deaths 0.8). Lasting wins: eval_protocol.py, death forensics, ASM wave-cycle.

## 2026-06-28 - Search teacher re-validated BROKEN (wave 3-4 << FSM). Rollout assumes FSM-continuation but actual play is search-continuation; mismatch compounds. Fix = value-guided (learned death-risk model). No autonomous path exceeds FSM now. Deploy evolved FSM 12.72.

## 2026-06-28 - Value-model step1: data collected, horizon too short (3pct near-death). FIX: re-collect uncapped ttl or H~600 before training V (task 14).

## 2026-06-28 - VALUE-GUIDED SEARCH **BLOCKED at root**: MAME save-state is LOSSY (search dead)
Completed the value-guided-search experiment end-to-end. **The value MODEL is excellent;
the search PRIMITIVE is fatally broken.**
- Data: `survival_v2b.npz` (H=600, 20.4% in danger zone — the v2 H=60 had only 2.8%) was
  already collected. Trained `train_value.py v2b` -> `models/value_v2b/` (ValueNet 945->512->
  512->1 sigmoid = ttl/600). **val corr 0.956, val_mse 0.0059** — a strong death-risk model.
- Built `mame_gym/search_value.py` (value-guided teacher, 3 modes via env vars): PURE
  argmax-V, guarded safety-override (default; keep FSM move unless its endpoint V<DANGER and
  an alt is safer by MARGIN), and FSMONLY paired baseline.
- Results on 6 deterministic reseed games (same seeds, paired):
  | mode | mean wave | note |
  |---|---|---|
  | FSMONLY (no search) | **9.83** | clean FSM thru this harness (g1=13,g3=14,g6=13) |
  | PURE argmax-V | 3.50 | over-cautious fleer; never clears waves |
  | guarded DANGER=0.35 | 6.75 | overrides net-harmful |
  | guarded DANGER=0.10 | 6.33 | g1 13->4, g3 14->7, g6 13->7 |
  | DANGER=-1 (override NEVER fires) | **6.33** | **same collapse with overrides OFF** |
- The DANGER=-1 control is decisive: with overrides disabled but the save/restore search
  still running every step, play collapses identically (13->4). So it's the **per-step
  save/restore machinery**, not the value model or override logic.
- ROOT CAUSE (fidelity test, scratchpad/fidelity.py): MAME save-state is **LOSSY for the
  Robotron driver**. save_state -> replay K identical actions, vs load -> replay SAME K
  actions DIVERGES by step 4 (score 400 vs 300) and ends totally different (lives2/900 vs
  lives1/500). `machine:load()` does not bit-exactly restore the machine, so every
  candidate-rollout ping-pong corrupts the live game. Episode-start resets tolerate this;
  search rollouts cannot.
- **CONCLUSION: the value-guided SEARCH teacher is dead on this MAME setup** — same root
  failure as raw search, now explained. Deployable best remains the evolved FSM (12.72).
  The value model (corr 0.956) is preserved and still usable WITHOUT save/restore.
- REMAINING value-model lever (next): score the 8 candidate moves with an **analytic
  1-step forward model** (player kinematics are known/deterministic; enemies extrapolated
  via the velocity channel already in obs) -> build the hypothetical next-obs in Python and
  evaluate V — NO emulator rollout, so no save/restore corruption. Distribution-shift risk
  (V trained on real obs) but worth one clean test before declaring the FSM final.

## 2026-06-28 - ANALYTIC forward-model V-override: CLEAN (no corruption) but ~net-neutral
Implemented the analytic lever in `search_value.py` (mode VSEARCH_ANALYTIC=1): measured
player kinematics (±9.5px/step horiz, ±9.2 vert per dir, deterministic; scratchpad/kin.py),
move the player synthetically + extrapolate enemies by their obs velocity, rebuild the
945-dim obs via `_extract_features` (stateless when velocity is threaded) and evaluate V on
the SYNTHETIC next-obs. NO emulator save/restore -> immune to the lossy save-state.
- Confirmed clean: analytic mean 9.50 vs FSM-only 9.83 on 6 paired seeds (NOT collapsed to
  6 like the save/restore version -> the corruption is gone, harness is sound).
- Paired per-seed (FSM | analytic-0.35 | analytic-0.18): g1 13|8|9, g2 7|7|12, g3 14|7|7,
  g4 4|10|4, g5 8|13|8, g6 13|12|13. The value model DOES detect danger (real saves: g4
  4->10, g2 7->12) but the corrective argmax-V move is NOT reliably better than the
  finely-tuned FSM -> it TRADES weak-game saves for strong-game damage (g1 13->8, g3
  14->7). Net wash: 0.35 mean 9.50, 0.18 mean 8.83, FSM 9.83 (6 noisy seeds).
- WHY override hurts strong games: synthetic obs is approximate (linear enemy extrapolation,
  no AI reaction) + V distribution-shift, so when it overrides good FSM play the chosen dir
  is ~coin-flip. Better criterion (untested): override toward the FSM-nearest SAFE dir
  (minimal deviation), not global argmax-V.
- LAUNCHED N=20 paired eval (bg): fsm_only_n20.log (port 9972) vs analytic_n20.log
  (DANGER=0.35, port 9973) for a decisive less-noisy read. Pending.

## 2026-06-28 - ANALYTIC override N=20: REACHES PARITY with the FSM (first lever to do so)
N=20 paired (deterministic seeds, same harness):
- FSM-only:  mean wave 11.30, median 12, mean score 198,974, deep games(>=15)=4
- Analytic:  mean wave **12.25**, median 12, mean score **221,061**, deep games(>=15)=**7**
- Paired: analytic 9 wins / 8 losses / 3 ties.
Earlier 6- and 9-game reads looked net-negative; the full 20 flipped to a hair AHEAD on
mean+score+deep-tail. wins≈losses => statistically a WASH at N=20 (project_eval_protocol:
"20 too noisy, use 40+"). BUT this is the FIRST autonomous lever to even MATCH the FSM
(every RL variant + raw/value save-restore search fell clearly below). The analytic
V-override neither corrupts the game nor degrades play, and may carry a small deep-tail
edge (the value model steers away from the rare catastrophic move). Worth a real N>=40
paired eval before any verdict. The known weakness (override hurts some strong games via
argmax-V coin-flip) suggests a "minimal-deviation safe dir" criterion could turn the wash
into a clear win. LAUNCHED N=40 paired (fsm_only_n40.log / analytic_n40.log).

## 2026-06-28 - MINIMAL-DEVIATION V-OVERRIDE (DEVW): looks like it BEATS the FSM (first ever)
Added VSEARCH_DEVW: when the FSM move is dangerous (analytic V<DANGER), pick the safe dir
NEAREST the FSM heading via score = V(d) - DEVW*compass_steps(d,fsm_dir), instead of erratic
global argmax-V. dirs 1..8 are circular (45deg/step). Fixes the "override wrecks strong
games" failure (the argmax-V coin-flip).
- 6-game smoke (DEVW=0.10), paired vs the known FSM / plain-analytic:
  | game | FSM | analytic-0.35 | DEVW |
  |---|---|---|---|
  | g1 | 13 | 8  | **17** |
  | g2 | 7  | 7  | 7 |
  | g3 | 14 | 7  | 12 |
  | g4 | 4  | 10 | **12** |
  | g5 | 8  | 13 | 8 |
  | g6 | 13 | 12 | **14** |
  | mean | 9.83 | 9.50 | **11.67** |
  DEVW vs FSM: 3 wins / 1 loss / 2 ties, big wins (g1 +4, g4 +8). It keeps the saves AND
  stops damaging strong games -> beats the FSM by +1.84 on these 6 seeds. FIRST autonomous
  method to exceed the FSM (everything prior fell below). 6 games is noisy (need 40+).
- LAUNCHED N=40 DEVW eval (analytic_devw_n40.log, port 9976) vs fsm_only_n40.log baseline.
  If DEVW > FSM holds at N>=40, this is a genuine teacher improvement past 12.72 -> validate
  at N=100 (eval_protocol) then it becomes the deployable best player.

## 2026-06-28 - RESULT: analytic V-override BEATS the FSM at N=40 (first autonomous method to)
Final N=40, all on the SAME 40 deterministic reseed seeds + same harness:
| teacher | mean wave | mean score | paired vs FSM | decisive win-rate |
|---|---|---|---|---|
| FSM-only (evolved v2) | 11.30 | 194,549 | - | - |
| **plain analytic argmax-V** | **12.35 (+1.05)** | **225,393 (+16%)** | W23/L15/T2 | **61%** |
| DEVW min-deviation | 11.88 (+0.58) | 217,529 (+12%) | W21/L14/T5 | 60% |
- BOTH analytic-override variants beat the FSM; **plain argmax-V (DANGER=0.35, MARGIN=0.10,
  NO deviation penalty) is best.** The DEVW minimal-deviation refinement HURT slightly — my
  "deviation penalty cuts strong-game losses" hypothesis was WRONG; plain greedy-V over the
  synthetic 1-step obs is better. (DEVW=0 == plain analytic.)
- The plain-analytic edge is CONSISTENT: N=20 12.25, N=40 12.35 (vs FSM 11.30) — ~+1 wave,
  +16% score, 61% paired win-rate. Not strongly significant on win-rate alone (23/38, p~0.13)
  but the mean+score+win-rate all agree across two sample sizes => a real modest improvement.
- This is the FIRST autonomous method to exceed the FSM (everything prior — all RL, raw
  search, save/restore value search — fell below). Mechanism: a learned death-risk value
  model V (corr 0.956) + an ANALYTIC 1-step forward model (no lossy emulator save/restore)
  greedily steers the player away from the rare catastrophic move while keeping the FSM's
  fire + default move. Cheap (FSM + one 8-row NN forward per step), deployable as-is.
- LAUNCHED N=100 paired confirmation per project_eval_protocol (fsm_only_n100.log /
  analytic_n100.log, plain argmax-V). If +1 wave holds, the analytic-V-override teacher is
  the new deployable best player; update project_rl_plateau_fsm_best + the FSM-is-best memory.

## 2026-06-28 - N=100 VERDICT: analytic V-override == FSM (PARITY, not a beat). +1 was seed-luck
Full N=100 paired (same 100 reseed seeds), plain argmax-V (DANGER=0.35, MARGIN=0.10):
| teacher | mean wave | median | max wave | mean score | max score |
|---|---|---|---|---|---|
| FSM-only (evolved v2) | 12.19 | 12 | 22 | 221,034 | 435,200 |
| analytic V-override   | 12.32 | 12 | **24** | 222,169 | **550,850** |
Paired: analytic 44W / 42L / 14T = **51% win-rate** -> a statistical DEAD HEAT on central
tendency (+0.13 wave, +0.5% score). The +1.05 wave from N=40 was SEED-LUCK: seeds 0-39
happened to favor the override; adding seeds 40-99 (where the FSM plays its deep games)
diluted it to parity. Watching the running estimate live showed the textbook regression:
+1.9 @ n=32 -> +0.7 @ n=50 -> +0.04 @ n=58 -> +0.13 @ n=100. Lesson reaffirmed
(project_eval_protocol): trust N>=100, never N<=40 on this high-variance task.
- HONEST CONCLUSION: the analytic V-override REACHES parity with the FSM (first method to
  even match cleanly — no corruption, cheap) and extends the deep tail slightly (max wave
  24 vs 22, max score 550k vs 435k), but does NOT decisively beat it. The DEPLOYABLE BEST
  remains the evolved FSM (simpler, identical mean). The whole-project conclusion stands:
  no autonomous method decisively exceeds the FSM.
- Assets preserved & reusable: models/value_v2b (death-risk V, corr 0.956), search_value.py
  (analytic forward-model override, VSEARCH_ANALYTIC), measured player kinematics. The
  value model could still feed a future approach (e.g. shaping or a slightly-deeper analytic
  lookahead) — its slight deep-tail extension hints the lever isn't fully exhausted, but the
  current 1-step override is a wash. Save/restore search remains DEAD (lossy save-state).

## 2026-06-28 - STRUCTURAL FSM FIX #1: Quark kiting (user-directed: improve the FSM)
User chose focused FSM work over more RL (RL is exhaustively proven to plateau < FSM).
Target = the #1 death cause. Forensics (deaths_fsm_v2, 536 deaths) on WHY Quarks kill:
- Quark = nearest-suspect in 172/536 = 32% (Hulk 98, Spark 94, Grunt 89 next).
- Concentrated at WAVE 7 (108/172), then 12/17/22 (the period-5 quark waves; wave 7 = first
  and deadliest). Median quark dist at death = 7.5px (direct contact); mutual-death frac 0.01
  (player moved INTO the quark, quark survives); NOT cornered (edge-dist median 72); 170/172
  had >=2 quarks present; 0 tanks -> direct quark contact, not the spawned tanks.
- ROOT CAUSE: Quarks are in CHASE_ENEMIES so the FSM moves TOWARD them to ~50px to shoot,
  but quarks TELEPORT (>40px -> contact in one frameskip-4 step) so the 40px ADJACENT flee
  radius reacts too late; AND on quark waves an Enforcer/bullet wins the adjacent-priority
  contest, so the player flees the bullet INTO the quark. (The 2026-06-27 CLOSE_MOVE_QUARK
  standoff-threshold tweak was a WASH because the chase path never actively flees.)
- STRUCTURAL FIX (robotron_fsm.py): new QUARK_TYPE threat — a close quark becomes a
  top-priority flee target (just below PROJECTILE, above PriorityEnemy) with a LARGE reaction
  radius ADJACENT_QUARK=75 (vs 40), and its adjacent handler ALWAYS flees (no STAY) while
  firing at it = KITE. Reverts to old chase if ADJACENT_QUARK<=0 (clean A/B via FSM_ADJACENT_QUARK
  env). Smoke-verified: quark to the right -> move LEFT (away) + fire RIGHT (at it).
- VALIDATION launched (paired, same 40 reseed seeds, FSMONLY): fsm_quarkfix_off_n40.log vs
  fsm_quarkfix_on_n40.log. Success = higher mean wave + fewer wave-7 quark deaths. If it
  holds at N>=40, sweep ADJACENT_QUARK (60/75/90) + N=100 confirm; this is a clonable teacher
  gain (propagates via BC->anchored-RL like the evolved-FSM threshold gains did).
- N=40 PAIRED RESULT (same 40 seeds, FSMONLY): off mean 12.28 / median 12 / score 221,617;
  on mean **13.45** / median **14** / score **254,286 (+15%)**; paired **on W24 L11 T5 = 69%
  win-rate**. Far more lopsided than the value-override wash; median +2 waves. Caveat: on had
  one min=1 (rare over-flee early death) vs off min=7 — worth a look. Per the value-override
  lesson (N=40 can flatter), launched N=100 confirm (fsm_quarkfix_off_n100.log / _on_n100.log,
  ports 9982/9983) before declaring.

## 2026-06-29 - METHODOLOGY FIX: evals were UNPAIRED (seed = f(port)) -> win-rates were invalid
While checking why the Quark N=100 partial disagreed with N=40, found evals are NOT
deterministic across runs: the lua reseed inits `rng_seed = PORT*2749+1`, so each PORT gets
its OWN game sequence. off and on always ran on DIFFERENT ports -> DIFFERENT games -> my
per-index "paired W/L/win-rate" numbers all session (value-override AND quark) were comparing
mismatched seeds = MEANINGLESS. Only the SAMPLE MEANS were valid comparisons (unpaired, full
variance) — which is why every result swung with N. (Conclusions on MEANS still stand:
value-override == FSM parity 12.32 vs 12.19; quark N=40 means off 12.28 vs on 13.45 were a
noisy unpaired read.)
- FIX (robotron_server.lua): `MAME_RL_SEED_BASE` env overrides the port-derived seed, so
  separate runs on different ports face the IDENTICAL game sequence = TRUE PAIRING. Verified:
  same config on ports 9986 & 9987 with SEED_BASE=777 -> byte-identical (13/16/12/9, same
  scores/steps). Massive variance reduction; use for ALL future A/Bs.
- Relaunched the Quark A/B as a TRUE PAIRED N=100 (SEED_BASE=777, off port 9988 / on 9989:
  fsm_qf_off_paired.log / fsm_qf_on_paired.log). This is the decisive, low-variance test of
  the Quark kiting fix. Killed the old unpaired N=100.
- PAIRED VERDICT (ADJACENT_QUARK=75, ~87/100, converged): off 12.34 vs on **12.68, delta
  +0.33 wave**, on W41/L32/T14 = **56% paired win-rate**. So the kiting fix is a SMALL but
  ROBUST positive (NOT the +1.17 the unpaired N=40 flattered; that was seed-mismatch noise).
  Mechanism read: it survives Quark waves more often (wins more games) but always-flee (no
  STAY) slightly caps the deepest runs (off's big games > on's), so the mean gain is small.
  -> Sweeping ADJACENT_QUARK 60/90 (paired vs same off, seed 777, fsm_qf_on60/on90_paired.log)
  to find a less-aggressive radius that keeps the survival edge without capping deep games.
- FINAL on=75 PAIRED N=100: off 12.35 vs on **13.07, delta +0.72 wave, 58% win-rate
  (W50/L36/T14)** — a solid real teacher improvement (the +0.33 mid-run rose as the last
  games landed). SWEEP gradient (paired vs same off): on=90 -0.36 (too aggressive, over-flees),
  on=75 +0.72, on=60 +1.00 (early, n=10). Monotonic: LESS-aggressive flee = better. Extended
  to on=50 (=CLOSE_MOVE_CHASE_ENEMY radius) to bracket the optimum. Best radius -> new FSM
  default; then re-run death forensics to confirm Quark deaths actually dropped, and propagate
  the improved teacher (BC->anchored-RL) like the evolved-FSM threshold gains.
- SWEEP RESOLVED: ADJACENT_QUARK=75 is the PEAK. Paired-vs-off N=100 (75) = +0.72; the
  neighbors regressed to negative as N grew (on50 -1.23, on60 -0.63@38, on90 -1.18) — all
  worse, none beats 75. **ADJACENT_QUARK=75 adopted as the new FSM default** (already the
  module default). VALIDATED Quark structural fix: +0.72 wave, 58% paired win-rate, N=100.
  Killed the sweep arms. Launched forensics confirmation (fix ON, 50 games seed-777, death
  log -> deaths_fsm_quarkfix/) to verify Quark's 32% death share actually dropped.
- FORENSICS CONFIRM (fix on, 320 deaths @25 games): Quark 32%->**26%** (real drop), but Hulk
  18%->**27% = now #1 killer**. Hulk deaths: 6px contact, NOT cornered (edge 54), spread over
  deep waves (peak w12). KEY INSIGHT: the aggressive Quark-flee partly TRADES quark deaths for
  hulk deaths (flee quark -> back INTO an invincible hulk). Net still +0.72, but the real next
  lever is a THREAT-FIELD-AWARE flee: move toward genuinely open space (repulsion summed over
  ALL threats, hulks weighted heavily since unshootable) instead of blindly away from one
  adjacent threat. This is the long-identified "global threat-field" wall — tractable in the
  FSM (full ground-truth positions) where the RL obs/CNN couldn't crack it. Should cut BOTH
  quark and hulk deaths. NEXT: implement as opt-in (env flag) in getMoveStick, validate paired
  (MAME_RL_SEED_BASE) vs current. The paired-eval infra now makes this measurable.
- FORENSICS FINAL (546 deaths): Quark 27%, Hulk 26%, Spark 16%, Grunt 10%; **46% of Hulk
  deaths have a Quark also among suspects** = strong confirmation of the flee-quark-into-hulk
  trade.
- IMPLEMENTED threat-field flee (robotron_fsm.py, opt-in FSM_THREAT_FIELD=1): new
  `threatFieldMove()` sums each nearby threat's toward-vector * weight / dist (HULK 2.2, Quark
  1.6, projectiles 1.5, others 1.0, electrode 0.5; within FSM_FIELD_RADIUS=140) and feeds the
  weighted-centroid vector through the EXISTING getMoveStick(...,AWAY,...) (reuses the AWAY
  flip + all wall handling = ZERO refactor). One hook after the adjacent dispatch overrides the
  flee move when fleeing (not STAY, not family). Smoke-verified: single threat -> unchanged
  flee; quark+hulk pincer -> no longer flees straight into the hulk. Known limit: a perfectly
  collinear pincer gives an axial (not perpendicular) escape — potential-field local-min, rare
  in play.
- VALIDATION launched: threat-field ON (ADJACENT_QUARK=75 + THREAT_FIELD=1, seed 777, N=100,
  fsm_threatfield_paired.log) paired vs the existing quark-fix-only baseline fsm_qf_on_paired.log
  (mean 13.07). Success = higher mean + fewer hulk deaths without losing the quark gain.
- BROAD threat-field RESULT: NET-NEGATIVE (paired 45 games vs qfix-only: delta -1.20, 40%
  win-rate). Diagnosis confirmed: applying the field to ALL flees hurts the common deep-wave
  SWARM (weighted centroid sits ~on the player -> weak/erratic flee; nearest-threat flee is
  decisive there). Helps only the rare 2-threat pincer.
- REFINED: added a PINCER GATE (FSM_FIELD_MAX_THREATS=3) — threatFieldMove returns None (defer
  to simple nearest-flee) unless 2..3 threats are in range. Smoke-verified: pincer engages,
  swarm/single defer. Re-validating gated version (FSM_FIELD_MAX_THREATS=3, seed 777, N=100,
  fsm_threatfield_gated_paired.log) vs the same qfix-only baseline. If still <= baseline, the
  threat-field idea is a dead end (stays opt-in OFF) and the Quark fix is the session deliverable.
- GATED threat-field RESULT: WORSE (paired 20 games: delta -2.65, 26% win-rate). So BOTH the
  broad (-1.20) and pincer-gated (-2.65) potential-field flees LOSE to the simple nearest-threat
  flee. THREAT-FIELD = DEAD END (stays opt-in OFF; default FSM unaffected). Confirms the
  project's long finding that global-threat-field reasoning is the real wall — and the naive
  potential field actively hurts (weak/erratic resultant vs decisive nearest-flee).
  - SESSION DELIVERABLE = the QUARK KITING FIX (ADJACENT_QUARK=75): +0.72 wave / 58% paired
    win-rate N=100, Quark deaths 32%->27%, now the FSM default. A real teacher improvement via
    focused structural work (per the user's "improve the FSM, not more RL" call).
  - Next idea for the hulk trade (documented, NOT a potential field): hulk-aware flee
    DEFLECTION — keep the decisive nearest-flee, only nudge it +/-45deg when it points straight
    at a nearby hulk. Surgical, avoids the swarm-dilution that killed the field.

## 2026-06-29 - STRUCTURAL FSM FIX #3 SHIPPED: hulk-aware flee deflection (+1.26 wave, stacks)
The surgical idea WON where the potential-field failed. `hulkDeflect()` (robotron_fsm.py,
FSM_HULK_DEFLECT, **now default ON**): keep the decisive nearest-threat flee; only when that
heading points within ~45deg of an invincible Hulk inside HULK_DEFLECT_R=60, nudge to the
nearest +/-45deg heading clear of hulks. Untouched in swarms/no-hulk-ahead -> no dilution.
- PAIRED N=100 (seed 777) vs quark-fix-only baseline: qfix-only 13.04 vs **+hulkdeflect 14.30,
  delta +1.26 wave, W50/L33 = 60% win-rate** (p~0.04). Strengthened monotonically with N
  (+0.35@23 -> +1.26@98), the opposite of the field's collapse.
- **COMBINED FSM gain this session (all structural, no RL): original evolved FSM ~12.35 ->
  quark kiting (+0.72) -> + hulk deflection (+1.26) ~= +2 waves over baseline.** Both default ON
  now (ADJACENT_QUARK=75, HULK_DEFLECT=1). The evolved FSM is the deployable best player and
  just got meaningfully better via targeted death-forensics-driven fixes — vindicates the
  "improve the teacher structurally, don't run more RL" direction.
- Threat-field (potential field) stays the documented DEAD END; the lesson = surgical,
  forensics-targeted deflection beats global potential fields for this evasion problem.
- NEXT (for a future session): re-run forensics on the combined FSM (expect Hulk share down);
  consider deflection for the other fast killers (EnfBullet/Spark 16%); then PROPAGATE the
  improved teacher downstream (collect demos -> BC -> anchored-RL) per [[project_fsm_teacher_evolution]].

## 2026-06-29 - WAVE-12 DEEP-DIVE (user-asked "is 14 the ceiling?") -> STRUCTURAL FIX #4
Combined-FSM death-by-wave: huge spike at WAVE 12 (116/646 = 18%, the single wall), plus
w7/w9/w17 (the quark/brain waves). 95% explained-direct (legit reaction-limited contacts, no
bugs). Wave-12 dissection: **killer = Quark 69%** (vs 28% overall) + Hulk 29%; **100% of w12
deaths have a Quark present AND 98% have a Hulk present**; NOT cornered (edge 74).
- ROOT CAUSE (code-confirmed): when a Hulk is the adjacent threat the FSM fires AT it
  (`getFireStick(adjacent)`) — but **Hulks are INVINCIBLE, so the shot is WASTED**. On wave 12
  (hulk near ~always) the FSM burns fire on un-killable hulks while the killable Quarks close in
  and contact-kill (69%). Twin-stick means it COULD flee the hulk and shoot the quark.
- FIX #4 (opt-in FSM_HULK_NOFIRE=1): adjacent-Hulk handler now fires at the nearest KILLABLE
  threat (`_nearest_killable` over projectile/chase(incl quark)/priority/enemy), falling back to
  the hulk only if nothing killable is in range. Smoke-verified: hulk-left+quark-right -> flee
  right, fire RIGHT at the quark (was firing left at the hulk).
- VALIDATION launched: FSM_HULK_NOFIRE=1 (+ quark+deflect defaults) paired N=100 seed-777
  (fsm_hulknofire_paired.log) vs the combined baseline fsm_hulkdeflect_paired.log (mean 14.27).
  If it cuts wave-12 quark deaths, this is the 3rd structural win and pushes the teacher past 14.
- FIX #4 RESULT: NEGATIVE (paired 61 games: delta -0.69, 44% win-rate; monotonic DECLINE
  +2.45@11 -> +0.96@26 -> -0.69@61, opposite of a real win's climb). Stays opt-in OFF.
  **KEY INSIGHT:** redirecting fire off invincible hulks is first-principles-correct yet doesn't
  help -> wave-12 quark deaths are TELEPORT-CONTACT (a positioning problem the deflection already
  addresses), NOT "the quark survived too long" (kill-speed). You can't react to a teleport, and
  killing quarks a bit faster doesn't stop one teleporting onto you. This is why the two MOVEMENT
  fixes (kiting, deflection) won but the FIRE fix didn't.
- CONCLUSION on "is 14 the ceiling?": for the REACTIVE FSM, ~14 is near the practical ceiling.
  The dominant remaining wall (wave 12 = 69% quark, the teleport-contact regime) is essentially
  IRREDUCIBLE for nearest-N reactive evasion — you can't react to a teleport. The two movement
  wins captured the avoidable gains (+0.72 quark kiting, +1.26 hulk deflection = ~+2 over the
  original evolved FSM, ~14.3 mean). Pushing materially past 14 needs either lookahead (search
  teacher — shelved, save-state lossy) or to PROPAGATE this improved teacher into the policy
  (collect demos -> BC -> anchored-RL). Structural FSM tweaking has reached diminishing returns.

## 2026-06-29 - HEADLINE: combined FSM = +1.92 wave / +23% score over the original evolved FSM
Definitive paired N=100 (seed 777, same games): ORIGINAL evolved FSM (fsm_evolved_reseed_v2,
no structural fixes) 12.35 / 220,803  ->  COMBINED (ADJACENT_QUARK=75 quark-kiting +
HULK_DEFLECT hulk-deflection, both default ON) **14.27 wave (+1.92) / 271,658 (+23%),
W55 L31 T14 = 64% win-rate, p~0.01.** Strong, statistically solid; all from death-forensics-
driven structural FSM work, ZERO RL. New deployable best player/teacher. Session deliverable DONE.
NEXT PHASE (deliberate start, ideally FRESH context — this session is very long): PROPAGATE the
+1.92 teacher into the policy (collect_fsm_demos w/ current FSM defaults -> BC -> anchored-RL per
[[project_fsm_teacher_evolution]]); expected policy lift modest (cloning loses ~3-4 waves).

## 2026-06-29 - PROPAGATION PREP COMPLETE (demos + BC warmstart ready); RL = fresh-session handoff
Two bounded prep steps done this session:
1. **Demos**: `demos/fsm_mame_demos.npz` = 800k pairs / 427 eps collected from the +1.92 COMBINED
   FSM (EVOLVE_PARAMS=fsm_evolved_reseed_v2.json + structural defaults ADJACENT_QUARK=75 +
   HULK_DEFLECT). This is the improved teacher's behavior.
2. **BC warmstart**: `models/bc_fsm_improved/` (final_model.zip + vec_normalize.pkl), 25 epochs,
   val move-acc 0.61 / fire-acc 0.58 (normal FSM-clone fidelity; BC-alone is weak ~median 2-3,
   it's a WARMSTART not a deliverable).
NEXT = anchored-RL fine-tune (the multi-hour, recipe-sensitive phase — do in a FRESH context):
   the best prior recipe is ANCHORED-RL (chunked PPO + separate-optimizer BC pull-back, see
   train_bc_anchored.py / [[project_fsm_teacher_evolution]]), NOT plain PPO-warmstart. Baseline
   to beat: bcanchor_v2anchor_best ~9.4 (that used the OLD v2 teacher; this demo set is from the
   +1.92-better teacher, so the hope is a higher policy plateau). Eval via eval_protocol.py port
   9970, paired, N>=100 ([[project_eval_protocol]]). Plain-PPO fallback command printed in
   bc_improved.log. All artifacts banked; nothing lost on /clear.

## 2026-06-29 - PROPAGATION RESTART: combanchor (combined-FSM teacher -> policy)
Loop restarted under user "make whatever changes necessary to reach our goal".
- LAUNCHED + COMPLETED `train_bc_anchored.py` -> `models/bcanchor_combanchor/`:
  ANCHOR_BASE=bcanchor_v2anchor_best (warm from the 9.41 best policy), ANCHOR_AGG=
  fsm_mame_demos.npz (800k pairs from the +1.92 COMBINED FSM = quark-kiting +
  hulk-deflect, N=100 paired 14.27 wave), MAME_RL_RESEED=1, NENV=16, 3M steps,
  PPO_CHUNK=65536 BC_STEPS=300 lr1e-4 ent0.005 bc_lr1e-4, CK_EVERY=200k. Log
  logs/anchored_combanchor.log. Final bc_nll 0.376 (healthy; 6M overfit hit 0.18).
- RATIONALE: changes exactly ONE variable vs the proven v2anchor recipe (the
  teacher demos), so the eval cleanly tests whether the +1.92 teacher propagates.
  Baseline to beat: bcanchor_v2anchor_best true N=100 = 9.41.
- NOW: triage eval (eval_continuous reseed N=30, ports 9971/73/75/77) on
  final / ck_2883k / ck_2621k / ck_2097k to pick the strongest, then the
  promotion gate = eval_protocol.py paired N>=100 port 9970 vs v2anchor_best,
  gated on life-economy (life-gen/wave >= deaths/wave). logs/triage/*.log.

## 2026-06-29 - PARALLEL TRACK: teacher re-evolution WITH structural constants (combteacher)
While the combanchor paired gate runs, launched the documented #1 ceiling-raiser
("raise the teacher") on idle compute. evolve_fsm.py EXTENDED: the 19 evolved
thresholds now also CO-OPTIMIZE the two 2026-06-29 structural constants
ADJACENT_QUARK (75) + HULK_DEFLECT_R (60) [HULK_DEFLECT bool stays default-ON].
Threshold-only evolution plateaued ~12.7; the +1.92 wave came from structural
fixes, so evolving their constants jointly is the path to push the teacher past
14.27. Warm-start tolerates the 2 keys being absent from v2.json -> defaults, so
evolution begins exactly at the current best teacher and explores around it.
- RUN: EVOLVE_WARM=fsm_evolved_reseed_v2.json MAME_RL_RESEED=1, gens=20 lambda=24
  mu=8 K=10 workers=12 port=9920 -> models/fsm_evolved_combteacher.json (best
  params checkpointed every gen). log logs/evolve_combteacher.log.
- GATE: validate the winner 40-game reseed before adopting (6-game fitness ran
  ~2.5 optimistic historically). If it beats the combined FSM's 14.27 N=100, it
  becomes the new teacher -> next propagation round clones a higher ceiling.

## 2026-06-29 - PROPAGATION VERDICT: combanchor = PARITY (teacher quality no longer the policy bottleneck)
Paired N=100 held-out (port 9970, reseed), combanchor ck_2883k (A) vs bcanchor_v2anchor_best (B):
- A: mean 9.45 [8.83-10.09] median 9.0 | P(w>=10)=0.42 P(w>=12)=0.33 | life-gen/wave 0.58 deaths/wave 0.79
- B: mean 9.29 [8.80-9.84] median 9.0 | P(w>=10)=0.38 P(w>=12)=0.26 | life-gen/wave 0.60 deaths/wave 0.79
- PAIRED delta A-B = +0.16 [95% -0.62..+0.93]; A wins 48 / ties 13 / B wins 39. INCONCLUSIVE (CI spans 0).
- A has a FAINT deep-wave edge (P(w>=12) 0.33 vs 0.26, more w12/13 hits) but mean is parity. NO PROMOTION
  (gate = clear beat + sustainability; A also bleeds lives life-gen<deaths). bcanchor_v2anchor_best stays
  best policy; combanchor preserved as co-equal.
- STRATEGIC: this is the SECOND teacher-jump -> ~parity policy. v2anchor (teacher 12.72) -> 9.41;
  combanchor (teacher 14.27, +1.55) -> 9.45 (+0.04). The policy is PINNED ~9.4 regardless of teacher.
  The bottleneck has MOVED from teacher-quality to the CLONING/RL gap itself => raising the teacher
  further (the combteacher evolution) will NOT lift the policy; its value is now ONLY the deployable
  FSM PLAYER (14.27 best), not as a propagation teacher. Breaking the NN policy past 9.4 needs a
  cloning-gap attack, but obs/arch/recurrence/curriculum/from-scratch are all documented dead ends.

## 2026-06-30 - NEW BEST FSM PLAYER: combteacher (+2.84 wave paired over combined)
Extended evolve_fsm (co-optimizing ADJACENT_QUARK + HULK_DEFLECT_R with the 19 thresholds),
warm from fsm_evolved_reseed_v2, reseed, K=10, 20 gens -> models/fsm_evolved_combteacher.json.
Winner pushed structural consts UP: ADJACENT_QUARK 75->84.2 (kite quarks from further),
HULK_DEFLECT_R 60->90.1 (deflect hulks earlier); ADJACENT 49->45.8, CLOSE->69.7.
HONEST PAIRED VALIDATION (validate_fsm.py, port 9968 reseed HELD-OUT from evolution ports
9920-42, same seeds both configs, N=100):
- combteacher mean 12.64 / median 12.0 / max 20
- combined FSM (reseed_v2 + structural defaults 75/60) mean 9.80 / median 9.0 / max 18
- PAIRED delta +2.84, W66/L28/T6 = 70% win-rate (66/94 decisive, p<<0.001). ADOPTED.
- NOTE: abs means here are lower than the seed-777-harness "14.27" because MAME_RL_RESEED
  port-LCG != MAME_RL_SEED_BASE seeding; only the paired delta is comparable. combteacher is
  decisively the better PLAYER. Co-evolving the structural constants (not just thresholds) was
  the unlock -> threshold-only had plateaued ~12.7.
- This improves the deployable FSM PLAYER (the product). It does NOT change the NN-policy story
  (propagation is parity-bound ~9.4 regardless of teacher; see prior entry) so NOT re-propagating.
- NEXT: evolution round 2 warm from combteacher (K=12) to test if co-evolution has more juice.

## 2026-06-30 - EVOLUTION SATURATED (round 2 lost); pivot to death-forensics on combteacher
combteacher2 (evolve round 2, warm from combteacher, K12) fitness fell to 14.00 < round-1 14.50,
and HONEST paired N=60 (held-out port 9968) = combteacher2 11.65 vs combteacher 12.50, delta
-0.85, W19/L32/T9 = 37% win-rate. NOT adopted. FSM threshold+structural-constant evolution is
SATURATED at the combteacher level (the v2->combteacher +2.84 was the real, banked win).
=> Pivoting to the death-forensics lever: collect combteacher deaths, audit, find the next
SURGICAL structural fix (the path that gave quark-kiting +0.72 and hulk-deflection +1.26).

## 2026-06-30 - FORENSICS on combteacher -> STRUCTURAL FIX #5: edge-aware flee deflection
combteacher death audit (665 deaths, 60 games, 97.3% explained-direct = no bugs):
- Walls: wave 12 (115, Quark 64%/Hulk 34%), wave 7 (99, Quark 81%), wave 17 (39, Quark 74%) =
  the IRREDUCIBLE quark-teleport-contact waves (253 deaths/38%; hulk-deflect already cut Hulk
  26%->15% vs combined FSM).
- #2 OVERALL killer = EnfBullet/Spark 20% (134 deaths), and 73% of those are at board
  EDGES/CORNERS (edge 66 + corner 32 vs open 36) + 51 grunt edge deaths => the FSM flees INTO an
  edge and a fast spark traps it in the 20-70px band BEFORE the at-the-wall handler (margin
  2-20px) fires. This is the dominant AVOIDABLE pattern.
- FIX #5 (opt-in FSM_EDGE_DEFLECT=1, robotron_fsm.py): edgeDeflect() — when a flee heading drives
  into a near edge (within EDGE_DEFLECT_M=55px), nudge +/-45deg to slide ALONG it. Same surgical
  shape as the winning hulkDeflect; runs right after it in the flee path. Smoke-tested (deflects
  off all 4 edges, leaves mid-board / along-edge headings untouched).
- VALIDATION: paired N=100 held-out port 9968, combteacher+EDGE_DEFLECT vs combteacher. ADOPT only
  on a clear positive paired delta. If parity/negative -> both levers (evolution+forensics)
  saturated -> summarize + stop.

## 2026-06-30 - EDGE-DEFLECT = DEAD END; LOOP CONCLUDED (both levers saturated)
Fix #5 (edge-aware flee deflection) HONEST paired N=100 (held-out port 9968):
edge-deflect 12.24 vs combteacher 12.64, delta -0.40, W35/L51/T14 = 41% win-rate. NEGATIVE.
Stays opt-in OFF. The spark-edge clustering was CORRELATION not causation — forcing flee off
edges disrupts the decisive nearest-flee (same failure shape as threat-field & hulk-nofire).

=== AUTONOMOUS WAVE-100 LOOP — SESSION CONCLUSION (2026-06-30) ===
Banked win this session: NEW BEST PLAYER models/fsm_evolved_combteacher.json, +2.84 wave paired
over the combined FSM (held-out reseed port 9968, 70% win-rate) — from EXTENDING evolve_fsm to
co-optimize the structural constants (ADJACENT_QUARK 75->84, HULK_DEFLECT_R 60->90) with the 19
thresholds. Hulk deaths cut 26%->15%.
Both ceiling-raising levers are now SATURATED:
  - EVOLUTION: round 2 (warm from combteacher, K12) regressed -0.85 paired -> threshold+structural
    evolution maxed at the combteacher level.
  - FORENSICS: combteacher 665-death audit -> dominant AVOIDABLE pattern (EnfBullet/Spark 20%, 73%
    at edges) -> surgical edge-deflect fix FAILED paired (-0.40). Remaining walls are the
    IRREDUCIBLE quark-teleport waves (7/12/17 = 38% of deaths; can't react to a teleport) and the
    grunt-open-field global-threat-field problem (the potential-field fix already failed).
Also confirmed this session: NN-policy propagation is PARITY-bound ~9.4 regardless of teacher
quality (combanchor 9.45 vs 9.41 baseline, INCONCLUSIVE) — teacher gains accrue to the FSM PLAYER,
not the policy.
STRATEGIC FORK (needs USER): reactive-FSM ceiling reached (~12.6 reseed / max 24). Reaching wave
~40+/100 requires value-guided LOOKAHEAD/SEARCH, which is shelved because MAME save-state is LOSSY
for this ROM (per-step rollout corrupts the live game) and the analytic 1-step forward model only
reached PARITY. No autonomous path past here without that strategy call. LOOP STOPPED (no re-arm).

## 2026-06-30 - LOOP RE-OPENED by user: pursue the LOOKAHEAD/SEARCH fork (multi-step analytic)
User: "continue autonomously toward 100, only quit if truly outside ability." Pursuing the one
lever prior search work flagged as NOT fully dead: multi-step analytic lookahead (1-step was
parity; save-state rollout is lossy). Built VSEARCH_CLEAR in mame_gym/search_value.py: a
multi-step analytic CLEARANCE planner — commit to each of 8 headings for H=6 env-steps, re-aim
CHASERS (grunt/hulk/brain) toward the predicted moving player each step (ballistic projectiles
fly straight; quark/electrode static), score the heading by WORST weighted clearance over the
horizon; GUARDED override (only replace the FSM move when the FSM heading clearance < CLEAR_DANGER
26 AND an alt beats it by CLEAR_MARGIN 8). Geometric (no learned-V drift). Directly attacks the
open-field CONVERGENCE deaths (faster chasers pincer the FSM's straight-flee). Unit test confirmed
it prefers DIAGONAL evasion vs a fast chaser (clearance 52.7) over straight-flee (39.0) — the
non-obvious move the reactive FSM misses. Validating paired N=70 (CLEAR vs FSMONLY) port 9968.

## 2026-06-30 - *** BREAKTHROUGH: minimal-deviation multi-step clearance search BEATS the FSM (+3.27) ***
The FSM-override paradigm finally WON — first method in the project to clearly beat the reactive FSM.
- AGGRESSIVE clearance (global argmax-clearance, DANGER=26): LOST -2.71 (abandons FSM offense).
- MINIMAL-DEVIATION clearance (override only when FSM heading clearance < DANGER=18, then pick the
  SAFE heading NEAREST the FSM intent; H=6 analytic horizon, chasers re-aim at predicted player):
  PAIRED N=60 port 9968 reseed -> CLEAR-v2 15.77 / median 17 / max 29 vs combteacher 12.50 / 12 / 20,
  **delta +3.27, W39/L18/T3 = 68% win-rate** (p<<0.001).
- WHY it works where 4 prior overrides failed: it preserves the FSM's tightly-tuned reactive move
  almost always, intervening ONLY to dodge predicted death and then by the SMALLEST course change
  (minimal-deviation) — so it adds genuine multi-step convergence-avoidance without breaking the
  FSM's coordinated move+fire+rescue. Max wave 29 reaches the period-20 generalization band.
- Flags: VSEARCH_CLEAR=1 VSEARCH_CLEAR_DANGER=18 VSEARCH_CLEAR_MARGIN=10 (H=6). Code in
  mame_gym/search_value.py (clearance_search + _heading_clearance + _classify_threat).
- NOW: N=100 paired confirm (port 9968) + parallel deeper-horizon sweep H=10 (port 9950). If
  confirmed, deployable best = combteacher + min-dev clearance override; CORRECT the standing
  "no method beats the FSM / search dead" memory.

## 2026-06-30 - *** CONFIRMED & ADOPTED: clearance-search beats the FSM +2.57 @ N=100 ***
Track A N=100 paired (port 9968 reseed): CLEAR (min-dev, H=6, DANGER=18, MARGIN=10) mean 15.21 /
median 17 / max 29 vs combteacher 12.64 / 12 / 20, **delta +2.57, W65/L33 = 66% win-rate**
(p<<0.001). H=10 variant also won (+1.67) but less than H=6. BREAKTHROUGH CONFIRMED & ADOPTED.
NEW DEPLOYABLE BEST = combteacher FSM + minimal-deviation multi-step analytic clearance override.
DEPLOY RECIPE: run via mame_gym/search_value.py logic (clearance_search + _heading_clearance +
_classify_threat) with VSEARCH_CLEAR=1 VSEARCH_CLEAR_DANGER=18 VSEARCH_CLEAR_MARGIN=10 (H=6) and
EVOLVE_PARAMS/SRC=models/fsm_evolved_combteacher.json. For live deploy the clearance_search needs
porting into the player loop (currently lives in the search_value eval harness). This is the FIRST
method in the project to beat the reactive FSM — corrects the "no method beats FSM / search dead"
conclusion. Max wave 29 reaches the period-20 generalization band (-> path toward 100).
NEXT: sweep DANGER {14,24} (ports 9952/9954) to push depth further; then threat-weight tweak.

## 2026-06-30 - Clearance sweep: DANGER robust (14/18/24 all +2.6..+3.1); trying weights + H=8
DANGER=14 +2.65 (16.27 vs 13.62, 69%), DANGER=18 +2.57 [N=100 champion], DANGER=24 +3.08 (15.75
vs 12.67, 65%) — all paired N=60 on their own ports. Deltas overlap within noise => DANGER not a
strong knob; method robust across 14-24. Champion stays DANGER=18 H=6 (the N=100-confirmed one).
Next substantive levers: threat weights (Hulk/projectile wider berth) + horizon H=8.

## 2026-06-30 - Clearance sweep CONVERGED: champion = DANGER=18 H=6 orig-weights (+2.57 N=100)
Sweep results (all paired N=60, own-port baselines): H=10 +1.67, H=8 +1.27, weights(Hulk3.0/proj2.2)
+1.73, DANGER14 +2.65, DANGER24 +3.08, DANGER18[champion] +2.57 @N=100. H=8/H=10/weights all BELOW
the champion band (+2.6-3.1) => champion H=6/DANGER=18/orig-weights stands; weights edit REVERTED.
Knob-tuning converged. Now: death-forensics on the CHAMPION (clearance search) to see if the
residual deep-wave deaths are the irreducible quark-teleport (=> method ceiling, report) or a NEW
avoidable pattern (=> next structural lever). deaths_clearance/, N=60.

## 2026-06-30 - CHAMPION FORENSICS -> ceiling diagnosed: QUARK-TELEPORT wall gates wave 100
Champion (clearance search) 848 deaths / 60 games. Death distribution pushed DEEP (peaks now
waves 7/12/17/22 vs combteacher 7/12) — the clearance search solved the CONVERGENCE deaths
(that's the +2.57 and reaching wave 22-29). Residual killers:
- QUARK 32% — concentrated on the period-5 quark waves (7/12/17/22: Quark 78-98%, quark-present
  95-100%). TELEPORT-ONTO-YOU = instant death, no reaction possible at ANY control frequency; the
  clearance planner can't help (quark's current position is irrelevant to teleport landing). This
  is a STOCHASTIC, IRREDUCIBLE wall for a reactive agent.
- EnfBullet/Spark 25% — non-quark waves; fast projectiles crossing the frameskip-4 action gap
  (control-frequency limited). Finer frameskip could help these but NOT the quark wall. And the
  weight-increase tweak already LOST (over-avoids).
- Hulk/Grunt/Brain ~36% mixed.
HONEST CEILING: wave 100 requires surviving ~19 quark waves in a row; each carries an
unavoidable teleport-onto death probability > 0 -> compounds to ~0 survival. This is a
game-mechanic-vs-reactive-agent barrier, not an effort/tuning limit. Finer control (frameskip)
or better offense (quark-kill-speed) might add a few waves but will NOT reach 100.
=== FINAL DELIVERABLE ===
NEW BEST PLAYER (first ever to beat the FSM): combteacher + minimal-deviation multi-step analytic
clearance override. CONFIRMED +2.57 wave @N=100 (15.21 mean / median 17 / max 29 vs FSM 12.64,
66% win). Flags: VSEARCH_CLEAR=1 VSEARCH_CLEAR_DANGER=18 VSEARCH_CLEAR_MARGIN=10 H=6; code in
mame_gym/search_value.py (clearance_search + _heading_clearance + _classify_threat). Deploy = port
clearance_search into the live player loop. LOOP STOPPED: incremental iteration ceiling reached;
wave 100 needs a non-reactive substrate (a fundamentally different approach to the quark wall).

## 2026-07-01 - FULL ASM ANALYSIS -> strategy verdict + Stage-1 planner dynamics fix (A/B running)

Full-repo strategy review (user-asked: "best way to 1->100, consider ALL options"). Read the
complete robomame.asm difficulty/wave machinery + full experiment history. VERDICT: stay on the
FSM+clearance-planner paradigm (the only method to ever beat the FSM) and make its forward model
ASM-exact; RL is exhaustively falsified (pinned ~9.4 across ~20 variants regardless of teacher).

KEY ASM FINDINGS (now in ENEMY_MODEL.md §9):
- Difficulty is BOUNDED: all $BE5C-$BE67 tables clamp+plateau; waves 41+ replay 21-40
  (asm:4030-4033). Wave 100 = the 21-40 band four more times. Not a wall.
- RNG = 3 readable bytes $9884-86, advanced only by 3 consumer wrappers -> exact forward
  model / RNG-phase tracking is feasible.
- Tank shells: two firing modes incl. deliberate WALL-BANK shots; COM-based (approximate)
  reflection; and the $F1 counter bug — shells that expire naturally are never decremented, so
  after ~20 fired shells TANKS FALL SILENT for the wave. Exploit: dodge, don't shoot, shells.
- Enforcers dive with velocity ∝ distance (slow when CLOSE); sparks never re-home; cruise
  re-aims only every <=8 ticks; brains chase HUMANS not the player; hulks with no family target
  wander (null-target bug); grunt speed ramps as grunts die (player-controlled pacing).
- Wave-clear set = grunts+spheroids+enforcers+brains+tanks+quarks only; wave-start safe
  rectangle (frozen at wave-6 size from w10) guarantees an opening window; no free life per
  wave (the INC at 2A9A compensates the respawn DEC at 27AA); extra lives are score-only.

FIXES SHIPPED:
1. robotron_server.lua MAX_ENTITIES 120 -> 190 (ENEMY_MODEL §7 item 6: list-1 floods could
   starve family/electrodes out of the obs; pool is 180 records so 190 = no truncation ever).
2. search_value.py Stage-1 ASM-TRUE DYNAMICS (VSEARCH_ASMDYN=1, default ON; =0 restores the
   pre-fix model for A/B): quark = slow wanderer weight 1.0 (was static "teleporter" w2.0 — a
   decoder phantom); TankShell = ballistic + WALL-REFLECT with velocity negation (was straight);
   CruiseMissile = chaser (homes, was ballistic); Enforcer = chaser (dives, was debris);
   Brain = ballistic (chases humans, was player-chaser); off-board extrapolation now clamps at
   borders like the game integrator (threats no longer drift off-board and vanish).
   Unit-tested (scratchpad test: bounce threat detection, weights, chase/ballistic contract).

A/B RUNNING (paired, MAME_RL_SEED_BASE=777, N=100 each, single-var):
- OFF (old dynamics) port 9958 -> logs/clear_asmdyn_off_n100.log
- ON  (ASM dynamics) port 9960 -> logs/clear_asmdyn_on_n100.log
Both: combteacher + VSEARCH_CLEAR=1 DANGER=18 MARGIN=10 H=6, corrected labels (both arms).
EARLY: paired g1 OFF wave 20 vs ON wave 24. NOTE: absolute level >> the old 15.21 champion run
(OFF g2 hit wave 26 / 535k and the 8000-STEP CAP -> the label fix alone already lifted play a
lot, and deep games are now TRUNCATED by the cap; raise steps cap for future depth evals).

NEXT (staged plan, see session analysis): Stage 2 = plan FIRE jointly (shoot-incoming-projectile,
hulk-push-when-pinned as PLANNED escapes, quark/spheroid-first opening) + dodge-don't-shoot-shells
policy ($F1 exploit) — each as opt-in flags, validated paired N=100 separately. Stage 3 = exact
Python forward model synced from RAM (records + $BE5C-67 + RNG) validated frame-parity vs MAME,
then beam search over move sequences H~16-32. Stage 4 = rescue-routing life economy + per-wave
opening book. Milestones: median 25 -> median 40 -> (periodicity) 100.

## 2026-07-01 - A/B in flight: WEDGE-BREAKS-PAIRING pitfall found (+ re-pair rule)
On-arm (port 9960) MAME wedged during game 20 -> bridge auto-relaunched. The relaunched Lua
process re-inits rng_seed = MAME_RL_SEED_BASE (777), so the seed LCG RESTARTS: game 20 is a
recovery artifact (wave=1, exclude) and games 21+ REPLAY the sequence from seed #1. Verified:
on g21 (w24/444k) == the same game as off g1 / original on g1 (w24; scores differ ~5% because
the module-global obs-builder velocity context leaks across games -> not byte-identical).
RE-PAIR RULE for analysis: pair on[1..19]<->off[1..19]; after a wedge marker at game W, pair
on[W+k]<->off[k]. Wedge markers: grep "instance wedged" in the log; seed index = games since
last marker. THIS APPLIES TO ALL SEED_BASE PAIRED A/Bs with the self-healing bridge (past ones
should be checked for unnoticed wedges). Also: on-arm g13 burned the full 40k-step cap stuck
at wave 11 (no-progress stall, single occurrence) — watch item; if stalls recur under ASMDYN,
add an anti-stall nudge.

## 2026-07-01 - A/B on-arm: DETERMINISTIC wedge loop found + seed-skip resume
Second wedge on port 9960 produced byte-identical replays (g37==g18, g38==g19, g39==g20
artifact, g40==g21): the wedge is DETERMINISTIC — the on-arm cycles seeds 2-19, wedges at the
same reset every time (after the seed-19 game's death at wave 8), and would loop forever,
never reaching seed 20. (Off-arm plays its seed-19/20 games fine -> a state-specific MAME
hang on that reset, not an ASMDYN code path.) Seed accounting: the recovery artifact game
consumes seed 1 (aborted ~206 steps), so post-wedge game W+k has seed k+1.
FIX: VSEARCH_SEED_SKIP env in search_value.py main() — burns K resets (LCG advances once per
reset) so play resumes at seed K+1; printed game numbers include the offset (seed-aligned).
On-arm relaunched with SEED_SKIP=19, N=81 (seeds 20-100), appending to the same log. Valid
on-arm data: games 1-19 = seeds 1-19 (minus artifacts); resumed run = seeds 20-100.
WATCH: if the resumed run wedges immediately at seed 20, that reset itself is the poison ->
resume again with SEED_SKIP=20 and drop the seed-20 pair.

## 2026-07-01 - VERDICT: ASM-dynamics planner upgrade = PARITY; label fix was the unlock (new baseline 22.7)
Paired A/B complete (n=97 valid paired seeds after wedge re-pairing + seed-skip resume + shard):
- OFF (old dynamics):  mean 22.69 / median 22 / max 43 (N=100 full-run: 22.73/22/43, max score 923,475)
- ON (ASM dynamics):   mean 22.86 / median 22 / max 42
- PAIRED delta +0.16, 95% CI [-1.94,+2.27], W/L/T 46/49/2 -> STATISTICAL PARITY.
Early +3.0@n=8 regressed to parity, the project's signature pattern (trust only N~100).
TWO conclusions:
1. Stage-1 ASM-true dynamics (VSEARCH_ASMDYN) do NOT add depth at H=6 with the rarely-firing
   min-deviation override — model fidelity matters at too few decision points. KEPT as default
   ON anyway (it is ground truth, costs nothing, and deeper horizons need it); the old model
   remains at ASMDYN=0 for reference.
2. THE REAL FINDING: the corrected SW labels (2026-07-01 decoder fix, present in BOTH arms)
   transformed the champion: recorded 15.21 mean / max 29 -> **22.7 mean / median 22 / max 43**
   on the same code+config. Max-43 games cross into the period-20 repeat band. The "quark
   teleport wall" conclusion is emphatically dead: with correct perception the same planner
   plays ~8 waves deeper.
NEW DEPLOYABLE BASELINE: combteacher FSM + clearance override (VSEARCH_CLEAR=1 DANGER=18
MARGIN=10 H=6, ASMDYN=1, corrected labels): 22.7 / 22 / 43 on seed-777 games (port-9958/9960
harness). NOTE: not directly comparable to the old port-9968 reseed numbers (different seed
mechanism); for promotion-grade numbers re-run eval_protocol.py on port 9970.
NEXT LEVERS (unchanged by this wash, now attacking from a 22.7 floor):
- Stage 2: planned FIRE (shoot-incoming, hulk-push-when-pinned), $F1 dodge-don't-shoot-shells
  exploit, quark-first openings — new capabilities, not better estimates.
- Stage 2b: re-test H=8/10/16 with ASMDYN=1 (the old H-sweep losses used the WRONG dynamics;
  correct dynamics should compound with horizon).
- Stage 3: exact RAM-synced forward model + beam search over move sequences.

## 2026-07-01 - Stage-2 arms launched + Stage-3 sim-state plumbing landed
FOUR paired arms in flight (all seed-777, vs completed baselines):
- H=10 ASMDYN=1 (port 9964, logs/clear_h10_asmdyn_n100.log) and H=14 (9966, clear_h14_...):
  re-test of the horizon sweep with CORRECT dynamics (old H-sweep losses used the wrong model).
- FSM_NO_SHOOT_SHELLS=1 (9952, clear_noshootshells_n100.log): $F1 shell-budget exploit —
  fire redirected off TankShells to the nearest killable (flee unchanged; smoke-tested).
  Implementation in robotron_fsm.py (nearestKillableProjectile/adjacentIsShell bookkeeping).
- VSEARCH_FIREPLAN=1 (9954, clear_fireplan_n100.log): planned fire — when the clearance
  override engages (predicted death), fire at the BINDING threat of the chosen escape heading
  (argmin-clearance entity): Hulk within FIREPLAN_HULK_R=90 => push it open; anything else =>
  shoot it. Unlike reactive HULK_PUSH/NOFIRE (both failed), only active in the predicted-death
  regime. 2-game probe sane. _dir_toward compass quantization unit-tested (8/8).
STAGE-3 PLUMBING (exact forward model substrate) LANDED + probe-validated:
- robotron_server.lua MAME_EMIT_SIM=1: appends per-entity $0A-$11 (8.8 pos/vel) + $12/$13
  (AI countdowns) + 27 global bytes ($BE5C-67 difficulty, $9884-86 RNG, $BE68-71 counts,
  $98F0/F1 spark/shell live counters). mame_bridge.py consumes; mame_obs.parse_sim_state()
  parses (validated live: wave-1 = 15 grunts/5 electrodes/mom/dad matches the ROM tables
  exactly; RNG advances per step; velocities respect ASM bounds; count-name mapping fixed
  to cur_grunts=$BE68 per asm:3834).
NEXT SESSION: build the exact Python forward model on parse_sim_state (per-type update rules
from ENEMY_MODEL.md §4/§9 + robomame.asm), validate frame-parity vs MAME (predict next
sim-state from current, compare over long rollouts), then beam search over move sequences.
Analysis note for the four arms: use the wedge-aware seed re-pairing (see 2026-07-01 pitfall
entry); baselines = clear_asmdyn_off_n100.log (old dyn) / on_n100+shard2 (ASMDYN).

## 2026-07-01 - CRITICAL EVAL INSIGHT: seed-pairing is nearly WORTHLESS for planner agents (r~0.18)
Correlation check across paired runs: even the CONTROL pair (asmdyn off vs on, near-identical
configs, same seed-777 games) shows per-seed wave correlation r=0.178; the four Stage-2 arms
vs baseline: r=-0.21..+0.46. Same-seed games DIVERGE CHAOTICALLY within a wave or two (one
different override decision and the trajectories separate), so "paired by construction" gives
~no variance reduction for planner-class agents — per-seed W/L and paired CIs are effectively
UNPAIRED. Consequences: (1) only large-N MEANS are meaningful (the eval_protocol paired-CI
machinery overstates precision for these agents); (2) mid-run subset comparisons are biased —
the ASMDYN baseline ran ~1.5 waves hot on early seeds, inflating the four arms' apparent -5
deltas at n~20-40. Full-N verdicts required before killing any arm.
EARLY (biased) standings vs ASMDYN baseline: H10 -4.8@37, H14 -5.4@34, FIREPLAN -5.9@20,
NOSHOOT -5.2@19 — all directionally consistent with the project's core lesson (any increase
in override/deviation frequency from the tuned FSM+H6 recipe costs waves), but final numbers
at N~100 will be substantially less negative after subset-bias washes out.

## 2026-07-01 - H-sweep VERDICT (monotonic loss) + RESCUE_SEEK feature launched
Full-N horizon results (vs ASMDYN H=6 baseline 22.86/22/max42):
- H=10: mean 18.19 / median 18 / max 38 (-4.7)
- H=14: mean 17.18 / median 17 / max 35 (-5.7)
Monotonic H6>H10>H14 — longer commitment horizons over-override and lose, even with correct
dynamics. H=6 definitively stays champion. (NOSHOOT/FIREPLAN arms still running, both
trending negative at n~60-70.)
NEW FEATURE (user-directed): FSM RESCUE-SEEKING. Audit found family collection purely
opportunistic: pursuit only within CLOSE_MOVE_CIVILIAN (~92px), family loses every adjacency
contest, and the idle fallback preferred chasing spheroids/quarks over rescuing. Yet family
is the life engine (1000->5000 escalation, 15-25 family on brain waves = 65k+ = 2.6 lives on
w5 alone; measured sustainability deficit only ~0.1 lives/wave). Mikey bug ASM-verified
(asm:2394-2405 + BRAIN_AI 1BF5/1BFF-1C08): all brains initially target $B354 (=Mikey), only
retarget when the target dies/is rescued, and NEVER chase the player while any human lives ->
brain waves are structurally safe to sweep; keeping Mikey alive keeps brains clustered.
Implementation (robotron_fsm.py, opt-in): FSM_RESCUE_SEEK=1 (idle fallback: rescue outranks
spheroid/quark hunting; nearest-civilian distance gate) + FSM_BRAIN_RESCUE_MULT=4 (civilian
pursuit radius x4 while a Brain is on screen). Smoke-tested. A/B launched port 9958
(logs/clear_rescueseek_n100.log, self-healing harness) vs ASMDYN baseline.
Note the honest life-economy framing (user question "push mean >40?"): the target is
deaths/wave < life-gen/wave (~0.8 at deep-band score rates) — crossing it flips games from
finite to unbounded; mean-40 follows discontinuously rather than incrementally.

## 2026-07-01 - NOSHOOT VERDICT: -3.04 [CI -5.2,-0.9] n=94 — $F1 exploit as blanket policy LOSES
Shooting shells was protective (shot-but-too-late deaths would rise); tank-silencing payoff
doesn't compensate. FSM_NO_SHOOT_SHELLS stays opt-in OFF. Stage-2 scoreboard so far:
H10 -4.7, H14 -5.7, NOSHOOT -3.0, FIREPLAN pending (~-5 trend), RESCUE_SEEK early but
promising (g10 = 25.9k/wave score rate vs baseline ~20.5k — above the 25k/wave
self-sustainability threshold if it holds).

## 2026-07-01 - FIREPLAN VERDICT: -0.85 [CI -3.1,+1.4] n=95 — parity, stays opt-in OFF
Stage-2 FINAL scoreboard (all vs ASMDYN H=6 baseline 22.9): H10 -4.7, H14 -5.7,
NOSHOOT -3.0, FIREPLAN -0.85 (parity). Every death-avoidance perturbation of the champion
lost or tied — the tuned FSM+minimal-override sits at a sharp local optimum for SURVIVAL.
The remaining live arm attacks the OTHER side of the ledger: RESCUE_SEEK (life economy),
early read n=20: wave parity (-0.3) with SCORE +23% (627k vs 509k, median 26 vs 24).

## 2026-07-01 - SCORE WAS WRAPPING AT 1M (fixed) + new all-time records from RESCUE_SEEK
RESCUE_SEEK game 60 reached **wave 48** (all-time depth record, prev 43) but read "score
408k" — impossible for that depth. ROOT CAUSE: p1_score is **4 BCD bytes $BDE4-7**
(asm:131); the obs header only carried $BDE5-7, so scores wrapped at 1,000,000. That game
actually scored **1,408,775** (all-time score record). FIX: header extended to 11 bytes
(byte10 = $BDE4 millions), parse_header/parse_obs_header decode it, SLOT_BASE_OFFSET 11.
Verified end-to-end on a fresh instance (entities/types/obs all correct). Affects: env
reward (score_delta went hugely negative on wrap — irrelevant for eval arms, matters for
any future RL), life-economy math (score//25000 undercounted past 1M), and eval logs.
HISTORY IMPACT: no prior game had crossed 1M (project max was 923,475), so all historical
numbers stand; only RESCUE_SEEK deep games were undercounted — its score gain is a FLOOR.
The RUNNING rescue-seek arm still has the old reader in memory: its logged scores for >1M
games are (true - 1M). Also fixed SIM_GLOBAL_NAMES: $98F0 = score-update counter (asm:121),
not a spark counter.

## 2026-07-01 - GOAL SET (user): wave 1->100 on REAL XBOX via YOLO screen input. Perception plan.
Deployment reality check (user): the XBLA build has its OWN RENDER LAYER — MAME pixels do
NOT match Xbox output. Perception pipeline therefore:
1. MAME = strength/training env (ROM logic identical; everything stands).
2. XENIA = the YOLO dataset source: run the bot there via XeniaMemory (memory maps validated
   identical 2026-06-09), capture Xenia's RENDERED frames + RAM ground truth -> auto-labeled
   dataset matching real-hardware visuals. (Check the xbox decompiled code for the render
   layer's coordinate transform when building the Xenia collector.)
3. Real Xbox: YOLO detections -> entity list -> same FSM+planner (the champion substrate
   already consumes entity lists — structurally YOLO-ready).
SCAFFOLDING BUILT (ports to Xenia): mame_gym/collect_yolo_dataset.py — FSM plays from the
432-state pool, snapshots via CMD_SNAP (bridge gained MAME_SNAP_DIR), writes YOLO-format
labels from RAM ground truth (blit dest = sprite top-left; box centers corrected). Pilot
verified visually: 292x240 snaps, boxes land on sprites (overlay_check.png in scratchpad).

## 2026-07-01 - *** RESCUE_SEEK ADOPTED: +35% score, all-time records wave 48 / 1,148,725 ***
FINAL (n=96 vs ASMDYN baseline): wave 24.27 vs 23.00 (+1.27, CI [-0.8,+3.4] — at worst
neutral); SCORE 621k vs 459k (+35%, decisive; a FLOOR since pre-restart >1M games logged
wrapped). Per-wave income 25.6k/wave — ABOVE the 25k sustainability threshold (baseline
20.0k). Records: max wave 48, max score 1,148,725 (~46 extra lives earned in one game).
ADOPTED into the champion recipe: FSM_RESCUE_SEEK=1 FSM_BRAIN_RESCUE_MULT=4.
LIFE-ECONOMY READ: deep-band deaths/wave ~1.0 vs income now ~0.96 — the margins race at
parity (that's why max-wave jumped 42->48). Income side is (nearly) solved; the binding
constraint for wave 100 is now the DEEP-BAND DEATH RATE (the 81% positioning problem ->
Stage-3 exact-model search / replan-every-step).
DEPLOY RECIPE (champion v2): clearance_planner.py (shared module, extracted from
search_value) + combteacher params + VSEARCH_ASMDYN=1 DANGER=18 MARGIN=10 H=6 +
FSM_RESCUE_SEEK=1 FSM_BRAIN_RESCUE_MULT=4. eval_protocol.py now supports spec
"clear:<json>" — OFFICIAL N=100 books run LAUNCHED on held-out port 9970
(logs/official_champion_9970.log).
NEXT levers (priority): (1) BRAIN_RESCUE_MULT sweep 2/6 + Mikey-last sequencing (keep the
brain cluster baited); (2) deep-band death rate via Stage-3; (3) Xenia YOLO collector for
the hardware goal (scaffold done, see project_goal_xbox_yolo).

## 2026-07-01 - BRAIN_RESCUE_MULT is a NO-OP under RESCUE_SEEK (sweep killed)
m2/m6 sweep arms produced byte-identical games to the mult-4 arm (seeds 1-50 checked): with
FSM_RESCUE_SEEK=1 the idle fallback already pursues the nearest civilian at ANY distance, so
the radius-gated civilian branch (where the mult applies) is subsumed — the only divergence
window (crowd-STAY between the two branches) evidently never fires. The +35% rescue gain is
from RESCUE_SEEK alone. CHAMPION RECIPE SIMPLIFIED: FSM_RESCUE_SEEK=1 (mult irrelevant,
dropped). Sweep arms killed; compute freed for the official 9970 run.
OFFICIAL RUN early: g10 wave 58 / 1.57M (~62 extra lives), g30 WAVE 67 / 1.89M (~75 extra
lives) — records obliterated; first games deep into the repeat band; wave 100 in reach of
the distribution tail.

## 2026-07-01 - *** OFFICIAL BOOKS (port 9970, n=98): mean 32.6 / MEDIAN 34 / MAX 69 ***
Champion v2 = clear:combteacher + ASMDYN + FSM_RESCUE_SEEK, held-out continuous wave-1:
wave mean 32.63 [30.0-35.3], median 34, max 69; P(w>=20)=.82 P(w>=25)=.64 P(w>=30)=.55
**P(w>=40)=.24** P(w>=50)=.15 P(w>=67)=.02; score mean 845,622 max 1,889,575;
score/wave 25,913; life-gen/wave 1.017 vs deaths/wave 1.078.
(For scale: yesterday's champion was 12.6 mean, all-time best game wave 29.)
LIFE-ECONOMY MODEL VALIDATED: expected length = start_lives/deficit = 2/0.061 = 33 waves ==
measured mean. PATH TO 100: cut deaths/wave by ~4% (1.078 -> ~1.04) => mean ~100; at the
current tail, an outright wave-100 game should occur ~1 per 100-200 attempts already.
(eval_protocol.py now self-heals wedges + EVAL_SEED_SKIP resume; books computed across both
segments, artifacts excluded.)
NEXT: deep-band death-rate cut = Stage-3 exact-model search (plumbing ready) and/or
forensics on champion-v2 deep-wave deaths (what kills it at 30-60 with correct labels).

## 2026-07-01 - Champion-v2 forensics: PROJECTILES now 55-57% of deaths -> frameskip-2 arm launched
deaths_champion_v2 (1595 deaths, 60 games, correct labels, nearest-suspect):
- waves 1-14:  Spark 36% + Shell 18% (+Cruise) | Hulk 9% Grunt ~7% (Mom/Dad = rescue-proximity artifact)
- waves 15-29: Spark 31% + Shell 19% + Cruise 9% | Hulk 11% Grunt 8%
- waves 30-60: Spark 20% + Shell 19% + Cruise 18% | Hulk 12% Grunt 9%
With income solved, the death profile FLIPPED from positioning (old: swarm 51%/hulk-pin 30%)
to PROJECTILES (~56%) — fast movers crossing the frameskip-4 blind window, but individually
predictable (sparks never re-home, shells straight+bounce, cruise re-aims <=8 ticks). Worst
waves = spawner waves 7/17/9/12.
LEVER: halve the blind window. VSEARCH_FRAMESKIP env added to search_value main +
clearance_planner DXY scales by frameskip/4 (player kinematics = 1px/frame/axis). fs2 arm
launched port 9960 N=100 seed-777 (logs/clear_fs2_n100.log, VSEARCH_STEPS=80000) vs the
rescueseek fs4 arm (24.27 mean, same seeds). NOTE: old "fs2 = no better" result was the
pre-label-fix reactive FSM; invalid for champion v2. If fs2 cuts projectile deaths ~in half,
deaths/wave 1.078 -> ~0.9 < life-gen 1.017 => UNBOUNDED GAMES (wave 100).

## 2026-07-02 - fs2 VERDICT: NEGATIVE (-2.88 @n=50, CI excl. 0) — killed early, fs4 confirmed
Frameskip-2 champion arm: 20.98 vs 23.86 (n=50, CI [-5.7,-0.05], W/L 18/29). Halving the
blind window also halves per-decision displacement — dodges execute slower per decision; net
loss, consistent across 50 seeds. fs4 stays. Killed at n=50 (CI already excluded zero at
n=40; core reassigned to the wave-100 hunt). Projectile-death lever must come from better
PREDICTION (Stage-3 exact model / bounce-aware avoidance already in ASMDYN) or positioning,
not control frequency.
OVERNIGHT BOARD: 8-instance hunt fleet (best 69/2.02M); 2x400k champion-v2 demo collectors
(teacher re-pass with corrected perception); BC->anchored-RL queued on GPU when demos land.

## 2026-07-02 - TEACHER RE-PASS VERDICT: NEGATIVE — cloning gap is FUNDAMENTAL, not perception
User asked whether teacher->student deserved a re-pass now that the decoder bugs are fixed.
Ran the clean experiment: 800k champion-v2 demos (planner-in-loop labels, corrected obs,
eps=0.08) -> BC (bc_champion_v2, val 0.50/0.56, triage 2.5 — normal BC-alone level) ->
anchored-RL with the EXACT combanchor recipe (warm from bcanchor_v2anchor_best 9.41, 3M
steps, 12 envs) -> triage x3 checkpoints (30 reseed games each):
  ck_2621k 6.1 / ck_2883k 6.3 / final 6.5  — ALL BELOW the 9.4 pin.
The anchor toward the champion's demos DEGRADED the 9.41 base: the champion's behavior
(clearance overrides + cross-board rescue treks) is a harder imitation manifold than the old
reactive FSM. With a 32.6-mean teacher and clean perception, distillation still loses ~26
waves to its teacher. CONCLUSION (final): no BC/RL path approaches the planner; the
deployable brain for the Xbox goal is the PLANNER ITSELF (YOLO -> entity list -> planner),
per project_goal_xbox_yolo. RL's last contingent role (noise-robustness fallback) is now
low-credibility. Compute redeployed to the wave-100 hunt.

## 2026-07-02 - ****** WAVE 100 ACHIEVED: WAVE 113, 3,433,800 pts, 45,581 steps ******
Gen-6 planner-re-evolution winner (models/fsm_evolved_planner_v2_final.json + clearance
planner ASMDYN H=6 DANGER=18 MARGIN=10 + FSM_RESCUE_SEEK), honest validation harness
(MAME_RL_RESEED, SEED_BASE=777, port 9946, continuous wave-1, no save states):
  game 13: wave=113 score=3433800 steps=45581
First wave-100+ game in project history — played all 40 unique waves + the 21-40 repeat
band ~3.6x, ~137 extra lives earned. Also all-time records: depth 77->113, score 2.28M->3.43M.
THE PATH THAT GOT HERE (one day): decoder label fix (12.6->22.7) -> rescue-seek life economy
(->24.3, income>25k/wave) -> planner-in-loop re-evolution of the 21 constants under the
DEPLOYED system w/ corrected labels (evolve_fsm EVOLVE_PLANNER=1, warm from combteacher,
STEPCAP 24k) -> gen-6 winner. Both evolved candidates still under N=100 validation
(gen-4 @9944: +4.54 CI[+0.5,+8.6] at n=50; gen-6 @9946: the 113 at game 13).
NEXT: finish both validations, promote the winner via official port-9970 books run, then
the deployment phase (Xenia YOLO dataset -> pixels-only loop -> real hardware) per
project_goal_xbox_yolo.

## 2026-07-02 - GEN-4 VALIDATION FINAL: +4.61 [CI +1.8,+7.4] — evolution DECISIVELY beats the incumbent
planner_v2 gen-4 constants, honest N=98 paired seeds vs incumbent champion (rescueseek arm):
28.53/med 29/max 70 vs 23.92/med 23/max 48; score 793k vs 610k (+30%); W/L 61/34. First
CI-excluding-zero winner of the whole campaign. Gen-6 (the wave-113 config) still validating
on 9946 — head-to-head decides the promotion, then official 9970 books run.

## 2026-07-02 - *** CHAMPION V3 PROMOTED: gen-6 evolved constants — mean 33.3 / median 32 / MAX 113 ***
GEN-6 FINAL (honest N~97 paired seed-777): vs incumbent +10.25 [CI +6.8,+13.7] 70W/23L,
score +55% (946k vs 611k); vs gen-4 +5.72 [CI +1.5,+9.9]. Absolute: mean 33.25 / median 32 /
max 113 (the wave-100+ game) / max score 3,433,800.
CHAMPION V3 RECIPE: models/fsm_evolved_planner_v2_final.json (gen-6 planner-in-loop
re-evolution) + clearance_planner (ASMDYN=1 DANGER=18 MARGIN=10 H=6) + FSM_RESCUE_SEEK=1.
Official port-9970 books run LAUNCHED (logs/official_champion_v3_9970.log).
One-day arc: 12.6 -> 22.7 (labels) -> 24.3 (rescue) -> 34.2 (re-evolution) -> WAVE 113.

## 2026-07-02 - *** CHAMPION V3 OFFICIAL BOOKS: mean 49.1 / median 39 / MAX 142 — GOAL DEMONSTRATED ***
Held-out port 9970, 94 valid continuous wave-1 games (1 wedge self-healed/excluded):
wave mean 49.13 [44.3-54.2], median 39, max 142; P(w>=25)=0.91; FOUR wave-100+ games
(107/124/132/142) = P(>=100) ~4%; score mean 1,309,134, max 4,041,225 (wave-142 game,
~161 extra lives); life-gen/wave 1.05 vs deaths/wave 1.09.
TWO-DAY ARC (official held-out means): 12.6 -> 22.7 (decoder labels) -> 32.6 (rescue-seek,
v2 books) -> 49.1 (planner-in-loop re-evolution, v3). Wave 1->100 is DEMONSTRATED and
REPRODUCIBLE (~1 in 23 games) on the real arcade ROM, continuous from boot, no save states.
DEPLOYABLE: models/fsm_evolved_planner_v2_final.json + mame_gym/clearance_planner.py
(ASMDYN=1 DANGER=18 MARGIN=10 H=6) + FSM_RESCUE_SEEK=1.
REMAINING (deployment phase, per project_goal_xbox_yolo): Xenia YOLO dataset -> pixels-only
loop -> latency compensation -> real Xbox hardware. Optional strength: another evolution
round (fitness gradient was still live at gen 8), Stage-3 exact-model search to push
P(>=100) higher.

## 2026-07-02 - Milestone committed (82bb3f1) + Xenia YOLO phase started
Committed the wave-100 milestone work (88 files; ROM/asm/save-states excluded per copyright).
XENIA PHASE (deployment, per project_goal_xbox_yolo) — recon + first drop into
~/win/code/robotron (the Windows Xenia toolkit; not a git repo):
- RECON: mature stack already exists — XeniaMemory (JIT-hook shared buffer),
  GameStateReader (30Hz entity reads), player.py vgamepad server (0.3s neutral timeout),
  screen_capture (PrintWindow), auto_labeler.py (synchronized PNG+JSON -> YOLO export,
  calibrated render-layer transform), reviewer.py (newest bounds calibration). Hard dep:
  custom xenia-canary build with C++ JIT hooks. Best prior brain reached wave 7.
- FOUND & FIXED: the Xenia label tables had the SAME pre-2026-07-01 bugs as MAME (phantom
  0x4800 tank, 0x4DF2-as-shell, 0x1F1F-as-missile, 0x4FEE shell anim in the quark set,
  cruise anim swallowed by the prog range, obsolete TS-near-Q dedup deleting real shells).
  Corrected per ENEMY_MODEL.md §2 in jit_entity_reader.py + game_state.py. CAVEAT: the C++
  hook (x64_sequences.cc kQuarkCmds) needs the same split + rebuild for the entity_list path.
- PORTED: brain_champion.py = the wave-142 champion on Xenia (GameStateReader -> 665x492
  pixel transform (identical to mame_obs) -> robotron_fsm+clearance_planner at 15Hz ->
  player.py sticks; per-tick nearest-match velocity tracker for the planner). Support files
  copied: robotron_fsm.py, clearance_planner.py, fsm_evolved_planner_v2_final.json.
- HANDOFF: ~/win/code/robotron/XENIA_CHAMPION_HANDOFF.md (run order, label verification
  checklist at waves 5/7+, calibration notes, open items). Windows-side execution (custom
  Xenia build + player.py + brain + auto_labeler) requires the user's Windows session.

## 2026-07-03 - ENDGAME HUNT fix launched (user-spotted wall-hiding stall)
User observation: with one enemy (+hulks) left, the champion runs to a wall and hides
until the enemy closes. Root cause confirmed in code + synthetic scenario: the idle
fallback only pursues CHASE_ENEMIES/family — a last Grunt/Brain/Tank beyond
CLOSE_FIRE_ENEMY (evolved 219px) is never approached OR fired at, while hulk-flee
(CLOSE_MOVE_HULK 62.5) herds the player wall-ward = the pinned posture the death
taxonomy flags, plus dead time while grunts speed up. FIX: HUNT_KILLABLE (robotron_fsm)
— idle fallback advances on the nearest killable (priority|regular) enemy to
HUNT_STANDOFF=90 and fires at it at any range; hulk/projectile flee still outranks it.
Settable via best_params json => models/fsm_evolved_planner_v2_hunt.json = gen-6
champion constants + HUNT_KILLABLE=1. Synthetic test: baseline move=STAY/no-fire vs
hunt move-toward+fire; hulk-close case unchanged (flee wins). GATE: official-books
replica run on held-out port 9970, N=100, same seed sequence as the 49.13 books
(logs/official_hunt_9970.log). Compare means (pairing unreliable for planner agents);
bar = 49.13 [44.3-54.2], P(w>=100) 4/94.

## 2026-07-03 - Overnight: evolution round 2 LAUNCHED (planner_v3, hunt genes added)
While the hunt A/B runs on 9970, launched the second planner-in-loop evolution round
(gradient was still live when planner_v2 stopped at gen 8). evolve_fsm.py now has 23
genes: the 21 from v2 + HUNT_KILLABLE (0/1 toggle, starts ON, evolution can disable)
+ HUNT_STANDOFF (40-220, seed 90). Because HUNT_* land in best_params, the winner
json is self-contained (no env flag needed at eval/deploy). Run: gens=10 lambda=24
mu=8 K=5 workers=10 ports 9920+, EVOLVE_STEPCAP=30000 (24k saturated at ~wave 75;
30k gives headroom to ~wave 90), EVOLVE_PLANNER=1 FSM_RESCUE_SEEK=1 MAME_RL_RESEED=1,
warm from fsm_evolved_planner_v2_final.json. Log: logs/evolve_planner_v3.log.
PROMOTION PATH: winner must beat gen-6 paired N>=50 on a training port, then official
9970 books. Windows sync + windows-agent notification required on promotion (user).

## 2026-07-03 - HUNT INTERIM (games 1-10): 8/10 wave-100+, THREE games at the wave-255 ceiling
Hunt-variant official replica, first 10 games: 133/255/119/171/255/36/255/77/159/117
(baseline books: 4 wave-100+ in 94). Score cross-validation: 9/10 games sit at the
champion's stable 26-28k/wave (the 255s carry ~7.2M = a true ~wave-265 game), so the
waves are REAL, not decode glitches. KEY MECHANICS CONFIRMED IN ASM (robomame.asm
$2A94): wave byte is u8; clearing wave 255 wraps INC->0->INC = back to WAVE 1 (plus a
bonus life at $2A9A) — deep games roll over into the easy band and continue. So the
wave metric right-censors at 255 and the LIFE ECONOMY HAS FLIPPED POSITIVE: the
endgame-hunt fix removed enough stall/pin deaths to make games effectively unbounded
until a variance spike. ANOMALY: game 4 score 41.2M at max-wave 171 (241k/wave) is
inconsistent (a >=1-wrap game must latch max=255); its score will be excluded from
books; wave kept. Watch for repeat anomalies at larger N.

## 2026-07-03 - Windows/Xenia sync: hunt champion delivered + windows agent notified
Interim evidence (11 games, 8 wave-100+) judged strong enough to sync early — the
Xenia YOLO dataset benefits immediately from deeper play (their brain loop peaked at
W27). Synced to ~/win/code/robotron: robotron_fsm.py (hunt code), NEW
fsm_evolved_planner_v2_hunt.json; brain_champion.py now prefers the _hunt json when
present (delete-to-fallback). Behaviorally verified from WSL (hunt scenario returns
move=3 fire=3 with the synced files). Notified the windows agent per user instruction:
update block at top of XENIA_CHAMPION_HANDOFF.md + memory file
project_champion_hunt_upgrade.md + MEMORY.md pointer (notes the 24/7 loop must be
RESTARTED to pick up the json preference). Official promotion still gated on the full
9970 books + planner_v3 evolution outcome.

## 2026-07-03 - Evolution restarted as planner_v3b: lives-margin fitness (cap saturation fix)
planner_v3 gen 1 confirmed the predicted saturation: best candidate dist
[80,80,81,81,81] = ALL 5 games at the 30k step cap (~wave 80) — with hunt-era
candidates the wave metric has no gradient left at the top. Fix (no wall-clock cost):
worker now returns lives-at-exit (env reports 0 on game over; banked lives if the cap
is reached alive) and fitness = mean_wave + 0.1*mean(lives_end) + score/1e6. Banked
lives at the cap = the life-economy margin = exactly the quantity that predicts how
far past the cap a candidate would run. Below the cap wave still dominates (dead
candidates have lives_end 0). Relaunched warm from the gen-1 best as planner_v3b
(pid 15540, ports 9920+, log logs/evolve_planner_v3b.log). Cleanup note: killing the
v3 run orphaned its spawn workers (mp children reparented to 1) — killed workers then
force-killed their MAMEs; eval MAME on 9970 untouched (verified only pid 4897 left).

## 2026-07-03 - LATENCY LAB launched (Xenia gap reproduction + extrapolation fix)
Xenia agent's report (via shared memory): hunt does NOT transfer to Xenia (still
W9-27; deaths/wave ~1.7-1.8 both eras vs MAME ~1.09) — bottleneck is the perception
path: entity positions arrive via a ~250ms accumulator (≈4 decision steps @15Hz)
while player pos reads fresh. Built mame_gym/latency_lab.py to reproduce EXACTLY that
mix on MAME (entities from packet t-K, player from t) and test the fix: linear
forward-extrapolation of each entity by its tracked per-step velocity x lag (comp
mode; clamped to board). Death counting is flicker-proof (start + score//25000 -
remaining; the $BDEC 1->2->1 transition flicker poisons per-step drop counting).
Arms: base K=0 + delay K=4 (port 9960), comp K=4 + delay K=2 (port 9962), N=20 each,
cap 15000, hunt params, reseed. HYPOTHESES: delay-K4 deaths/wave should land near
Xenia's ~1.7-1.8 (validating the diagnosis); comp-K4 should recover most of it.
If comp works -> ship extrapolation into the Xenia brain (its VelocityTracker
already computes velocities) + windows-agent note.

## 2026-07-03 - LATENCY LAB ROUND 1: Xenia gap FULLY EXPLAINED by perception lag; extrapolation = partial fix
N=20/arm, cap 15000, hunt params, reseed (ports 9960/9962):
  base  K=0: wave mean 39.05 (cap-censored)  deaths/wave 0.844
  delay K=2: 12.45  d/w 0.980   | delay K=4 (~267ms ≈ Xenia): 6.90  d/w 0.971
  comp  K=4:  9.60  d/w 0.844   (+39% waves vs delay-K4; restores BASE death rate)
VERDICT 1: mixed-staleness lag (entities t-K, player fresh) alone collapses the
champion into exactly Xenia's observed W9-27 band -> the transfer gap is ~entirely
perception latency, not planner strategy or game differences.
VERDICT 2: linear velocity extrapolation recovers the per-wave death rate but only
part of the depth (9.6 vs 39) — the residual is events INSIDE the lag window
(new spawns/shots invisible for K steps; matches Xenia's wave-start spawn-in
deaths). No extrapolation can see those: the REAL fix on Xenia is shrinking the
accumulator window (C++ side), with extrapolation as the multiplier on top.
ROUND 2 RUNNING: comp K=2 / comp K=1 (value at realistic shorter lags), delay K=1
(what a shorter accumulator alone buys), comp K=4 + FSM_BUFFER_SCALE=1.3 (margin
scaling stacked on extrapolation).

## 2026-07-03 - *** HUNT OFFICIAL BOOKS: mean 173.7 / median 180.5 / 76% of games reach wave 100+ ***
Held-out port 9970, N=100 (90 valid, 10 wedge-invalidated — wedge rate elevated by
parallel lab load, all self-healed): wave mean 173.73 [157.1-190.4], median 180.5,
min 20, MAX = the 255 wire ceiling in 40/90 games (44%). P(w>=25)=0.99, P(w>=20)=1.00.
**P(w>=100) = 68/90 = 76% (baseline books: 4.3%).** Score mean 5,212,117; score/wave
30,001 (up from 25.9k — hunting also speeds waves); life-gen/wave 1.20 vs deaths/wave
1.21 (parity at 3x the throughput). Note: the reported max score 41,242,365 is game 4,
whose score is inconsistent with its wave-171 latch (a >=1-wrap game must latch 255);
treat as artifact — next-best max 7,336,225 is trustworthy. THE WAVE 1->100 GOAL IS
NOW THE TYPICAL GAME, not the tail. CHAMPION RECIPE v4 = v3 + HUNT_KILLABLE=1/
HUNT_STANDOFF=90 (models/fsm_evolved_planner_v2_hunt.json).

## 2026-07-03 - LATENCY LAB ROUND 2: extrapolation recovers 90% at K=1; buffer scaling doesn't stack
comp K=1: 35.05 (= 90% of base 39.05!) | comp K=2: 20.74 (delay-K2 12.45)
delay K=1: 18.31 | comp K=4 + FSM_BUFFER_SCALE=1.3: 9.30 (vs comp-K4 9.60 -> WASH)
CURVE (wave mean): K=0 39.1 | K=1 18.3->35.1 comp | K=2 12.5->20.7 | K=4 6.9->9.6.
XENIA PRESCRIPTION: (1) ship velocity extrapolation now (pure win at any lag);
(2) the dominant lever is shrinking the 250ms accumulator window — at ~66ms (K=1)
plus extrapolation, Xenia would sit at ~90% of the MAME ceiling, which post-hunt
means WAVE-100+ GAMES ON XENIA. Margin scaling (buffer 1.3) does not stack; skip.

## 2026-07-03 - Evolution planner_v3b DONE: gen-3 winner fit 94.4, kept hunt ON
10 gens complete. All-time best from gen 3: best_wave 89.6 at 30k cap, lives_end
22.4, fit 94.40 (warm-start reference: gen-1-of-v3 fit 81.48 under old fitness).
Gens 4-10 never beat it (top8_mean drifted 64->83; sigma annealed to 0.065).
Winner kept HUNT_KILLABLE=1, HUNT_STANDOFF->84.6, tightened ADJACENT->38.6,
CLOSE_MOVE_HULK->42.6. models/fsm_evolved_planner_v3b.json. VALIDATION LAUNCHING:
books-replica on 9970 (same seeds as both prior books) — score mean is the primary
comparator (wave mean is ceiling-censored at 255 in 44% of hunt-books games).
