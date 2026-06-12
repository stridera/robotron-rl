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
