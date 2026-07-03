# Robotron 2084 — Ground-Truth Enemy Model

Source of truth: the annotated Williams 6809 disassembly `mame_gym/robomame.asm`
(~21,183 lines). Every claim below is grounded in `robomame.asm:<line>`. This
document exists so we can validate our memory decoder (`robotron_server.lua` +
`mame_obs.py`) against how the ROM *actually* spawns, moves, and kills entities,
and so FSM work is grounded in real AI rather than guesses.

> **Why this doc exists (2026-07-01):** we discovered our decoder was labeling
> **tank shells as Quarks** and dropping **Tanks entirely** (wrong state-word
> table). This manufactured the long-standing "quarks teleport" belief that drove
> the FSM design and the "wave-100 is an unavoidable teleport wall" ceiling
> conclusion. Nothing in this game teleports. See §7 (Decoder mismatches).

---

## 1. Object pool & slot model

- **Object pool:** 180 records of **24 bytes ($18)** each, `$9900 .. $A9C8`.
  Built by `INITIALISE_ALL_OBJECT_LISTS` `$D7A5` (`robomame.asm:16479-16487`).
- **Free list:** head at `$981B` (ZP `$1B`). A slot is *free* iff it is threaded
  on this list. Allocation (`RESERVE_OBJECT_IN_LINKED_LIST` `$D28F`,
  `robomame.asm:15497`) pops the head; free (`FREE_OBJECT` `$D2A7`,
  `robomame.asm:15518`) pushes it straight back. **No generation counter, no
  in-use flag** — a freed slot is handed back out immediately at the same address.
- **Player object** is fixed at `$985A`, outside the pool (`robomame.asm:81`).

### Object record layout (offset from record pointer)

| Off | Field |
|-----|-------|
| `$00-01` | next-pointer in linked list (also free-list next) |
| `$02-03` | current animation-frame metadata pointer |
| `$04-05` | **blitter dest = on-screen position** (A=`$04` screen-X col, B=`$05` screen-Y line) |
| `$06-07` | pointer to this object's metadata/AI record |
| `$08-09` | **collision / death-handler address ("state word") — our type discriminator** |
| `$0A-0B` | X coordinate whole.frac (8.8) |
| `$0C-0D` | Y coordinate whole.frac (8.8; often just the `$0C` byte) |
| `$0E-0F` | X-velocity delta (8.8 signed) |
| `$10-11` | Y-velocity delta (8.8 signed) |
| `$14-15` | previous animation-frame metadata pointer |
| `$16-17` | (cruise missile) collision-mask metadata pointer |

Cruise missiles are a quirk: they pack X/Y at `$0A`/`$0B` (`robomame.asm:3091,3102`).

### The four entity linked lists (what our decoder walks)

| Head | Contents |
|------|----------|
| `$9817` | spheroids, enforcers, quarks, **sparks**, **tank shells** |
| `$981F` | family members (Mikey / Mommy / Daddy) |
| `$9821` | grunts, hulks, brains, progs, **cruise missiles**, **tanks** |
| `$9823` | electrodes |

Liveness is structural: an entity is live iff reachable from a list head via the
`$00` next-pointer. Dead objects are unlinked (`robomame.asm:15518-15526`), so the
walk yields only live entities. A NULL blitter dest (`$04`=0) also means "not live".

**Not walked** (invisible to us, mostly correctly): player lasers (metadata list
`$9813`, never in the 4 entity lists), explosions (`$98A9`), metadata/task/free
lists (`$9811/$9813/$9815/$981B/$981D`).

### Spawn-slot saturation (the classic exploit)

Spawning is gated by (a) per-type on-screen caps and (b) a global free-slot check
(`LDX $1B / BEQ fail` inside `CREATE_ENFORCER_QUARK_SPARK_SHELL` `$D32B`,
`robomame.asm:15668`). Fill the pool / hit the caps and spawners silently bail.
Caps: **enforcers 8**, **cruise missiles 8**, **sparks 20**, **tanks 20**.

---

## 2. State-word → type table (GROUND TRUTH)

The "state word" is the collision-handler pointer at record `$08-09`, set by each
constructor. This is the correct, ROM-verified mapping:

| State word | Type | List | Evidence |
|------------|------|------|----------|
| `$3A76` | Grunt | `$9821` | GRUNT_COLLISION_HANDLER |
| `$00B6` | Hulk **or** standalone Prog | `$9821` | shared — disambiguate by motion |
| `$1DD6` | Brain | `$9821` | `BRAIN_COLLISION_HANDLER` (asm:2729) |
| `$2119` | **Cruise Missile** | `$9821` | `CRUISE_MISSILE_COLLISION_HANDLER` (asm:3175); anim `$206B` (asm:3054) — NOT a brain |
| `$4DF2` | **Tank** | `$9821` | `robomame.asm:7388,7427` "tank collision handler" |
| `$1F1F` | **Prog** (brain-mutated human) | `$9821` | anim in human range (validated); the real cruise handler is `$2119` |
| `$12C8` | Spheroid | `$9817` | SPHEROID_COLLISION_HANDLER |
| `$1483` | Enforcer | `$9817` | ENFORCER_COLLISION_HANDLER |
| `$14DC` | Enforcer Spark | `$9817` | `CREATE_SPARK` passes `$14DC` (`robomame.asm:1990-1994`) |
| `$4BC9` | **Quark** | `$9817` | `QUARK_COLLISION_HANDLER` (`robomame.asm:7252`) |
| `$4FD5` | **Tank Shell** | `$9817` | `SHELL_COLLISION_HANDLER` (`robomame.asm:7863`) |
| `$3AA9` | Electrode | `$9823` | (list 4 forces Electrode regardless) |
| `$0330/$0335/$033A` | Mikey / Mommy / Daddy | `$981F` | family |

**Ambiguous handler** (shared address, must disambiguate by motion/context):
- `$00B6` = **Hulk** (indestructible) *or* **standalone Prog** (shootable). Inverting
  these inverts correct play. Prog moves in the ~(8,20] px/step band; Hulk drifts slower.

There is **no such thing as three Quark state words.** Only `$4BC9` is a Quark.

### Animation-pointer validation (2026-07-01, `validate_entity_labels.py`)

The animation-frame pointer at object `$02` (the sprite the hardware actually blits)
is an identity source **independent** of the `$08` collision-handler SW. Running the
champion FSM to waves 17-25 with `MAME_EMIT_ANIM=1`, each SW maps 100% to a distinct,
tight anim cluster — confirming the corrected table and catching the brain/cruise mixup:

| SW (label) | anim cluster | verdict |
|------------|--------------|---------|
| `$4BC9` Quark | `$50Cx` | distinct sprite ✓ |
| `$4DF2` Tank | `$502x` | distinct sprite ✓ |
| `$4FD5` TankShell | `$4FEE` | matches ASM shell anim block ✓ |
| `$14DC` Spark | `$1A3x` | matches ASM spark anim ✓ |
| `$2119` "Brain (alt)" | `$206B` | = **cruise-missile** anim → it's a Cruise Missile, not a brain |
| `$1F1F` "Cruise Missile" | human-sprite range | → it's a **Prog**, not a cruise |

Measured per-step velocities also respect the ASM bounds (Quark med 3 / max 10 u/step;
Tank med 1; TankShell med 4 / p95 11 — projectile-fast, as expected). Node-address
recycling is real and frequent (641 SW-changes-at-a-reused-node + 1032 >25 u/step fake
jumps over 6 games, mostly Hulks/Electrodes as slots free on death) — velocity keyed on
raw address needs a generation guard (§7 item 4).

---

## 3. Spawn & death, per type

| Type | Origin | Count / cap | Death → score |
|------|--------|-------------|---------------|
| **Grunt** | wave-start (`INITIALISE_ALL_GRUNTS` `robomame.asm:5544`) | wave table `$2E24` | bullet → **100**; also dies on electrodes |
| **Hulk** | wave-start (`robomame.asm:602`) | wave table `$2EEC` | **INDESTRUCTIBLE** — bullets only shove it (`robomame.asm:471`) |
| **Brain** | wave-start (`robomame.asm:2296`) | wave table `$2F14` (every 5th) | bullet → **500**; spawns Progs & Cruise Missiles |
| **Spheroid** | wave-start (`robomame.asm:1564`) | wave table `$2F3C` | shot → **1000**; or self-exits after dropping enforcers |
| **Enforcer** | **mid-level**, from Spheroid (`CREATE_ENFORCER` `robomame.asm:1838`) | cap **8** | bullet → **150** |
| **Quark** | wave-start (`robomame.asm:7163`) | wave table `$2F64` | shot → **1000**; or self-exits after dropping tanks |
| **Tank** | **mid-level**, from Quark (`SPAWN_TANK` `robomame.asm:7374`); nominal wave-start init exists but no count table | cap **20** | bullet → **200** |
| **Electrode** | wave-start (`robomame.asm:5666`), stationary | wave table `$2E4C` | bullet → **0** (no score); hulks stomp them |
| **Prog** | **mid-level**, from Brain programming a human (`CREATE_PROG` `robomame.asm:2764`) | slot-limited | bullet → **100** |
| **Cruise Missile** | **mid-level**, from Brain (`robomame.asm:3035`) | cap **8** | bullet → **25** |
| **Spark** | **mid-level**, from Enforcer (`CREATE_SPARK` `robomame.asm:1978`) | cap **20**, lifespan 20-35 | on hit/expire freed |
| **Tank Shell** | **mid-level**, from Tank (`CREATE_TANK_SHELL` `robomame.asm:7608`) | cap **20**, lifespan 48-79 | on hit/expire freed |
| **Family** | wave-start (`robomame.asm:862`) | wave tables | rescue **1000→5000** (capped); or killed/progged |

**Placed once, can only disappear (never re-created, slot stable):** Grunts, Hulks,
Electrodes, Brains, Family. Spheroids and Quarks are also placed once but may
self-exit after spawning. **Dynamically (re)created mid-wave:** Enforcers, Tanks,
Progs, Cruise Missiles, Sparks, Tank Shells.

**Key consequence:** the "quark waves" (7, 12, 17, 22) are really **quark→tank→shell**
waves. Quarks spawn up to 20 tanks; tanks fire up to 20 bouncing shells. The lethal,
fast-moving things on those waves are the **shells and tanks**, not the quarks.

---

## 4. Movement / AI, per type

Two move engines:
- **Global integrator** `$DCFF/$DD90` (`robomame.asm:17302`): 8.8 `pos += delta`
  with clamp-by-skip at borders (no reflection). Used by list `$9817`:
  spheroids, enforcers, sparks, quarks, tank shells.
- **Inline per-type routines** for list `$9821` (grunts, hulks, brains, progs,
  cruise missiles, tanks) — each moves itself; cruise missiles & tank shells
  reflect off walls (`NEG`/`COM` the delta).

| Type | Targeting | Speed (approx) | Re-decide cadence |
|------|-----------|----------------|-------------------|
| Grunt | homes on player, 4px axis steps | rate ∝ `$BE5C` (speeds up as grunts die) | countdown, AI every 4f |
| Hulk | 4-dir drift toward family/player; indestructible | ~2-4 px/update | ctr 1-32, AI every `$BE61`f |
| Brain | homes on nearest human (→prog), else player | 1 px/tick | every tick, AI every `$BE63`f |
| Spheroid | random curved glide, no homing | **≤1 px/f X, ≤2 px/f Y** (hard clamp) | ctr, AI every 2f |
| Enforcer | dives at player, velocity ∝ distance | fastest, accelerating | ctr 0-31, AI every 3f |
| Spark | ballistic toward player at launch + curvature | ballistic | none (constant curve), AI 4f |
| **Quark** | **random wander, bounces off borders, NO homing** | **≤~1 px/f X, ≤~2 px/f Y** | ctr 1-32, AI every 3f |
| Tank | mixed random/player, 4/8-dir | ~0.5 px/f X, slow | ctr 1-32, AI every `$98EF`f |
| Tank Shell | aimed straight shot, **reflects off all walls** | straight & fast | on wall hit, move every 2f |
| Cruise Missile | homes on player, bounces off walls | ~2 px/f | re-aim ctr 0-7, move ×2 every 2f |
| Prog | runs toward random point near player, 4-dir only | 4px steps | random, AI every 3f |

**Quark speed ceiling (the teleport-refutation math):** quark delta = `signed_rand × 4`
(X) / `× 8` (Y) in 8.8 fixed point, `rand ≤ $BE67` (max table value `$44`=68)
(`CHANGE_QUARK_DIRECTION` `robomame.asm:7201-7244`). That's **≤ ~1.1 px/frame X,
~2.1 px/frame Y** → at frameskip 4, **≤ ~9.5px diagonal per decision step.** A quark
physically cannot move >40px in a step. Any observed ">40px quark jump" is a decoder
artifact, not game behavior.

---

## 5. Projectiles — where they live

| Projectile | Storage | Cap | Identify by |
|------------|---------|-----|-------------|
| Player laser | metadata list `$9813` (NOT in the 4 entity lists) | 4 | not walked by us |
| Enforcer spark | object in `$9817` list | 20 | SW `$14DC` / anim `$1A34-40` |
| Tank shell | object in `$9817` list | 20 | **SW `$4FD5`** / anim `$4FEE` block |
| Cruise missile | object in `$9821` list (mixed w/ enemies) | 8 | SW `$2119` / anim `$206B` |

---

## 6. Coordinates & sanity bounds

- Playfield: X ∈ [`$07`, ~`$8F`], Y ∈ [`$18`, ~`$EB`] (`robomame.asm:452-464`).
- X whole-part is in **bytes (2px each)**; Y whole-part in **pixels**.
- Decoded per-frame velocity for `$9817` movers should be **≤ ~2.0 px/frame**
  (spheroid/quark clamp) or accelerating-but-bounded (enforcer). A larger delta ⇒
  slot/identity aliasing, not real motion.

---

## 7. Decoder mismatches (2026-07-01)

Cross-checking §2 against `mame_obs.py` (`_LIST1_SW` / `_LIST3_SW` / `ENTITY_TYPES`):

1. **[FIXED]** **Tank shells labeled "Quark"** — mapped `0x4FD5 → "Quark"`, but `$4FD5`
   is `SHELL_COLLISION_HANDLER`. **Root cause of the "quarks teleport / quark = 32%
   of deaths" phantom.** Now `0x4FD5 → "TankShell"`. Re-classifying the champion's
   665-death log: Quark killers 27.3% → **0.3%**; the old "Quark" deaths were really
   57% TankShell + 41.5% Tank.
2. **[FIXED]** **Tanks invisible** — Tank was mapped to phantom `0x4800` (nothing
   loads `#$4800` in the ROM); real tank SW `$4DF2` was absent → tanks dropped. Now
   `0x4DF2 → "Tank"`.
3. **[FIXED]** **Fabricated Quark SWs** — only `$4BC9` is a Quark; `0x4DF2`/`0x4FD5`
   were wrongly listed as Quark variants. Removed.
4. **[FIXED] Brain / Cruise / Prog mixup** (found by anim validation) — `$2119` is
   the cruise-missile handler (asm:3175, anim `$206B`) but was labeled "Brain (alt)"
   → cruise missiles counted as brains (e.g. the "Brain (alt) = 40 deaths at wave 10"
   were cruise missiles). And `$1F1F`, labeled "Cruise Missile", has human-range anim
   → it's actually a **Prog**. Fixed in `mame_obs.py`: `$2119 → CruiseMissile`,
   `$1F1F → Prog`, `$1DD6 → Brain` (ENTITY_TYPES + _LIST3_SW).
5. **[FIXED] Node-address identity has no generation guard** — slots recycle
   immediately; velocity keyed on raw address aliased dead/new entities (validated
   live: 641 recycle + 1032 fake-jump events over 6 games; ~5% of deaths). Fixed:
   velocity now keyed on `(addr, sw)` + a `_VMAX_PX=130` clamp; obs velocity capped
   at 123px (was 300px+). Affects RL obs only — the FSM uses positions, not velocity.
6. **[FIXED 2026-07-01] Cross-list truncation** — `walk_lists` shared one `n < 120`
   counter across all four lists; a flood on list 1 could starve electrodes/family.
   Fixed: `MAX_ENTITIES = 190` (> the 180-record pool, so truncation is impossible;
   wire format is a single n byte, safe to 255).

## 8. Combat / escape mechanics (for the FSM)

Collision handlers dispatch on `$9848` (`player_collision_detection`): `1` = player
contact (you die), `0` = laser hit. Verified handlers:

- **Everything is destructible by the laser EXCEPT the Hulk.** Grunt/Enforcer/Tank/
  Brain/Spheroid/Quark/Prog/Electrode die on a laser hit. **Sparks (`$14DC`, asm:14E0)
  and tank shells (`$4FD5`, asm:4FD9) also die to the laser** (+25 pts each) — so
  shooting an incoming projectile is a valid escape.
- **Hulks cannot be killed but CAN be pushed.** `HULK_BULLET_COLLISION_HANDLER`
  (asm:471-496) adds the laser's velocity delta to the hulk's position (shoved
  ~6-12px in the shot direction, random-doubled). So firing at a hulk that's
  pinning you shoves it away — a real escape. (Explains why the old "don't fire at
  hulks" experiment regressed: it removed this push.)

### Death taxonomy (1167 deaths, champion FSM, corrected labels, `analyze_escape.py`)

`swarm 51%` (≥3 lethal near — can't shoot+dodge all) · `hulk-pin 30%` ·
`shot-but-too-late 9%` · `decode-artifact 5%` (now fixed) · `dodge-avoidable 4%` ·
`shoot-avoidable 1%`. **~81% are positioning failures** (swarm + hulk-pin), only ~5%
single-move-avoidable — the reactive ceiling is near-saturated; the real lever is
anticipation/spacing (or hulk-push to break pins).

## 9. Deep-dive AI findings (2026-07-01, full ASM read)

Per-type mechanics beyond §4, all bot-relevant:

- **RNG (`GENERATE_RANDOM_NUMBER $D6CD`, asm:16346-16364):** 3-byte state
  `$9884/$9885/$9886`, seeded once (asm:15222). Advances ONLY inside the three
  consumer wrappers (`$D6AC` multiply, `$D6B6` bounded, `$D6C8` A+B); the many
  direct `TST/LDA $84/$85/$86` reads (hulk push doubling, quark border checks,
  tank mode pick, spheroid curvature) do **not** mutate state. Fully deterministic
  and readable → an exact forward model can track RNG phase.
- **Tank shells have TWO firing modes** (asm:7631-7757): rand≥$80 → direct-lead at
  player blit ±16px jitter blended by accuracy `$BE65`; rand<$80 → deliberate
  **wall-bank shot** aimed at the far wall to ricochet into the player. Reflection
  uses `COM` (one's-complement), not `NEG` (asm:7846-7857) → bounces drift slightly
  (approximate mirror). Lifespan (rand&$1F)+48 frames.
- **Shell counter bug (`$F1`, asm:7830-7836):** decremented on laser/player hit but
  NOT on natural expiry. After ~20 shells have been fired and left to fizzle in a
  wave, the game believes 20 are live and **tanks stop firing for the rest of the
  wave**. Exploit: DODGE shells (don't shoot them) to permanently exhaust the
  wave's shell budget; shooting a shell frees capacity for another.
- **Enforcer dive velocity ∝ distance** (asm:1933-1964): slow when close, fastest
  when far → hug-and-perpendicular-dodge beats fleeing. Re-aims only every
  rand&$1F ticks; straight-line-predictable between decisions.
- **Sparks never re-home** after launch (fixed curvature added per tick,
  asm:2082-2087) — the whole arc is deterministic at creation. Lifespan
  (rand&$0F)+20.
- **Cruise missiles re-aim only every ≤8 ticks** and lead by just ±6px
  (asm:3082-3118); they fly straight between re-aims → sidestep after they commit.
- **Brains ignore the player while any human lives** (asm:2466-2475) and at wave
  start (empty family list) all brains target Mikey `$B354` (asm:2394-2405) —
  predictable clumping.
- **Hulk null-target bug** (asm:532-538): with no live family target the hulk
  reads garbage (`$7E01`) → wanders/corner-sticks. Hulk-pin pressure drops once
  the family question is resolved.
- **Grunt speed is player-influenced** (asm:5843-5849): `$BE5C` (move delay)
  decays toward floor `$BE5D` as grunts die — each kill speeds up survivors.
- **Player:** 1px/frame per axis (diagonal 1px both), max 4 lasers on screen
  (`$87`, asm:4749-4751), laser 6px/frame, must hold a fire direction ≥2 ticks
  before a shot (asm:4741-4747). Hitbox ~8×12px (asm:5420).
- **Difficulty saturates:** every `$BE5C-$BE67` table plateaus by the mid-30s
  band and waves 41+ reuse entries 21-40 (asm:4030-4033) → max difficulty is
  bounded; wave 100 is "the 21-40 band, 4 more times."
- **Wave-clear set** (`COUNT_ENEMIES_ON_SCREEN $2A73`, asm:3834-3841): only
  grunts+spheroids+enforcers+brains+tanks+quarks must hit zero. Hulks, electrodes,
  progs, cruise missiles never block a wave. (The `INC lives` at wave-clear
  asm:3854 pre-compensates the `DEC` in the shared respawn path asm:3464 — no
  free life per wave.)
- **Wave-start safe rectangle** (asm:5586-5637): spawn positions re-roll inside a
  player-centred exclusion rect; shrinks per wave but freezes at the wave-6 size
  from wave 10 → guaranteed opening clear zone every wave.
