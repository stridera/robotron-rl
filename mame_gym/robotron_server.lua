-- robotron_server.lua — MAME-side RL server for Robotron 2084 (arcade ROM).
--
-- MAME runs this as an -autoboot_script. It opens a TCP listen socket
-- (emu.file "socket.host:port" listens), boots the game to gameplay, then
-- runs a synchronous step loop: block for an action from Python, apply it to
-- the input fields, advance `frameskip` frames, send back the observation.
--
-- Protocol (little-endian, fixed-size):
--   Python -> MAME : 4 bytes [cmd, move_dir(0-8), fire_dir(0-8), frameskip]
--                    cmd: 0=step, 1=reset(load state), 2=quit
--   MAME -> Python : OBS_LEN bytes (see pack_obs)
--
-- Memory map (6809 space, identical arcade==XBLA, verified 2026-06-09):
--   $BDED wave, $BDEC lives, $BDE5-$BDE7 score(BCD), $9864/$9866 player X/Y,
--   $98D4 slot pool (101 slots * 24 bytes).

local PORT = os.getenv("MAME_RL_PORT") or "8123"
local FRAMESKIP_DEFAULT = 4
-- Optional RNG reseed (opt-in MAME_RL_RESEED=1). Robotron's RNG state is the
-- 3-byte LFSR at $9884-$9886 (robomame.asm $D6CD: "$84, DP=$98 -> $9884"). The
-- boot presses start at a fixed frame, freezing it in rl_reset -> identical games.
-- Poke a fresh per-reset value (decorrelated per instance by PORT) so enemy RNG
-- diverges -> varied games at save-state speed. OFF by default.
local RESEED = (os.getenv("MAME_RL_RESEED") == "1")
-- Seed base: port-derived by default (each port = its own game sequence). Set
-- MAME_RL_SEED_BASE to a fixed value so SEPARATE runs (e.g. an A/B on different
-- ports) face the IDENTICAL game sequence -> true PAIRED comparison (much lower
-- variance than the per-port-random sequences). Added 2026-06-29.
local rng_seed = tonumber(os.getenv("MAME_RL_SEED_BASE")) or ((tonumber(PORT) or 8123) * 2749 + 1)
local pending_reseed = false

local sock, opened = nil, false
local mem, F = nil, {}
local state = "boot"
local f = 0
local recv_buf = ""
local frames_to_run = 0

local function field(t, n) return manager.machine.ioport.ports[t].fields[n] end

local function setup()
    mem = manager.machine.devices[":maincpu"].spaces["program"]
    F.coin  = field(":IN2", "Coin 1")
    F.start = field(":IN0", "1 Player Start")
    F.mu = field(":IN0", "Move Up");   F.md = field(":IN0", "Move Down")
    F.ml = field(":IN0", "Move Left"); F.mr = field(":IN0", "Move Right")
    F.fu = field(":IN0", "Fire Up");   F.fd = field(":IN0", "Fire Down")
    F.fl = field(":IN1", "Fire Left"); F.fr = field(":IN1", "Fire Right")
end

-- 9-direction -> {up, down, left, right}. 0=none,1=N,2=NE,3=E,4=SE,5=S,6=SW,7=W,8=NW
local DIR = {
    [0]={0,0,0,0}, [1]={1,0,0,0}, [2]={1,0,0,1}, [3]={0,0,0,1}, [4]={0,1,0,1},
    [5]={0,1,0,0}, [6]={0,1,1,0}, [7]={0,0,1,0}, [8]={1,0,1,0},
}

local function apply_action(move, fire)
    local m, fr = DIR[move] or DIR[0], DIR[fire] or DIR[0]
    F.mu:set_value(m[1]); F.md:set_value(m[2]); F.ml:set_value(m[3]); F.mr:set_value(m[4])
    F.fu:set_value(fr[1]); F.fd:set_value(fr[2]); F.fl:set_value(fr[3]); F.fr:set_value(fr[4])
end

local function read_n(n)
    -- Spin-read inside the frame callback. Blocks MAME's loop (frame doesn't
    -- advance until callback returns), so this is synchronous with Python.
    while #recv_buf < n do
        local chunk = sock:read(n - #recv_buf)
        if chunk and #chunk > 0 then recv_buf = recv_buf .. chunk end
    end
    local out = recv_buf:sub(1, n)
    recv_buf = recv_buf:sub(n + 1)
    return out
end

-- Observation packet layout:
--   [wave, lives, sc5, sc6, sc7, pX, pXsub, pY, dir, dead]  (10 bytes)
--   2424 bytes of the slot pool starting at $98D4 (kept for death forensics
--     and kill counters)
--   [n_entities] (1 byte), then n * 7-byte typed records:
--     [addr_hi, addr_lo, list_id, sw_hi, sw_lo, x, y]
--   from walking the game's own per-category linked lists (robomame.asm):
--     list 1 = $9817 spheroids/enforcers/quarks/sparks/tankshells
--     list 2 = $981F family members
--     list 3 = $9821 grunts/hulks/brains/progs/cruise missiles/tanks
--     list 4 = $9823 electrodes
--   Node layout: +0/1 next ptr, +4 display X, +5 display Y, +8/9 state word.
--   Dead objects unlink ⇒ the walk yields only LIVE entities by construction.
local SLOT_BASE, SLOT_BYTES = 0x98D4, 101 * 24
local LIST_HEADS = { [1] = 0x9817, [2] = 0x981F, [3] = 0x9821, [4] = 0x9823 }
-- Cap must exceed the object pool size (180 records, ENEMY_MODEL.md §1) so a
-- flooded list 1 (20 shells + 20 sparks + 8 enforcers + tanks/quarks) can never
-- starve later lists (family/electrodes) out of the shared-counter walk.
-- Wire format is a single n byte, so anything ≤255 is protocol-safe.
local MAX_ENTITIES = 190
-- Validation only: also emit each entity's animation-frame pointer (node+2/+3,
-- the actual sprite bitmap the hardware blits). This is an identity source
-- INDEPENDENT of the +8/9 collision-handler SW the decoder types on, so it can
-- catch SW-table mislabels (e.g. tank shells labeled Quark). Appended as a
-- separate trailing block (n*2 bytes, same walk order) so the 7-byte record
-- layout above is byte-identical for all normal consumers.
local EMIT_ANIM = os.getenv("MAME_EMIT_ANIM") == "1"
-- Exact-forward-model state (Stage 3): per-entity dynamics fields + game-global
-- sim variables, appended as trailing blocks (base layout untouched):
--   per entity (n*8 bytes, walk order): $0A/$0B X whole.frac, $0C/$0D Y whole.frac,
--     $0E/$0F X-velocity 8.8, $10/$11 Y-velocity 8.8
--   per entity (n*2 bytes): $12/$13 AI/move countdown fields
--   globals (27 bytes): $BE5C..$BE67 (12 difficulty vars), $9884-$9886 (RNG),
--     $BE68..$BE71 (10 live-enemy counts: grunts,..., quarks, tanks),
--     $98F0/$98F1 (spark / tank-shell live counters — the $F1 budget exploit)
local EMIT_SIM = os.getenv("MAME_EMIT_SIM") == "1"

local function walk_lists()
    local recs, anims, sims, timers, n = {}, {}, {}, {}, 0
    for list_id = 1, 4 do
        local node = mem:read_u8(LIST_HEADS[list_id]) * 256
                   + mem:read_u8(LIST_HEADS[list_id] + 1)
        local hops = 0
        while node ~= 0 and n < MAX_ENTITIES and hops < MAX_ENTITIES do
            hops = hops + 1
            n = n + 1
            recs[n] = string.char(
                math.floor(node / 256) % 256, node % 256, list_id,
                mem:read_u8(node + 8), mem:read_u8(node + 9),
                mem:read_u8(node + 4), mem:read_u8(node + 5))
            if EMIT_ANIM then
                anims[n] = string.char(mem:read_u8(node + 2), mem:read_u8(node + 3))
            end
            if EMIT_SIM then
                sims[n] = string.char(
                    mem:read_u8(node + 0x0A), mem:read_u8(node + 0x0B),
                    mem:read_u8(node + 0x0C), mem:read_u8(node + 0x0D),
                    mem:read_u8(node + 0x0E), mem:read_u8(node + 0x0F),
                    mem:read_u8(node + 0x10), mem:read_u8(node + 0x11))
                timers[n] = string.char(mem:read_u8(node + 0x12), mem:read_u8(node + 0x13))
            end
            node = mem:read_u8(node) * 256 + mem:read_u8(node + 1)
        end
    end
    local out = string.char(n) .. table.concat(recs)
    if EMIT_ANIM then out = out .. table.concat(anims) end
    if EMIT_SIM then
        out = out .. table.concat(sims) .. table.concat(timers)
        local g = {}
        for a = 0xBE5C, 0xBE67 do g[#g + 1] = string.char(mem:read_u8(a)) end
        for a = 0x9884, 0x9886 do g[#g + 1] = string.char(mem:read_u8(a)) end
        for a = 0xBE68, 0xBE71 do g[#g + 1] = string.char(mem:read_u8(a)) end
        g[#g + 1] = string.char(mem:read_u8(0x98F0), mem:read_u8(0x98F1))
        out = out .. table.concat(g)
    end
    return out
end

local function pack_obs()
    local hdr = string.char(
        mem:read_u8(0xBDED), mem:read_u8(0xBDEC),
        mem:read_u8(0xBDE5), mem:read_u8(0xBDE6), mem:read_u8(0xBDE7),
        mem:read_u8(0x9864), mem:read_u8(0x9865), mem:read_u8(0x9866),
        mem:read_u8(0x983F), mem:read_u8(0x9859),  -- byte9 = game_state ($9859); $1B=KILL_PLAYER, $FF=game over
        mem:read_u8(0xBDE4))  -- byte10 = score MILLIONS byte (p1_score is 4 BCD bytes
                              -- $BDE4-7, asm:131; without it scores wrap at 1M —
                              -- caught 2026-07-01 by a wave-48 game reading 408k)
    local bytes = {}
    for i = 0, SLOT_BYTES - 1 do bytes[i + 1] = string.char(mem:read_u8(SLOT_BASE + i)) end
    return hdr .. table.concat(bytes) .. walk_lists()
end

local function send_obs() sock:write(pack_obs()) end

local RESET_STATE = "rl_reset"
local CMD_STEP, CMD_RESET, CMD_QUIT, CMD_SAVE, CMD_SNAP = 0, 1, 2, 3, 4
local settle = 0

-- Read the next 4-byte command and dispatch it. This is the server's FIRST
-- socket op (a read, which triggers accept), and the read half of every
-- subsequent exchange. Sets `state` + `frames_to_run`/`settle` for the loop.
--   cmd 0 step : apply move/fire, run `skip` frames, then send obs
--   cmd 1 reset: byte2==0 -> load boot state; byte2==N -> load "w5_N"
--   cmd 2 quit : exit
--   cmd 3 save : byte2==N -> save current machine state as "w5_N", send obs
local function read_and_dispatch()
    local cmd = read_n(4)
    local c, b2, b3, skip = cmd:byte(1), cmd:byte(2), cmd:byte(3), cmd:byte(4)
    if c == CMD_QUIT then
        manager.machine:exit()
    elseif c == CMD_RESET then
        -- idx is 2 bytes (b2 high, b3 low): indices >=256 used to wrap mod 256
        -- and collide in the save-state pool. b3 is free here (only STEP uses it).
        local idx = b2 * 256 + b3
        if idx > 0 then
            manager.machine:load("w5_" .. idx)
        else
            manager.machine:load(RESET_STATE)
        end
        pending_reseed = RESEED and (idx == 0)   -- poke AFTER load settles (load is async)
        state = "loading"; settle = 3
    elseif c == CMD_SAVE then
        manager.machine:save("w5_" .. (b2 * 256 + b3))
        state = "loading"; settle = 3   -- reuse settle->checkpoint path to reply
    elseif c == CMD_SNAP then
        -- Screenshot to the MAME snapshot dir (paired with the obs the
        -- caller already holds — for visual classification verification
        -- and YOLO dataset generation).
        manager.machine.video:snapshot()
        state = "loading"; settle = 1   -- reply with obs next frame
    else
        apply_action(b2, b3)
        frames_to_run = (skip and skip > 0) and skip or FRAMESKIP_DEFAULT
        state = "run"
    end
end

-- After an operation (frames advanced, or state loaded) completes: send the
-- resulting obs, then block for the next command.
local function checkpoint()
    send_obs()
    read_and_dispatch()
end

emu.register_frame_done(function()
    if not opened then
        setup()
        sock = emu.file("rwc")
        sock:open("socket.127.0.0.1:" .. PORT)
        opened = true
        return
    end
    f = f + 1
    if state == "boot" then
        if f >= 700 and f <= 706 then F.coin:set_value(1)
        elseif f == 707 then F.coin:set_value(0) end
        if f >= 800 and f <= 810 then F.start:set_value(1)
        elseif f == 811 then F.start:set_value(0) end
        if f >= 900 then
            -- Capture the clean wave-1 start as the reset state, then await
            -- the first command. First socket op is the read in read_and_dispatch.
            manager.machine:save(RESET_STATE)
            state = "saving"; settle = 3
        end
        return
    elseif state == "saving" then
        settle = settle - 1
        if settle <= 0 then read_and_dispatch() end  -- first read (triggers accept)
        return
    elseif state == "loading" then
        settle = settle - 1
        if settle <= 0 then
            if pending_reseed then
                -- load has fully applied now; poke the RNG state ($9884-$9886)
                rng_seed = (rng_seed * 1103515245 + 12345) % 2147483648
                mem:write_u8(0x9884, math.floor(rng_seed / 8388608) % 256)
                mem:write_u8(0x9885, math.floor(rng_seed / 32768) % 256)
                mem:write_u8(0x9886, math.floor(rng_seed / 128) % 256)
                pending_reseed = false
            end
            checkpoint()  -- send loaded-state obs, read next
        end
        return
    end
    -- state == "run": count down this step's frames, then reply + read next.
    frames_to_run = frames_to_run - 1
    if frames_to_run > 0 then return end
    checkpoint()
end)
