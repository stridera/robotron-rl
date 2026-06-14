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
local MAX_ENTITIES = 120

local function walk_lists()
    local recs, n = {}, 0
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
            node = mem:read_u8(node) * 256 + mem:read_u8(node + 1)
        end
    end
    return string.char(n) .. table.concat(recs)
end

local function pack_obs()
    local hdr = string.char(
        mem:read_u8(0xBDED), mem:read_u8(0xBDEC),
        mem:read_u8(0xBDE5), mem:read_u8(0xBDE6), mem:read_u8(0xBDE7),
        mem:read_u8(0x9864), mem:read_u8(0x9865), mem:read_u8(0x9866),
        mem:read_u8(0x983F), mem:read_u8(0x9859))   -- byte9 = game_state ($9859); $1B=KILL_PLAYER, $FF=game over
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
        if settle <= 0 then checkpoint() end  -- send loaded-state obs, read next
        return
    end
    -- state == "run": count down this step's frames, then reply + read next.
    frames_to_run = frames_to_run - 1
    if frames_to_run > 0 then return end
    checkpoint()
end)
