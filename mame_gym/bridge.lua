-- bridge.lua — synchronous Gymnasium bridge over named pipes (FIFOs).
-- Protocol per frame:
--   Lua  -> Py : OBS_BYTES of observation
--   Py   -> Lua: 4 bytes  [move_dir, fire_dir, frameskip, flags]
--     flags bit 0 = save state ("checkpoint") at next quiesce
--     flags bit 1 = load state ("checkpoint") at next quiesce
-- Directions: 0=none, 1=N, 2=NE, 3=E, 4=SE, 5=S, 6=SW, 7=W, 8=NW
--
-- Auto-starts a real game (Coin+Start) and waits a boot delay before letting
-- Python in, so the first observation Python sees is post-boot.

local mem, frame = nil, 0
local fld_move, fld_fire = {}, {}
local fifo_in, fifo_out

local SLOTS_REPORTED = 24
local OBS_BYTES = 2 + (SLOTS_REPORTED - 1) * 4   -- player(X,Y) + 23 slots*(b0,b1,sw_hi,sw_lo)

-- direction -> {up, down, left, right} bits
local DIR = {
  [0]={0,0,0,0}, [1]={1,0,0,0}, [2]={1,0,0,1}, [3]={0,0,0,1}, [4]={0,1,0,1},
  [5]={0,1,0,0}, [6]={0,1,1,0}, [7]={0,0,1,0}, [8]={1,0,1,0},
}

local function attach()
  mem = manager.machine.devices[":maincpu"].spaces["program"]
  local p0 = manager.machine.ioport.ports[":IN0"]
  local p1 = manager.machine.ioport.ports[":IN1"]
  local p2 = manager.machine.ioport.ports[":IN2"]
  fld_move.up=p0.fields["Move Up"]; fld_move.down=p0.fields["Move Down"]
  fld_move.left=p0.fields["Move Left"]; fld_move.right=p0.fields["Move Right"]
  fld_fire.up=p0.fields["Fire Up"]; fld_fire.down=p0.fields["Fire Down"]
  fld_fire.left=p1.fields["Fire Left"]; fld_fire.right=p1.fields["Fire Right"]
  -- stash coin/start for auto-start
  fld_move.coin  = p2.fields["Coin 1"]
  fld_move.start = p0.fields["1 Player Start"]
end

local function set_dir(fields, d)
  local t = DIR[d] or DIR[0]
  fields.up:set_value(t[1]); fields.down:set_value(t[2])
  fields.left:set_value(t[3]); fields.right:set_value(t[4])
end

local function r8(a) return mem:read_u8(a) end

local function build_obs()
  local parts = {string.char(r8(0x98D4)), string.char(r8(0x98D5))}
  for i = 1, SLOTS_REPORTED - 1 do
    local b = 0x98D4 + i * 24
    parts[#parts+1] = string.char(r8(b))
    parts[#parts+1] = string.char(r8(b+1))
    parts[#parts+1] = string.char(r8(b+4))
    parts[#parts+1] = string.char(r8(b+5))
  end
  return table.concat(parts)
end

local function open_fifos()
  local pin  = os.getenv("BRIDGE_IN")  or "/tmp/robo_py_to_mame.fifo"
  local pout = os.getenv("BRIDGE_OUT") or "/tmp/robo_mame_to_py.fifo"
  -- MUST match open order on Python side so opens rendezvous instead of deadlocking.
  fifo_in  = assert(io.open(pin,  "rb"))
  fifo_out = assert(io.open(pout, "wb"))
  fifo_in:setvbuf("no"); fifo_out:setvbuf("no")
  print("[bridge] fifos open  in="..pin.." out="..pout)
end

attach()
open_fifos()
print(string.format("[bridge] OBS_BYTES=%d  ready", OBS_BYTES))

_G._sub = emu.add_machine_frame_notifier(function()
  frame = frame + 1
  -- auto-start the game (Coin + 1P Start) before handing control to Python
  if frame ==  60 then fld_move.coin:set_value(1)  end
  if frame == 120 then fld_move.coin:set_value(0)  end
  if frame == 240 then fld_move.start:set_value(1) end
  if frame == 300 then fld_move.start:set_value(0) end
  if frame <  600 then return end          -- skip boot frames

  -- send observation
  fifo_out:write(build_obs())
  -- read next action (blocking 4 bytes)
  local data = fifo_in:read(4)
  if not data or #data < 4 then return end
  local move  = data:byte(1)
  local fire  = data:byte(2)
  local flags = data:byte(4)
  -- save/load take effect at the next quiesce point (start of next frame)
  if (flags & 1) ~= 0 then
    manager.machine:save("checkpoint")
    print(string.format("[bridge] save queued at frame %d", frame))
  end
  if (flags & 2) ~= 0 then
    manager.machine:load("checkpoint")
    print(string.format("[bridge] load queued at frame %d", frame))
  end
  set_dir(fld_move, move)
  set_dir(fld_fire, fire)
end)
