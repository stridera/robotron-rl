-- scan_player.lua — hold Move Right for a long time, then report which RAM
-- bytes in $9000-$BFFF actually responded to the input. The playable player's
-- X should increase substantially over the hold (player walks right until hitting
-- a wall). HUD bytes won't, attract demo bytes wouldn't either.

local mem, frame = nil, 0
local in0, in1, in2, right, coin, start_btn

local function attach()
  in0 = manager.machine.ioport.ports[":IN0"]
  in1 = manager.machine.ioport.ports[":IN1"]
  in2 = manager.machine.ioport.ports[":IN2"]
  right     = in0.fields["Move Right"]
  coin      = in2.fields["Coin 1"]
  start_btn = in0.fields["1 Player Start"]
  mem = manager.machine.devices[":maincpu"].spaces["program"]
end

local function snap()
  local s = {}
  for a = 0x9000, 0xBFFF do s[a] = mem:read_u8(a) end
  return s
end

attach()
local snap_a, snap_b

_G._sub = emu.add_machine_frame_notifier(function()
  frame = frame + 1
  if frame ==  60 then coin:set_value(1) end
  if frame == 120 then coin:set_value(0) end
  if frame == 240 then start_btn:set_value(1) end
  if frame == 300 then start_btn:set_value(0) end
  if frame >= 800 then right:set_value(1) end

  if frame == 1000 then snap_a = snap(); print("[scan] snap_a at f1000") end
  if frame == 2800 then
    snap_b = snap()
    print("[scan] snap_b at f2800; comparing across 1800 frames of held Right")
    -- bytes whose value INCREASED substantially
    local up = {}
    for a = 0x9000, 0xBFFF do
      local d = snap_b[a] - snap_a[a]
      if d >= 20 then up[#up+1] = {a, snap_a[a], snap_b[a], d} end
    end
    table.sort(up, function(x,y) return x[4] > y[4] end)
    print(string.format("[scan] %d bytes increased by >=20:", #up))
    for i = 1, math.min(30, #up) do
      local c = up[i]
      print(string.format("  $%04X: %3d -> %3d  (+%d)", c[1], c[2], c[3], c[4]))
    end
    -- and bytes that DECREASED substantially (Y might invert)
    local down = {}
    for a = 0x9000, 0xBFFF do
      local d = snap_b[a] - snap_a[a]
      if d <= -20 then down[#down+1] = {a, snap_a[a], snap_b[a], d} end
    end
    table.sort(down, function(x,y) return x[4] < y[4] end)
    print(string.format("[scan] %d bytes decreased by <=-20:", #down))
    for i = 1, math.min(15, #down) do
      local c = down[i]
      print(string.format("  $%04X: %3d -> %3d  (%d)", c[1], c[2], c[3], c[4]))
    end
    io.flush()
  end
end)
print("[scan_player] hold Right f800..f2800, scan RAM-wide for moved bytes")
