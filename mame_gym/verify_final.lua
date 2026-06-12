-- verify_final.lua — read absolute addresses derived from DP=$98:
--   $9864-$9865 = player X (16-bit); pixel X = high byte
--   $9866       = player Y
--   $98EF       = player lives
--   $985A..     = player render entity
-- Phased holds + monitor changes; this is the decisive verification.

local mem, frame = nil, 0
local in0, in2, right, left, down, up, coin, start_btn
local function attach()
  mem = manager.machine.devices[":maincpu"].spaces["program"]
  in0 = manager.machine.ioport.ports[":IN0"]
  in2 = manager.machine.ioport.ports[":IN2"]
  right=in0.fields["Move Right"]; left=in0.fields["Move Left"]
  down=in0.fields["Move Down"];   up=in0.fields["Move Up"]
  coin=in2.fields["Coin 1"];      start_btn=in0.fields["1 Player Start"]
end
attach()
local function r8(a) return mem:read_u8(a) end

_G._sub = emu.add_machine_frame_notifier(function()
  frame = frame + 1
  if frame ==  60 then coin:set_value(1) end
  if frame == 120 then coin:set_value(0) end
  if frame == 240 then start_btn:set_value(1) end
  if frame == 300 then start_btn:set_value(0) end

  if frame == 1200 then right:set_value(1) end
  if frame == 1800 then right:set_value(0) end
  if frame == 2000 then left:set_value(1) end
  if frame == 2600 then left:set_value(0) end
  if frame == 2800 then down:set_value(1) end
  if frame == 3400 then down:set_value(0) end

  if frame % 60 == 0 then
    local tag = "idle"
    if frame >= 1200 and frame < 1800 then tag = "RIGHT" end
    if frame >= 2000 and frame < 2600 then tag = "LEFT"  end
    if frame >= 2800 and frame < 3400 then tag = "DOWN"  end
    -- Absolute addresses (DP=$98 makes <$64 = $9864):
    local x16 = (r8(0x826E7864) << 8) | r8(0x826E7865)
    local y   = r8(0x826E7866)
    local lives = r8(0x826E78EF)
    local death = r8(0x826E7848)
    local dir   = r8(0x826E783F)
    print(string.format("[f%4d %-5s] X=$%04X(px %3d) Y=%3d  lives=%d dead=%d dir=$%02X  IN0=$%02X",
          frame, tag, x16, x16 >> 8, y, lives, death, dir, in0:read()))
    io.flush()
  end
end)
