-- verify_real_addrs.lua — finally verify the bank1_analysis addresses:
-- player X (16-bit) at $64-$65, Y at $66, lives at $EF, player entity at $985A.
-- Phased holds while watching all of them; we also dump enemy slot count for sanity.

local mem, frame = nil, 0
local in0, in2, right, left, down, up, coin, start_btn

local function attach()
  mem = manager.machine.devices[":maincpu"].spaces["program"]
  in0 = manager.machine.ioport.ports[":IN0"]
  in2 = manager.machine.ioport.ports[":IN2"]
  right     = in0.fields["Move Right"]
  left      = in0.fields["Move Left"]
  down      = in0.fields["Move Down"]
  up        = in0.fields["Move Up"]
  coin      = in2.fields["Coin 1"]
  start_btn = in0.fields["1 Player Start"]
end

attach()
local function r8(a) return mem:read_u8(a) end

local function count_alive_enemies()
  local n = 0
  for i = 0, 100 do
    local sw = (r8(0x98D4 + i*24 + 4) << 8) | r8(0x98D4 + i*24 + 5)
    if sw ~= 0 then n = n + 1 end
  end
  return n
end

_G._sub = emu.add_machine_frame_notifier(function()
  frame = frame + 1
  if frame ==  60 then coin:set_value(1) end
  if frame == 120 then coin:set_value(0) end
  if frame == 240 then start_btn:set_value(1) end
  if frame == 300 then start_btn:set_value(0) end
  -- phases (longer holds, with idle gaps)
  if frame ==  900 then right:set_value(1) end
  if frame == 1300 then right:set_value(0) end
  if frame == 1500 then left:set_value(1)  end
  if frame == 1900 then left:set_value(0)  end
  if frame == 2100 then down:set_value(1)  end
  if frame == 2500 then down:set_value(0)  end
  if frame == 2700 then up:set_value(1)    end
  if frame == 3100 then up:set_value(0)    end

  if frame % 90 == 0 then
    local tag = "idle"
    if frame >=  900 and frame < 1300 then tag = "RIGHT" end
    if frame >= 1500 and frame < 1900 then tag = "LEFT"  end
    if frame >= 2100 and frame < 2500 then tag = "DOWN"  end
    if frame >= 2700 and frame < 3100 then tag = "UP"    end
    -- 16-bit X is $64-$65 big-endian; pixel X is the HIGH byte.
    local x_hi = r8(0x826DE064)
    local x_lo = r8(0x826DE065)
    local x16  = (x_hi << 8) | x_lo
    local y    = r8(0x826DE066)
    local lives= r8(0x826DE0EF)
    local death= r8(0x826DE048)
    local dir  = r8(0x826DE03F)
    -- player entity in render buffer at $985A
    local p_x  = r8(0x826E785A)
    local p_y  = r8(0x826E785A + 1)
    local p_sw = (r8(0x826E785A + 4) << 8) | r8(0x826E785A + 5)
    local enemies = count_alive_enemies()
    print(string.format(
      "[f%4d %-5s] $64-65=$%04X (%3d px) $66=%3d  $EF=%d  $48=%d $3F=$%02X | $985A pos=(%3d,%3d) sw=$%04X | enemies=%d  IN0=$%02X",
      frame, tag, x16, x_hi, y, lives, death, dir,
      p_x, p_y, p_sw, enemies, in0:read()))
    io.flush()
  end
end)
print("[verify_real] watching $64-65/$66/$EF/$3F/$48 + $985A entity + enemy count")
