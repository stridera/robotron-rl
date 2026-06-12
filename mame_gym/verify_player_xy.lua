-- verify_player_xy.lua — test whether 6809 $5E/$5F really are player X/Y
-- by holding Move Right and watching them change.

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

_G._sub = emu.add_machine_frame_notifier(function()
  frame = frame + 1
  if frame ==  60 then coin:set_value(1) end
  if frame == 120 then coin:set_value(0) end
  if frame == 240 then start_btn:set_value(1) end
  if frame == 300 then start_btn:set_value(0) end

  -- Phased holds: right 800-1100, left 1200-1500, down 1600-1900, up 2000-2300
  if frame == 800  then right:set_value(1)  end
  if frame == 1100 then right:set_value(0)  end
  if frame == 1200 then left:set_value(1)   end
  if frame == 1500 then left:set_value(0)   end
  if frame == 1600 then down:set_value(1)   end
  if frame == 1900 then down:set_value(0)   end
  if frame == 2000 then up:set_value(1)     end
  if frame == 2300 then up:set_value(0)     end

  if frame % 60 == 0 then
    local tag = "idle"
    if frame >= 800  and frame < 1100 then tag = "RIGHT" end
    if frame >= 1200 and frame < 1500 then tag = "LEFT"  end
    if frame >= 1600 and frame < 1900 then tag = "DOWN"  end
    if frame >= 2000 and frame < 2300 then tag = "UP"    end
    -- 6809 $5E/$5F: candidate player X/Y; also dump alt $64/$66, $48 being-killed, $59 pause
    print(string.format("[f%4d %-5s] $5E=%3d $5F=%3d  alt $64=%3d $66=%3d  $48=%d $59=%d  IN0=$%02X",
          frame, tag,
          r8(0x826DE05E), r8(0x826DE05F),
          r8(0x826DE064), r8(0x826DE066),
          r8(0x826DE048), r8(0x826DE059),
          in0:read()))
    io.flush()
  end
end)
print("[verify_xy] cycling R/L/D/U with $5E/$5F as candidate player X/Y")
