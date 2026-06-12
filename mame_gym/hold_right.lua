-- hold_right.lua — auto-start a game, hold Move Right continuously, and report
-- both port state AND player-slot bytes. Distinguishes "input not reaching port"
-- from "input reaches port but game ignores it."

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

attach()
_G._sub = emu.add_machine_frame_notifier(function()
  frame = frame + 1
  if frame == 60  then coin:set_value(1) end
  if frame == 120 then coin:set_value(0) end
  if frame == 240 then start_btn:set_value(1) end
  if frame == 300 then start_btn:set_value(0) end
  -- Hold Move Right continuously from frame 800 onward
  if frame >= 800 then right:set_value(1) end

  if frame % 60 == 0 then
    -- Dump first few slot pool entries' b0,b1,sw to see who's moving
    local s0  = string.format("[0]X=%3d Y=%3d sw=$%02X%02X",
      mem:read_u8(0x98D4), mem:read_u8(0x98D5), mem:read_u8(0x98D8), mem:read_u8(0x98D9))
    local s1  = string.format("[1]X=%3d Y=%3d sw=$%02X%02X",
      mem:read_u8(0x98EC), mem:read_u8(0x98ED), mem:read_u8(0x98F0), mem:read_u8(0x98F1))
    local s2  = string.format("[2]X=%3d Y=%3d sw=$%02X%02X",
      mem:read_u8(0x9904), mem:read_u8(0x9905), mem:read_u8(0x9908), mem:read_u8(0x9909))
    print(string.format("[f%4d] IN0=$%02X IN1=$%02X | %s | %s | %s",
          frame, in0:read(), in1:read(), s0, s1, s2))
    io.flush()
  end
end)
print("[hold_right] auto-coin+start, hold Right from f800, log slots every 60f")
