-- verify_lives.lua — read $AA6B continuously while injecting a coin+start.
-- If it's the lives counter, we expect to see it become 3/4/5 after a game starts
-- (default robotron87 lives), then potentially decrement on death.

local mem, frame = nil, 0
local in0, in2, coin, start_btn

local function attach()
  mem = manager.machine.devices[":maincpu"].spaces["program"]
  in0 = manager.machine.ioport.ports[":IN0"]
  in2 = manager.machine.ioport.ports[":IN2"]
  coin      = in2.fields["Coin 1"]
  start_btn = in0.fields["1 Player Start"]
end

attach()
print("[verify_lives] monitoring $AA6B over time")

_G._sub = emu.add_machine_frame_notifier(function()
  frame = frame + 1
  if frame ==  60 then coin:set_value(1) end
  if frame == 120 then coin:set_value(0) end
  if frame == 240 then start_btn:set_value(1) end
  if frame == 300 then start_btn:set_value(0) end

  if frame % 120 == 0 then
    -- watch $AA6B and a small neighborhood for context
    local b = {}
    for i = -2, 4 do b[#b+1] = string.format("%02X", mem:read_u8(0xAA6B + i)) end
    print(string.format("[f%4d] $AA6B-2..+4 = %s   [center]$AA6B=%d",
          frame, table.concat(b, " "), mem:read_u8(0xAA6B)))
    io.flush()
  end
end)
