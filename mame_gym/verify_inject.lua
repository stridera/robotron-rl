-- verify_inject.lua — does field:set_value actually press the button?
-- Print field state + raw port read before/after to confirm whether the
-- override is sticking.

local frame = 0
local right, in0
local function attach()
  in0 = manager.machine.ioport.ports[":IN0"]
  right = in0.fields["Move Right"]
end

local function probe(tag)
  -- ioport_field exposes a `live` substructure with the current state.
  local fv = pcall(function() return right.live.value end) and right.live.value or "(no .live.value)"
  -- direct attributes
  local rv = pcall(function() return right.value end) and right.value or "(no .value)"
  local pr = in0:read()
  print(string.format("[%s] right.value=%s  right.live.value=%s  IN0:read()=$%X",
        tag, tostring(rv), tostring(fv), pr))
  io.flush()
end

attach()
print(string.format("[verify] Move Right field mask=$%X type=%s",
      right.mask, tostring(right.type)))

_G._sub = emu.add_machine_frame_notifier(function()
  frame = frame + 1
  if frame == 100 then
    probe("before set_value(1)")
    right:set_value(1)
    probe("after  set_value(1)")
  end
  if frame == 150 then probe("frame 150 (50 after set)") end
  if frame == 200 then
    right:set_value(0)
    probe("after  set_value(0)")
  end
end)
