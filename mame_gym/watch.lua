-- watch.lua — timeline of player slot bytes during demo + long-hold RIGHT test.

local mem, frame = nil, 0
local fld = {}

local function attach()
  mem = manager.machine.devices[":maincpu"].spaces["program"]
  local m = {["Coin 1"]=":IN2", ["1 Player Start"]=":IN0", ["Move Right"]=":IN0",
             ["Move Down"]=":IN0"}
  for n, t in pairs(m) do local p = manager.machine.ioport.ports[t]; if p then fld[n] = p.fields[n] end end
end
local function r8(a) return mem:read_u8(a) end
local function press(n, v)
  if fld[n] then fld[n]:set_value(v); print(string.format("[f%d] %s := %d", frame, n, v)) end
end

attach()
_G._sub = emu.add_machine_frame_notifier(function()
  frame = frame + 1
  -- start a game so player is controllable
  if frame ==   60 then press("Coin 1", 1) end
  if frame ==  120 then press("Coin 1", 0) end
  if frame ==  240 then press("1 Player Start", 1) end
  if frame ==  300 then press("1 Player Start", 0) end
  -- long Move Right hold (10 seconds emulated)
  if frame == 1500 then press("Move Right", 1) end
  if frame == 2100 then press("Move Right", 0) end
  -- then Move Down
  if frame == 2200 then press("Move Down", 1) end
  if frame == 2700 then press("Move Down", 0) end

  if frame >= 600 and frame % 60 == 0 then
    -- annotate which input phase we're in
    local tag = "demo/idle"
    if frame >= 1500 and frame < 2100 then tag = "RIGHT held" end
    if frame >= 2200 and frame < 2700 then tag = "DOWN held" end
    print(string.format("[f%-4d %s] slot0  +0=%3d +1=%3d +2=$%02X +3=$%02X  sw=$%02X%02X  +9=$%02X",
          frame, tag, r8(0x98D4), r8(0x98D5), r8(0x98D6), r8(0x98D7),
          r8(0x98D8), r8(0x98D9), r8(0x98DD)))
    io.flush()
  end
end)
print("[watch] timeline of player slot during demo + long Right/Down holds")
