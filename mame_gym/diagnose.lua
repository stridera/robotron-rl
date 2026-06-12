-- diagnose.lua — figure out why the player isn't moving:
--   * try the alternative port:write() injection (vs field:set_value())
--   * dump slots 0..15 periodically; if enemies move while player doesn't,
--     it's a player-state issue; if nothing moves, we're in a non-gameplay screen.

local mem, frame = nil, 0
local function attach() mem = manager.machine.devices[":maincpu"].spaces["program"] end
local function r8(a) return mem:read_u8(a) end

local ports = {}
local function get_port(t) ports[t] = ports[t] or manager.machine.ioport.ports[t]; return ports[t] end

-- Two injection methods to compare
local function inject_field(port_tag, field_name, on)
  local p = get_port(port_tag); if not p then return end
  local f = p.fields[field_name]; if not f then return end
  f:set_value(on and 1 or 0)
  print(string.format("[f%d] field %s/%s := %d", frame, port_tag, field_name, on and 1 or 0))
end
local function inject_write(port_tag, mask, on)
  local p = get_port(port_tag); if not p then return end
  p:write(on and mask or 0, mask)
  print(string.format("[f%d] port %s write mask=$%X val=%d", frame, port_tag, mask, on and mask or 0))
end

local function dump_slots(tag)
  local present, moving = 0, 0
  local lines = {}
  for i = 0, 15 do
    local base = 0x98D4 + i * 24
    local sw_hi, sw_lo = r8(base + 4), r8(base + 5)
    local sw = (sw_hi << 8) | sw_lo
    if sw ~= 0 then
      present = present + 1
      lines[#lines+1] = string.format("[%d]b0=%-3d b1=%-3d sw=$%04X", i, r8(base), r8(base+1), sw)
    end
  end
  print(string.format("[f%d %s] active_slots=%d  %s", frame, tag, present,
        table.concat(lines, " | ")))
end

attach()
_G._sub = emu.add_machine_frame_notifier(function()
  frame = frame + 1
  -- start a game (try BOTH methods on coin to maximize chances)
  if frame ==   60 then inject_field(":IN2", "Coin 1", true) end
  if frame ==  120 then inject_field(":IN2", "Coin 1", false) end
  if frame ==  240 then inject_field(":IN0", "1 Player Start", true) end
  if frame ==  300 then inject_field(":IN0", "1 Player Start", false) end

  -- method A: field:set_value()
  if frame ==  900 then inject_field(":IN0", "Move Right", true) end
  if frame == 1500 then inject_field(":IN0", "Move Right", false) end
  -- method B: port:write()
  if frame == 1700 then inject_write(":IN0", 0x08, true) end   -- Move Right mask
  if frame == 2300 then inject_write(":IN0", 0x08, false) end

  if frame == 400 or frame == 700 or frame == 1000 or frame == 1300 or
     frame == 1600 or frame == 1900 or frame == 2200 or frame == 2500 or
     frame == 2800 then
    dump_slots(({[400]="post-coin",[700]="post-start",
      [1000]="field-right held",[1300]="field-right held",[1600]="released",
      [1900]="port-write right",[2200]="port-write right",
      [2500]="released",[2800]="released"})[frame])
  end
end)
print("[diagnose] try both injection methods, dump slot activity")
