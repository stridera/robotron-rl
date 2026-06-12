-- discover.lua — locate player X/Y + score + wave + lives in a single pass.
--
-- Plan:
--   * frame 60     : enumerate MAME input ports & fields (so we can inject later)
--   * f 100-3000   : track per-byte change counts in $9000-$BFFF
--   * f 500 & 3000 : two RAM snapshots; their diff finds monotonic increases (score)
--   * f 3000       : report player-slot churn, score candidates, lives/wave candidates

local mem, frame = nil, 0
local prev, change_count = {}, {}
local first_snap, second_snap

local function attach() mem = manager.machine.devices[":maincpu"].spaces["program"] end
local function r8(a) return mem:read_u8(a) end

local function dump_ports()
  print("[ports] input ports/fields available for injection:")
  local ok, err = pcall(function()
    for tag, port in pairs(manager.machine.ioport.ports) do
      for name, field in pairs(port.fields) do
        print(string.format("  port %-20s field %-32s mask=$%X",
              tag, "'"..name.."'", field.mask or 0))
      end
    end
  end)
  if not ok then print("  enumeration error: " .. tostring(err)) end
  io.flush()
end

local function snap()
  local s = {}
  for a = 0x9000, 0xBFFF do s[a] = r8(a) end
  return s
end

local function track()
  for a = 0x9000, 0xBFFF do
    local v = r8(a)
    if prev[a] ~= nil and prev[a] ~= v then change_count[a] = (change_count[a] or 0) + 1 end
    prev[a] = v
  end
end

local function report()
  -- Player slot byte-by-byte
  print("\n[player] slot 0 ($98D4..$98EB) change counts and values:")
  for i = 0, 23 do
    local a = 0x98D4 + i
    print(string.format("  +%2d ($%04X)  changes=%-4d  f500=$%02X  f3000=$%02X",
          i, a, change_count[a] or 0, first_snap[a], second_snap[a]))
  end

  -- Monotonic increase candidates (score)
  local cands = {}
  for a = 0x9000, 0xBFFE do
    local d = second_snap[a] - first_snap[a]
    if d > 0 then cands[#cands+1] = {a, first_snap[a], second_snap[a], d, change_count[a] or 0} end
  end
  table.sort(cands, function(x,y)
    -- prefer bytes that changed a lot AND ended much higher (score-like)
    return (x[4] * (x[5] + 1)) > (y[4] * (y[5] + 1))
  end)
  print("\n[mono] top monotonic-increase candidates (score/timer-like):")
  for i = 1, math.min(25, #cands) do
    local c = cands[i]
    -- annotate if it looks like BCD (high nibble 0-9 and low nibble 0-9)
    local bcd = ""
    if (c[3] & 0xF0) <= 0x90 and (c[3] & 0x0F) <= 9 then bcd = "  (BCD-shape)" end
    print(string.format("  $%04X  %3d->%3d  delta=+%d  changes=%d%s",
          c[1], c[2], c[3], c[4], c[5], bcd))
  end

  -- Small stable bytes (lives/wave candidates)
  print("\n[const] small stable bytes in $98xx-$9Cxx (lives/wave candidates):")
  local n = 0
  for a = 0x9800, 0x9CFF do
    local v = second_snap[a]
    local ch = change_count[a] or 0
    if v >= 1 and v <= 9 and ch <= 3 and v == first_snap[a] then
      print(string.format("  $%04X  val=%d  changes=%d", a, v, ch))
      n = n + 1
      if n >= 30 then print("  ... (capped at 30)") break end
    end
  end

  io.flush()
end

attach()
_G._sub = emu.add_machine_frame_notifier(function()
  frame = frame + 1
  if frame == 60   then dump_ports() end
  if frame == 500  then first_snap = snap() end
  if frame >= 100 and frame <= 3000 then track() end
  if frame == 3000 then second_snap = snap(); report() end
end)
print("[discover] enum ports f60, snap f500/f3000, report f3000")
