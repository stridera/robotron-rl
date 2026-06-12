-- inject.lua — start a real game via Coin+Start, then drive each direction in turn
-- to pin player X/Y. Also watch for score/wave/lives via long-window snapshots.

local mem, frame = nil, 0
local fld = {}
local snaps = {}
local change_count = {}
local prev = {}

local function attach()
  mem = manager.machine.devices[":maincpu"].spaces["program"]
  local map = {
    ["Coin 1"]         = ":IN2",
    ["1 Player Start"] = ":IN0",
    ["Move Up"]        = ":IN0", ["Move Down"]  = ":IN0",
    ["Move Left"]      = ":IN0", ["Move Right"] = ":IN0",
  }
  for name, tag in pairs(map) do
    local p = manager.machine.ioport.ports[tag]
    fld[name] = p and p.fields[name] or nil
  end
end

local function press(name, v)
  if fld[name] then fld[name]:set_value(v); print(string.format("[f%d] %s := %d", frame, name, v)) end
end

local function r8(a) return mem:read_u8(a) end
local function read_slot0()
  local s = {}; for i = 0, 23 do s[i] = r8(0x98D4 + i) end; return s
end
local function diff(a, b, tag)
  local out, total = {}, 0
  for i = 0, 23 do
    if a[i] ~= b[i] then
      local d = b[i] - a[i]
      out[#out+1] = string.format("+%d:%02X->%02X(%+d)", i, a[i], b[i], d)
      total = total + 1
    end
  end
  print(string.format("[%s] slot0 deltas (%d): %s", tag, total, table.concat(out, " ")))
end
local function snap_all()
  local s = {}; for a = 0x9000, 0xBFFF do s[a] = r8(a) end; return s
end

-- phased schedule: {frame, action, args...}
local PHASES = {
  {  60, function() press("Coin 1", 1) end},
  { 120, function() press("Coin 1", 0) end},
  { 240, function() press("1 Player Start", 1) end},
  { 300, function() press("1 Player Start", 0) end},
  -- wave-start countdown (~few seconds), so wait
  {1200, function() snaps.before_right = read_slot0(); press("Move Right", 1) end},
  {1320, function() press("Move Right", 0); diff(snaps.before_right, read_slot0(), "RIGHT") end},
  {1400, function() snaps.before_left  = read_slot0(); press("Move Left",  1) end},
  {1520, function() press("Move Left",  0); diff(snaps.before_left,  read_slot0(), "LEFT") end},
  {1600, function() snaps.before_down  = read_slot0(); press("Move Down",  1) end},
  {1720, function() press("Move Down",  0); diff(snaps.before_down,  read_slot0(), "DOWN") end},
  {1800, function() snaps.before_up    = read_slot0(); press("Move Up",    1) end},
  {1920, function() press("Move Up",    0); diff(snaps.before_up,    read_slot0(), "UP") end},
}

local function track()
  for a = 0x9000, 0xBFFF do
    local v = r8(a)
    if prev[a] ~= nil and prev[a] ~= v then change_count[a] = (change_count[a] or 0) + 1 end
    prev[a] = v
  end
end

attach()
_G._sub = emu.add_machine_frame_notifier(function()
  frame = frame + 1
  for _, p in ipairs(PHASES) do if frame == p[1] then p[2]() end end

  if frame == 600 then snaps.early = snap_all() end
  if frame >= 600 and frame <= 3000 then track() end
  if frame == 3000 then
    snaps.late = snap_all()
    -- score-like: monotonic between f600 and f3000, BCD-shape, NOT in player slot
    local cands = {}
    for a = 0x9000, 0xBFFE do
      if a < 0x98D4 or a >= 0x98EC then
        local v1, v2 = snaps.early[a], snaps.late[a]
        local d = v2 - v1
        if d > 0 then
          local bcd = (v2 & 0xF0) <= 0x90 and (v2 & 0x0F) <= 9
          cands[#cands+1] = {a, v1, v2, d, change_count[a] or 0, bcd}
        end
      end
    end
    table.sort(cands, function(x, y) return (x[4] * (x[6] and 5 or 1)) > (y[4] * (y[6] and 5 or 1)) end)
    print("\n[score?] monotonic-increase candidates f600..f3000 (BCD preferred):")
    for i = 1, math.min(25, #cands) do
      local c = cands[i]
      print(string.format("  $%04X  %3d->%3d  d=+%-4d ch=%-5d %s",
            c[1], c[2], c[3], c[4], c[5], c[6] and "BCD" or ""))
    end
    -- stable small bytes (lives/wave)
    print("\n[stable-small] late-frame bytes 1-9 with <=5 changes:")
    local n = 0
    for a = 0x9000, 0xBFFF do
      local v = snaps.late[a]
      if v >= 1 and v <= 9 and (change_count[a] or 0) <= 5 then
        print(string.format("  $%04X = %d  ch=%d", a, v, change_count[a] or 0))
        n = n + 1; if n >= 40 then print("  ...(40)"); break end
      end
    end
    io.flush()
  end
end)
print("[inject] coin+start + directional probe")
