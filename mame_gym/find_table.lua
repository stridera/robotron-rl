-- find_table.lua v2 — find the object table by structure, and validate the
-- recomp's reverse-engineered entity markers against live MAME RAM.

local mem
local frame = 0
local function attach() mem = manager.machine.devices[":maincpu"].spaces["program"] end

-- recomp's claimed state_word markers (entity_hook.cpp)
local MARKERS = {[0xAADD]="player",[0x3A76]="grunt",[0x00B6]="hulk",[0x3AA9]="electrode"}

local function read_w(a) return (mem:read_u8(a) << 8) | mem:read_u8(a + 1) end

local function analyze()
  print(string.format("[table] frame=%d", frame))

  -- (1) Where do the recomp's markers appear (if at all)?
  for val, nm in pairs(MARKERS) do
    local hits = {}
    for a = 0x0000, 0xBFFE do if read_w(a) == val then hits[#hits+1] = a end end
    local s = ""
    for i = 1, math.min(6, #hits) do s = s .. string.format("$%04X ", hits[i]) end
    print(string.format("   marker %-9s $%04X : %d hits  %s", nm, val, #hits, s))
  end

  -- (2) Any 16-bit value repeating at a CONSTANT stride = table signature.
  local pos = {}
  for a = 0x8000, 0xBFFE, 2 do            -- step 2: aligned words in the hot region
    local w = read_w(a)
    if w ~= 0 and w ~= 0xFFFF then pos[w] = pos[w] or {}; local t = pos[w]; t[#t+1] = a end
  end
  local cand = {}
  for w, locs in pairs(pos) do
    if #locs >= 8 then
      local d = locs[2] - locs[1]
      local regular = true
      for i = 3, #locs do if locs[i] - locs[i-1] ~= d then regular = false break end end
      cand[#cand+1] = {w, #locs, locs[1], d, regular}
    end
  end
  table.sort(cand, function(x,y) return x[2] > y[2] end)
  print("   constant-stride repeated words (val xcount @first stride regular?):")
  for i = 1, math.min(10, #cand) do
    local c = cand[i]
    print(string.format("     $%04X x%-3d @$%04X stride=%d %s",
          c[1], c[2], c[3], c[4], c[5] and "REGULAR" or ""))
  end
  io.flush()
end

attach()
_G._sub = emu.add_machine_frame_notifier(function()
  frame = frame + 1
  if frame == 2000 then analyze() end
end)
print("[table] analyze at frame 2000")
