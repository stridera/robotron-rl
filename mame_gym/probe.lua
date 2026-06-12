-- probe.lua — locate Robotron's active RAM (where game state lives) in MAME.
-- Strategy: snapshot the whole 6809 address space periodically and report which
-- 256-byte pages change the most. Active game variables = frequently-changing pages.

local mem
local frame = 0
local snap = {}          -- last full-space byte snapshot
local churn = {}         -- per-page cumulative change counter (page = addr>>8)

local function attach()
  local cpu = manager.machine.devices[":maincpu"]
  mem = cpu.spaces["program"]
end

local function sample()
  for page = 0, 0xFF do churn[page] = churn[page] or 0 end
  for a = 0x0000, 0xBFFF do          -- skip $C000+ (I/O) and $D000+ (ROM)
    local v = mem:read_u8(a)
    local p = (a >> 8) & 0xFF
    if snap[a] ~= nil and snap[a] ~= v then churn[p] = churn[p] + 1 end
    snap[a] = v
  end
end

local function top_pages(n)
  local arr = {}
  for p, c in pairs(churn) do if c > 0 then arr[#arr+1] = {p, c} end end
  table.sort(arr, function(x, y) return x[2] > y[2] end)
  local s = ""
  for i = 1, math.min(n, #arr) do
    s = s .. string.format("$%02Xxx:%d ", arr[i][1], arr[i][2])
  end
  return s
end

local function on_frame()
  frame = frame + 1
  if frame <= 3 or frame % 180 == 0 then
    sample()
    print(string.format("[probe] frame=%d  hottest pages: %s", frame, top_pages(12)))
    io.flush()
  end
end

attach()
if emu.add_machine_frame_notifier then
  -- keep the subscription alive in a global so it isn't GC'd
  _G._sub = emu.add_machine_frame_notifier(on_frame)
elseif emu.register_frame_done then
  emu.register_frame_done(on_frame)
end
print("[probe] attached; will report hottest RAM pages over $0000-$BFFF")
