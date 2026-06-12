-- dump_slots.lua — read out Robotron's object table during live gameplay.
-- recomp claims: slot pool base $98D4, stride 24, up to 101 slots, state_word at +4.
-- Player confirmed at slot 0 (state_word $AADD @ $98D8). Dump real contents so we
-- learn the true enemy markers and per-slot field layout the gym needs.

local mem
local frame = 0
local BASE, STRIDE, N = 0x98D4, 24, 64
local function attach() mem = manager.machine.devices[":maincpu"].spaces["program"] end
local function r8(a) return mem:read_u8(a) end
local function rw(a) return (r8(a) << 8) | r8(a + 1) end

local function dump()
  print(string.format("[slots] frame=%d  base=$%04X stride=%d", frame, BASE, STRIDE))
  local typecount = {}
  for i = 0, N - 1 do
    local s = BASE + i * STRIDE
    local sw = rw(s + 4)
    if sw ~= 0 then
      typecount[sw] = (typecount[sw] or 0) + 1
      if i < 24 then          -- show first 24 slots' raw fields
        print(string.format("   slot %2d @$%04X  sw=$%04X  b0..b11=%02X %02X %02X %02X | %02X %02X %02X %02X | %02X %02X %02X %02X",
          i, s, sw, r8(s),r8(s+1),r8(s+2),r8(s+3),r8(s+4),r8(s+5),r8(s+6),r8(s+7),r8(s+8),r8(s+9),r8(s+10),r8(s+11)))
      end
    end
  end
  print("   state_word histogram (sw: count) -- shared values = enemy types:")
  local arr = {}
  for sw, c in pairs(typecount) do arr[#arr+1] = {sw, c} end
  table.sort(arr, function(x,y) return x[2] > y[2] end)
  for _, e in ipairs(arr) do print(string.format("     $%04X x%d", e[1], e[2])) end
  io.flush()
end

attach()
_G._sub = emu.add_machine_frame_notifier(function()
  frame = frame + 1
  if frame == 1500 or frame == 3000 then dump() end
end)
print("[slots] dump at frames 1500 and 3000")
