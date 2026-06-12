-- check_dp.lua — track the 6809 DP register over time. If it's non-zero during
-- gameplay, the "<$5E" etc. addresses in the disassembly are at DP*256+offset.

local cpu, mem, frame = nil, nil, 0
local in0, in2, coin, start_btn
local function attach()
  cpu = manager.machine.devices[":maincpu"]
  mem = cpu.spaces["program"]
  in0 = manager.machine.ioport.ports[":IN0"]
  in2 = manager.machine.ioport.ports[":IN2"]
  coin      = in2.fields["Coin 1"]
  start_btn = in0.fields["1 Player Start"]
end

attach()

-- list state keys once
print("[state-keys] available 6809 CPU state symbols:")
for k, v in pairs(cpu.state) do
  print("  '" .. tostring(k) .. "' -> symbol='" .. tostring(v.symbol) .. "'")
end

_G._sub = emu.add_machine_frame_notifier(function()
  frame = frame + 1
  if frame ==  60 then coin:set_value(1) end
  if frame == 120 then coin:set_value(0) end
  if frame == 240 then start_btn:set_value(1) end
  if frame == 300 then start_btn:set_value(0) end

  if frame % 60 == 0 then
    -- read each candidate state slot
    local dp = cpu.state["DP"] and cpu.state["DP"].value or "?"
    local x = cpu.state["X"] and cpu.state["X"].value or "?"
    local y = cpu.state["Y"] and cpu.state["Y"].value or "?"
    local pc = cpu.state["PC"] and cpu.state["PC"].value or "?"
    -- assuming DP is the page selector: read DP*256 + offsets to find live player data
    local dpb = (type(dp) == "number") and dp * 256 or 0
    print(string.format("[f%4d] DP=%s PC=$%04X X=$%04X Y=$%04X  dpb+5E=%d dpb+64-65=%d dpb+66=%d dpb+EF=%d",
          frame, tostring(dp), pc, x, y,
          mem:read_u8(0x826DE000 + dpb + 0x5E),
          (mem:read_u8(0x826DE000 + dpb + 0x64) << 8) | mem:read_u8(0x826DE000 + dpb + 0x65),
          mem:read_u8(0x826DE000 + dpb + 0x66),
          mem:read_u8(0x826DE000 + dpb + 0xEF)))
    io.flush()
  end
end)
