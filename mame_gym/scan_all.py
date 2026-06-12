#!/usr/bin/env python3
"""Scan both XBLA containers at byte alignment for every CRC across all robotron
sets, so we can identify exactly which MAME set the dump is and where each chip is."""
import zlib

CONTAINERS = [
    "/home/strider/Code/Robotron2084/extracted/classic/robotron.sr",
    "/home/strider/Code/Robotron2084/extracted/classic/fw.sr",
]

# crc -> list of "set:filename" (all sets, from `mame -listroms`)
SETS = {
"robotron": [("2084_rom_10b_3005-22.a7",4096,0x13797024),("2084_rom_11b_3005-23.c7",4096,0x7e3c1b87),("2084_rom_12b_3005-24.e7",4096,0x645d543e),("2084_rom_1b_3005-13.e4",4096,0x66c7d3ef),("2084_rom_2b_3005-14.c4",4096,0x5bc6c614),("2084_rom_3b_3005-15.a4",4096,0xe99a82be),("2084_rom_4b_3005-16.e5",4096,0xafb1c561),("2084_rom_5b_3005-17.c5",4096,0x62691e77),("2084_rom_6b_3005-18.a5",4096,0xbd2c853d),("2084_rom_7b_3005-19.e6",4096,0x49ac400c),("2084_rom_8b_3005-20.c6",4096,0x3a96e88c),("2084_rom_9b_3005-21.a6",4096,0xb124367b),("video_sound_rom_3_std_767.ic12",4096,0xc56c1d28),("decoder_rom_4.3g",512,0xe6631c23),("decoder_rom_6.3c",512,0x83faf25e)],
"robotronyo":[("2084_rom_10b_3005-10.a7",4096,0x4a9d5f52),("2084_rom_11b_3005-11.c7",4096,0x2afc5e7f),("2084_rom_12b_3005-12.e7",4096,0x45da9202),("2084_rom_3b_3005-3.a4",4096,0x67a369bc),("2084_rom_4b_3005-4.e5",4096,0xb0de677a),("2084_rom_5b_3005-5.c5",4096,0x24726007),("2084_rom_6b_3005-6.a5",4096,0x028181a6),("2084_rom_7b_3005-7.e6",4096,0x4dfcceae)],
"robotron87":[("fixrobo_rom_11b.c7",4096,0xe83a2eda),("fixrobo_rom_5b.c5",4096,0x827cb5c9)],
"robotron12":[("wave201.a4",4096,0x85eb583e)],
"robotrontd":[("tiedie_rom_10b.a7",4096,0x952bea55),("tiedie_rom_11b.c7",4096,0x4c05fd3c),("tiedie_rom_4b.e5",4096,0xe8238019),("tiedie_rom_7b.e6",4096,0x3ecf4620),("tiedie_rom_8b.c6",4096,0x752d7a46)],
"robotronun":[("roboun11.7b",4096,0x8981a43b)],
}
crc_to_names, crc_to_size = {}, {}
for s, roms in SETS.items():
    for name, size, crc in roms:
        crc_to_names.setdefault(crc, []).append(f"{s}:{name}")
        crc_to_size[crc] = size

sizes = sorted(set(crc_to_size.values()))
seen = {}
for path in CONTAINERS:
    data = open(path, "rb").read()
    tag = path.split("/")[-1]
    for off in range(0, len(data) - min(sizes) + 1):
        for sz in sizes:
            if off + sz > len(data): continue
            crc = zlib.crc32(data[off:off+sz]) & 0xffffffff
            if crc in crc_to_names:
                key = crc
                if key not in seen:
                    seen[key] = (tag, off, sz, crc_to_names[crc])
                    print(f"{tag} @0x{off:06x} sz={sz:5d} crc={crc:08x}  -> {', '.join(crc_to_names[crc])}")
print(f"\n{len(seen)} distinct chip CRCs located")
