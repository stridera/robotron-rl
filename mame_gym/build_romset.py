#!/usr/bin/env python3
"""Build robotron87.zip directly from the owned XBLA container by ROM grid position.

The container lays the 6809 ROMs out on a clean 0x1000 grid in order
10b,11b,12b,1b,2b,...,9b, then the 2 decoder PROMs, then the sound ROM. 13 of 15
chips match MAME's robotron87 CRCs exactly; 7b/10b and the decoder PROMs are a
patched XBLA revision (no cataloged CRC). We use the container's bytes regardless —
the program ROMs are the game logic we train against; decoder/sound only affect
video/audio, which the RL agent never reads.
"""
import zlib, zipfile

SR = "/home/strider/Code/Robotron2084/extracted/classic/robotron.sr"
OUT = "/home/strider/Code/robotron-rl/mame_gym/roms/robotron87.zip"

# robotron87 chip name -> (container offset, size, expected_crc or None if XBLA-variant)
LAYOUT = [
    ("2084_rom_10b_3005-22.a7", 0x2800, 4096, None),        # XBLA variant
    ("fixrobo_rom_11b.c7",      0x3800, 4096, 0xe83a2eda),
    ("2084_rom_12b_3005-24.e7", 0x4800, 4096, 0x645d543e),
    ("2084_rom_1b_3005-13.e4",  0x5800, 4096, 0x66c7d3ef),
    ("2084_rom_2b_3005-14.c4",  0x6800, 4096, 0x5bc6c614),
    ("2084_rom_3b_3005-15.a4",  0x7800, 4096, 0xe99a82be),
    ("2084_rom_4b_3005-16.e5",  0x8800, 4096, 0xafb1c561),
    ("fixrobo_rom_5b.c5",       0x9800, 4096, 0x827cb5c9),
    ("2084_rom_6b_3005-18.a5",  0xa800, 4096, 0xbd2c853d),
    ("2084_rom_7b_3005-19.e6",  0xb800, 4096, None),        # XBLA variant
    ("2084_rom_8b_3005-20.c6",  0xc800, 4096, 0x3a96e88c),
    ("2084_rom_9b_3005-21.a6",  0xd800, 4096, 0xb124367b),
    ("decoder_rom_4.3g",        0xe800,  512, None),        # XBLA variant
    ("decoder_rom_6.3c",        0xea00,  512, None),        # XBLA variant
    ("video_sound_rom_3_std_767.ic12", 0xf000, 4096, 0xc56c1d28),
]

data = open(SR, "rb").read()
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as z:
    for name, off, size, exp in LAYOUT:
        chunk = data[off:off + size]
        crc = zlib.crc32(chunk) & 0xffffffff
        ok = "match" if exp is None else ("OK" if crc == exp else f"MISMATCH(exp {exp:08x})")
        flag = " [XBLA-variant]" if exp is None else ""
        print(f"  {name:32s} @0x{off:06x} sz={size:5d} crc={crc:08x} {ok}{flag}")
        z.writestr(name, chunk)
print(f"\nwrote {OUT}")
