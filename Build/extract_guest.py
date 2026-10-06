import pathlib
import struct
import sys

source = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
data = source.read_bytes()
count = struct.unpack_from('<H', data, 24)[0]
elf_pos = 32 + count * 32
elf = data[elf_pos:elf_pos + 64]
assert elf[:4] == b'\x7fELF'
phoff = struct.unpack_from('<Q', elf, 32)[0]
phsize, phcount = struct.unpack_from('<HH', elf, 54)
headers = [struct.unpack_from('<IIQQQQQQ', data, elf_pos + phoff + i * phsize)
           for i in range(phcount)]
output = bytearray(max(h[2] + h[5] for h in headers))
output[:64] = elf
output[phoff:phoff + phsize * phcount] = data[elf_pos + phoff:elf_pos + phoff + phsize * phcount]
for i in range(count):
    flags, off, packed, unpacked = struct.unpack_from('<QQQQ', data, 32 + i * 32)
    if flags & 0x800:
        assert not flags & 10, 'encrypted/compressed segments are unsupported'
        header = headers[(flags >> 20) & 0xfff]
        output[header[2]:header[2] + header[5]] = data[off:off + header[5]]
target.write_bytes(output)
print(target, len(output))
