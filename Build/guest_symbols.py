import pathlib
import struct
import sys

data = pathlib.Path(sys.argv[1]).read_bytes()
assert data[:4] == b'\x7fELF'
phoff = struct.unpack_from('<Q', data, 32)[0]
phsize, phcount = struct.unpack_from('<HH', data, 54)
headers = [struct.unpack_from('<IIQQQQQQ', data, phoff + i * phsize) for i in range(phcount)]
dynamic = next(h for h in headers if h[0] == 2)
dynlib = next(h for h in headers if h[0] == 0x61000000)
tags = dict(struct.unpack_from('<QQ', data, i) for i in range(dynamic[2], dynamic[2] + dynamic[5], 16))
strings = dynlib[2] + tags[0x61000035]
symbols = dynlib[2] + tags[0x61000039]
for i in range(0, tags[0x6100003f], tags[0x6100003b]):
    name, info, other, section, value, size = struct.unpack_from('<IBBHQQ', data, symbols + i)
    text = data[strings + name:].split(b'\0', 1)[0].decode()
    if any(x in text for x in sys.argv[2:]):
        print(text, hex(value), hex(size))
