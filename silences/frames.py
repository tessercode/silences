"""The byte frames the five RP2040 boards exchange over UART (Section 9).

Every frame is  A5 5A | module id | payload length (u16) | payload | CRC-8 , with the CRC (poly 0x07,
initial 0, MSB first) over the module id, both length bytes and the payload. Multi-byte fields are
big-endian two's complement.

    0x06  front end -> segmenter, router   kind 1, seq u16, time unit i32 (us), n marks u16,
                                           then per mark: completion time i32 (us), channel u8
          kind 0x7F (transmission too long): seq u16, marks observed u16
    0x03  segmenter -> router              kind 1, vector id u16, n gaps u16, then per gap three
                                           i32 Q16.16 log-posteriors (intra, inter-letter, inter-word)
    0x05  router -> decoder                kind 1, vector id u16, n marks u16, the n channel bytes,
                                           then per gap the three log-posteriors
"""

START = b"\xa5\x5a"
MODULE_FE = 0x06
MODULE_SEGMENTER = 0x03
MODULE_ROUTER = 0x05
KIND_FORWARD = 0x01
KIND_OVERFLOW = 0x7F
N_MARKS_MAX = 256
MAX_GAPS = N_MARKS_MAX - 1


def crc8(data):
    c = 0
    for b in data:
        c ^= b
        for _ in range(8):
            c = ((c << 1) ^ 0x07) & 0xFF if (c & 0x80) else (c << 1) & 0xFF
    return c


def frame(module_id, payload):
    assert len(payload) <= 0xFFFF
    hdr = bytes([module_id, (len(payload) >> 8) & 0xFF, len(payload) & 0xFF])
    return START + hdr + payload + bytes([crc8(hdr + payload)])


def _u16(v):
    assert 0 <= v <= 0xFFFF
    return v.to_bytes(2, "big")


def _i32(v):
    return int(v).to_bytes(4, "big", signed=True)


def _open(buf, module_id):
    assert buf[:2] == START, "bad start marker"
    length = (buf[3] << 8) | buf[4]
    payload = buf[5:5 + length]
    assert len(payload) == length, "truncated payload"
    assert crc8(buf[2:5 + length]) == buf[5 + length], "CRC-8 mismatch"
    assert len(buf) == 6 + length, "trailing bytes"
    assert module_id is None or buf[2] == module_id, f"module 0x{buf[2]:02x}"
    return buf[2], payload


def _rows(payload, o, n):
    return [[int.from_bytes(payload[o + 12 * g + 4 * k:o + 12 * g + 4 * k + 4], "big", signed=True) for k in range(3)]
            for g in range(n)]


# ---- 0x06: front end ----
def fe_payload(seq, tu_us, marks):
    pl = bytearray([KIND_FORWARD]) + _u16(seq) + _i32(tu_us) + _u16(len(marks))
    for end, ch in marks:
        pl += _i32(end) + bytes([int(ch) & 0xFF])
    return bytes(pl)


def overflow_payload(seq, n_marks_observed):
    return bytes([KIND_OVERFLOW]) + _u16(seq) + _u16(min(n_marks_observed, 0xFFFF))


def fe_frame(seq, tu_us, ticks, ch):
    return frame(MODULE_FE, fe_payload(seq, int(tu_us), [(int(t), int(c)) for t, c in zip(ticks, ch)]))


def parse_fe(buf):
    module_id, pl = _open(buf, None)
    kind, seq = pl[0], (pl[1] << 8) | pl[2]
    if kind == KIND_FORWARD:
        nm = (pl[7] << 8) | pl[8]
        assert len(pl) == 9 + 5 * nm, "payload size != 9 + 5 n"
        marks = [(int.from_bytes(pl[9 + 5 * i:13 + 5 * i], "big", signed=True), pl[13 + 5 * i]) for i in range(nm)]
        return dict(module_id=module_id, kind=kind, seq=seq, tu_us=int.from_bytes(pl[3:7], "big", signed=True), marks=marks)
    assert kind == KIND_OVERFLOW and len(pl) == 5, f"unknown kind 0x{kind:02x}"
    return dict(module_id=module_id, kind=kind, seq=seq, n_marks_observed=(pl[3] << 8) | pl[4])


# ---- 0x03: segmenter ----
def segmenter_frame(vid, lp3):
    assert len(lp3) <= MAX_GAPS
    pl = bytearray([KIND_FORWARD]) + _u16(vid) + _u16(len(lp3))
    for row in lp3:
        assert len(row) == 3
        pl += b"".join(_i32(v) for v in row)
    return frame(MODULE_SEGMENTER, bytes(pl))


def parse_segmenter(buf):
    _, pl = _open(buf, MODULE_SEGMENTER)
    ng = (pl[3] << 8) | pl[4]
    assert pl[0] == KIND_FORWARD and len(pl) == 5 + 12 * ng
    return dict(vector_id=(pl[1] << 8) | pl[2], n_gaps=ng, lp3=_rows(pl, 5, ng))


# ---- 0x05: router ----
def router_frame(vid, ch, lp3):
    assert len(ch) <= N_MARKS_MAX and len(lp3) == len(ch) - 1
    pl = bytearray([KIND_FORWARD]) + _u16(vid) + _u16(len(ch)) + bytes(int(c) & 0xFF for c in ch)
    for row in lp3:
        assert len(row) == 3
        pl += b"".join(_i32(v) for v in row)
    return frame(MODULE_ROUTER, bytes(pl))


def parse_router(buf):
    _, pl = _open(buf, MODULE_ROUTER)
    n = (pl[3] << 8) | pl[4]
    assert pl[0] == KIND_FORWARD and len(pl) == 5 + n + 12 * (n - 1)
    return dict(vector_id=(pl[1] << 8) | pl[2], n_spikes=n, ch=list(pl[5:5 + n]), lp3=_rows(pl, 5 + n, n - 1))
