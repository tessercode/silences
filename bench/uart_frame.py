#!/usr/bin/env python3
"""
Reusable DAQ-side 8N1 UART decoder + RESULT_FRAME parser (see common/RESULT_FRAME.md).

Module-agnostic: decode_8n1() + parse_frames() work for any module; only the per-payload parsers
(parse_frontend_*) are front-end specific. SG-64 / gap9 reuse decode_8n1 + parse_frames unchanged.
"""
import numpy as np

START_MARKER = bytes([0xA5, 0x5A])
MODULE_FRONTEND = 0x01


def crc8(data: bytes) -> int:
    c = 0
    for byte in data:
        c ^= byte
        for _ in range(8):
            c = ((c << 1) ^ 0x07) & 0xFF if (c & 0x80) else ((c << 1) & 0xFF)
    return c


def decode_8n1(line, sample_rate, baud, idle_high=True):
    """Decode an 8N1 UART byte stream from oversampled binary line samples.
    line: 1-D array of 0/1. Returns list of decoded byte values (ints 0..255)."""
    x = (np.asarray(line) & 1).astype(np.int8)
    spb = sample_rate / float(baud)          # samples per bit
    n = len(x)
    out = []
    i = 1
    # idle is high(1); a start bit is a high->low transition.
    while i < n:
        if x[i] == 0 and x[i - 1] == 1:      # falling edge = potential start bit
            start = i
            # verify start bit still low at its center
            c0 = int(round(start + 0.5 * spb))
            if c0 >= n or x[c0] != 0:
                i += 1
                continue
            val = 0
            ok = True
            for b in range(8):               # 8 data bits, LSB first, centered
                c = int(round(start + (1.5 + b) * spb))
                if c >= n:
                    ok = False
                    break
                if x[c]:
                    val |= (1 << b)
            if not ok:
                break
            cstop = int(round(start + 9.5 * spb))   # stop bit must be high
            if cstop < n and x[cstop] != 1:
                i += 1
                continue
            out.append(val)
            # re-sync per byte: jump into the stop bit (high) and let the loop find the NEXT
            # start-bit falling edge. Never advance by a fixed 10*spb stride (fractional spb would
            # otherwise drift across bytes and corrupt alignment).
            i = int(round(start + 9.5 * spb))
        else:
            i += 1
    return out


def parse_frames(data):
    """Scan a byte list/bytes for RESULT_FRAMEs. Returns list of dicts with verified frames."""
    b = bytes(data)
    frames = []
    i = 0
    while i + 5 <= len(b):
        if b[i] == START_MARKER[0] and b[i + 1] == START_MARKER[1]:
            if i + 4 > len(b):
                break
            module_id = b[i + 2]
            length = b[i + 3]
            end = i + 4 + length
            if end + 1 > len(b):
                i += 1
                continue
            payload = b[i + 4:end]
            chk = b[end]
            calc = crc8(bytes([module_id, length]) + payload)
            if calc == chk:
                frames.append({"module_id": module_id, "length": length,
                               "payload": payload, "crc_ok": True})
                i = end + 1
                continue
        i += 1
    return frames


# ---- front-end payload parsers (MODULE_ID 0x01) ----
def _i32(b, o):
    v = int.from_bytes(b[o:o + 4], "big", signed=True)
    return v


def _u16(b, o):
    return int.from_bytes(b[o:o + 2], "big", signed=False)


def parse_frontend_payload(payload):
    """Return dict describing a front-end payload (KIND 0x01 per-vector or 0x02 tick-test)."""
    if not payload:
        return None
    kind = payload[0]
    if kind == 0x01:
        vid = _u16(payload, 1)
        tua = _i32(payload, 3)
        tub = _i32(payload, 7)
        nfr = payload[11]
        pairs = []
        o = 12
        for _ in range(nfr):
            idx = _u16(payload, o)
            ch = payload[o + 2]
            pairs.append([idx, ch])
            o += 3
        return {"kind": "vector", "vector_id": vid, "tu_shortest_isi": tua,
                "tu_antimode": tub, "frame_first_letter": pairs}
    if kind == 0x02:
        n = payload[1]
        ticks = [_i32(payload, 2 + 4 * k) for k in range(n)]
        return {"kind": "tick_test", "tick_test_outputs": ticks}
    if kind == 0x7F:   # raw capture: n_dot, n_dash, first 6 raw counts each
        n_dot = _u16(payload, 1); n_dash = _u16(payload, 3)
        rd = [_i32(payload, 5 + 4 * k) for k in range(6)]
        ra = [_i32(payload, 5 + 4 * (6 + k)) for k in range(6)]
        return {"kind": "raw", "n_dot": n_dot, "n_dash": n_dash, "c_dot": rd, "c_dash": ra}
    if kind == 0x7E:   # reconstructed marks
        cal_counts = _i32(payload, 1); nm = payload[5]
        marks = []; o = 6
        while o + 9 <= len(payload):
            marks.append({"end": _i32(payload, o), "dur": _i32(payload, o + 4), "ch": payload[o + 8]})
            o += 9
        return {"kind": "debug", "cal_counts": cal_counts, "nm": nm, "marks": marks}
    return {"kind": "unknown", "raw": payload.hex()}


# ---- helpers to BUILD frames/waveforms (used by the decoder self-validation) ----
def build_frame(module_id, payload):
    length = len(payload)
    body = bytes([module_id, length]) + payload
    return START_MARKER + body + bytes([crc8(body)])


def bytes_to_8n1_waveform(data, sample_rate, baud, idle_pad_bits=8):
    """Render bytes as an idle-high 8N1 line at `baud`, sampled at `sample_rate`. Returns 0/1 array.
    Uses fractional (cumulative) bit boundaries so the synthetic line is at the EXACT baud a real
    UART would emit (bit period 1/baud, not quantized to whole samples)."""
    spb = sample_rate / float(baud)
    bit_seq = [1] * idle_pad_bits                      # leading idle
    for byte in data:
        bit_seq.append(0)                              # start bit
        for b in range(8):
            bit_seq.append((byte >> b) & 1)            # LSB first
        bit_seq.append(1)                              # stop bit
    bit_seq.extend([1] * idle_pad_bits)                # trailing idle
    total = int(round(len(bit_seq) * spb))
    idx = np.floor(np.arange(total) / spb).astype(int)
    idx = np.clip(idx, 0, len(bit_seq) - 1)
    return np.array(bit_seq, dtype=np.uint8)[idx]
