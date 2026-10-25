# Bench scripts (Section 9's hardware measurements)

These ran the measurements in `records/deployment/` on the physical boards. They need the five RP2040
boards wired as in Figure 9, an NI USB-6535 digital I/O device tapping the inter-board UART lines, a CMSIS-DAP
probe with OpenOCD, and Python with `pyserial` and `nidaqmx`. Set `OPENOCD` (and `OPENOCD_SCRIPTS` if
OpenOCD's scripts are not beside it), `ENCODER_PORT` and `DECODE_PORT` for your bench.

| Script | Measures | Record |
|---|---|---|
| `conv_run.py` | The five-board chain: keys every chain-golden sentence into the encoder, captures each hop's frame, and compares frames and decoded text with `firmware/golden/chain_*.json` | `chain_{lif,mlp}.json` (2026-07-30, the record); `chain_{lif,mlp}_20261014.json` (a repeat on the same images, run to confirm the chain before further on-device work; the decode matched July's exactly, and July's one tap-window miss on sentence 57 did not recur) |
| `latency_test.py` | Per-gap compute time of the LIF and MLP segmenter boards: each of the 50 test vectors, timed from the silence before its output burst; the slope over the vectors is the per-gap cost | `latency_{lif,mlp}.json` |
| `transport_test.py` | The segmenter board's receive path: every test vector sent as a 0x06 frame at line rate, plus a back-to-back burst, each 0x03 reply checked against the golden | `transport_{lif,mlp}.json` |
| `encoder_console.py` | The encoder board's console (used by `conv_run.py`) | |

`firmware/common/measure_footprint.sh` rebuilds the LIF image and writes its size and stack frames
(`footprint_lif.txt`).

The segmenter board's DAQ wiring for `latency_test.py` and `transport_test.py`:

| NI-6535 line | Board pin | Role |
|---|---|---|
| port0/line0 (DO bit 0) | GP2 | start trigger (held low during transport) |
| port0/line1 (DO bit 1) | GP3 | 0x06 frames in |
| port1/line0 (DI bit 0) | GP4 | 0x03 frames out, and the latency build's output |
| port1/line1 (DI bit 1) | GP5 | compute marker |

Driving a frame on DO bit 0 by mistake lands it on the start trigger, and the board answers with its
end-of-pass sentinel instead of a result.
