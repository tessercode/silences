// gap9_3c fixed-point segmenter — A2.1 3-class port of the phase2b Q16.16 module (bit-exact spec
// = a2_gap9_3c_golden/fixedpoint_3c.py). Derived from phase2b/firmware/gap9.h (sha 6a84378e...).
#ifndef GAP9_3C_H
#define GAP9_3C_H
#include <stdint.h>

/* B2b.1: the cap raise. Cost is SRAM ONLY, not time — feats/fwd_h/rev_h/psum3 are .bss statics,
 * boot-zeroed once and overwritten per gap; there is no per-invocation memset on the callee path
 * (chain_pacing_analysis.md 2.3 refuted the anticipated O(MAX_GAPS) clear). Compute-side bss:
 * 9,796 B -> 80,580 B (see b2b_1_port.md 5). MAX_SPIKES is consumed by BOTH tracks — the compute
 * buffers here and the wrapper's RX guard (gap9_hw.c) — which is why it lives in this header. */
#define MAX_SPIKES 256           /* B2b.1: 32 -> 256 (N_MARKS_MAX); was the deployed word cap */
#define MAX_GAPS   (MAX_SPIKES - 1)

// Compute per-gap 3-way log-posteriors for vector index `vi` of the baked VEC3_* set.
// Writes 3 Q16.16 values per gap, gap-major ([lp0,lp1,lp2] x n_gaps, up to 3*MAX_GAPS ints);
// returns the number of gaps. D4: argmax-free — NO decision bits. Integer-only (FPU-free).
int gap9_run(int vi, int32_t *lp3_out);

// PROCESS-ON-ARRIVAL entry: same compute on caller-supplied whole-transmission marks
// (absolute completion ticks[], next-channel ch[], n_spikes ns) + T-hat-u ticks (tut) — e.g. a
// front-end 0x06 forward frame. Returns n_gaps (= ns-1). Integer-only (FPU-free).
int gap9_run_marks(const int32_t *ticks, const int32_t *ch, int ns, int32_t tut, int32_t *lp3_out);

#endif
