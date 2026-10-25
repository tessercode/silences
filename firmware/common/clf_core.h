// clf_core.h — SHARED fixed-point (Q16.16) compute core for the neuromorphic gap classifiers.
//
// USED BY (one module per gap-segmenter architecture, one-dir-one-module like the rest of the pipeline):
//   - firmware/spiking_gap9_lif/  (LIF spiking segmenter)   — defines clf_forward() as the LIF forward
//   - firmware/spiking_gap9_mlp/  (tanh MLP segmenter)      — defines clf_forward() as the MLP forward
//
// WHAT LIVES HERE (identical for both, single-sourced so the LIF-vs-MLP comparison can NEVER drift —
// the two must consume byte-identical features for the ablation to be valid):
//   - the Q16.16 primitives (fxmul / fxmul_round / lut_interp / fxlog / floordiv / matvec_q16),
//   - the windowed log(ISI/T̂u)+bit feature extraction + standardization (build_feats / gap_feats),
//   - the softmax -> lp3 head (softmax_lp3),
//   - the shared working buffers (logisi/g_bit/feat_raw/feat/mv/logits),
//   - the driver that walks the marks and calls clf_forward() per gap (clf_run_marks / clf_run).
//
// WHAT EACH MODULE PROVIDES (the ONLY per-architecture code):
//   - `void clf_forward(void)` — reads the shared `feat[]`, writes the shared `logits[3]`. Its own
//     model-specific buffers live in that module's clf_forward.c.
//   - its weights header `clf_weights.h` (Q16.16), and the shared LUTs/vectors, INCLUDED BEFORE this
//     header (they supply CLF_NFEAT/CLF_D*/CLF_W/CLF_PAD_LOGX_Q/CLF_MU_Q/CLF_INSD_Q used below).
//
// Single-header-library style: including this gives DECLARATIONS only; the compute TU (each module's
// clf_forward.c) does `#define CLF_CORE_IMPL` before the include to instantiate the shared IMPLEMENTATION
// once. The transport (common/clf_hw.c) and bench (common/clf_bench_main.c) include it WITHOUT the macro
// and just call the declared clf_run_marks/clf_run.
//
// BIT-EXACT mirror of spiking_gap9/fixedpoint_clf.py; reuses the gap9 phase2b Q16.16 datapath.
#pragma once
#include <stdint.h>

#define CLF_MAX_SPIKES 256
#define CLF_MAX_GAPS   255
// aliases so the reused gap9_hw.c transport (clf_hw.c) compiles unchanged against this header
#define MAX_SPIKES CLF_MAX_SPIKES
#define MAX_GAPS   CLF_MAX_GAPS
#define N_CLASSES  3
#define VEC3_ID    CLF_VEC_VID      // baked-vector id array (clf_vectors.h) for the replay path

// Whole-transmission marks (front-end 0x06) -> per-gap 3xi32 Q16.16 log-posteriors. Returns n_gaps.
int clf_run_marks(const int32_t *ticks, const int32_t *ch, int ns, int32_t tut, int32_t *lp3_out);
// Baked-vector entry (host/Renode bench), same compute.
int clf_run(int vi, int32_t *lp3_out);
// Per-architecture forward: reads shared feat[CLF_NFEAT], writes shared logits[3]. DEFINED BY THE MODULE.
void clf_forward(void);


#ifdef CLF_CORE_IMPL
// ================================================================================================
// SHARED IMPLEMENTATION — instantiated once, in the module's compute TU (clf_forward.c). Requires the
// module's clf_weights.h + common/luts.h + common/exp_lut.h + common/clf_vectors.h to be included first.
// ================================================================================================
#define HALF (Q_ONE >> 1)
_Static_assert(CLF_NLAYERS == 3, "clf_core hardcodes a 3-layer net");

// ---- Q16.16 primitives (mirror fixedpoint.py / gap9_3c.c) ----
static inline int32_t fxmul(int32_t a, int32_t b) { return (int32_t)(((int64_t)a * b) >> Q_FRAC); }
// round-to-nearest beta/scale multiply: ((b_q * M) + HALF) >> FRAC  (mirrors _fxmul_round)
static inline int64_t fxmul_round(int32_t b_q, int64_t M) { return (((int64_t)b_q * M) + HALF) >> Q_FRAC; }

static int32_t lut_interp(const int32_t *t, int32_t lo_q, int32_t step_q, int n, int32_t x_q) {
    int32_t hi_q = lo_q + (int32_t)n * step_q;
    if (x_q <= lo_q) return t[0];
    if (x_q >= hi_q) return t[n];
    int32_t off = x_q - lo_q;
    int32_t idx = off / step_q;
    int32_t rem = off - idx * step_q;
    int32_t frac = (int32_t)(((int64_t)rem << Q_FRAC) / step_q);
    int32_t t0 = t[idx], t1 = t[idx + 1];
    return t0 + (int32_t)(((int64_t)(t1 - t0) * (int64_t)frac) >> Q_FRAC);
}
static inline int32_t tanh_q(int32_t x) { return lut_interp(TANH_LUT, TANH_LUT_LO_Q, TANH_LUT_STEP_Q, 4096, x); }
static inline int32_t exp_q(int32_t x) { return lut_interp(EXP_LUT, EXP_LUT_LO_Q, EXP_LUT_STEP_Q, 4096, x); }

static int32_t fxlog(int32_t p_q) {
    if (p_q <= 0) p_q = 1;
    int e = 0; int32_t x = p_q;
    while (x < Q_ONE) { x <<= 1; e--; }
    while (x >= (Q_ONE << 1)) { x >>= 1; e++; }
    return (int32_t)(e * LN2_Q) + lut_interp(MANT_LUT, MANT_LUT_LO_Q, MANT_LUT_STEP_Q, 1024, x);
}

// floor division for T (not a power of two); mirrors python // for possibly-negative membrane sums.
static inline int64_t floordiv(int64_t a, int64_t b) {
    int64_t q = a / b, r = a % b;
    return (r != 0 && ((r < 0) != (b < 0))) ? q - 1 : q;
}

// out[k] = (sum_j W[k*ncol+j]*x[j]) >> FRAC  (real Q16.16 input; mirror matvec_q16)
static void matvec_q16(const int32_t *W, int rows, int ncol, const int32_t *x, int32_t *out) {
    for (int k = 0; k < rows; k++) {
        int64_t acc = 0;
        const int32_t *Wr = W + (int)k * ncol;
        for (int j = 0; j < ncol; j++) acc += (int64_t)Wr[j] * (int64_t)x[j];
        out[k] = (int32_t)(acc >> Q_FRAC);
    }
}

// ---- shared working buffers (static .bss; single-threaded). feat[] is the forward's INPUT, logits[]
//      its OUTPUT — the contract between this core and each module's clf_forward(). ----
static int32_t logisi[CLF_MAX_GAPS];
static int32_t g_bit[CLF_MAX_GAPS];
static int32_t feat_raw[CLF_NFEAT];
static int32_t feat[CLF_NFEAT];   // forward()'s INPUT (clf_forward is in this same TU, via CLF_CORE_IMPL)
static int32_t mv[256];           // matvec scratch (>= max layer width), used by the module forwards
static int32_t logits[3];         // forward()'s OUTPUT; softmax_lp3 reads it

static int build_feats(const int32_t *ticks, const int32_t *ch, int ns, int32_t tut) {
    int ng = ns - 1;
    for (int i = 0; i < ng; i++) {
        int32_t dt = ticks[i + 1] - ticks[i];
        int32_t isi_q = (int32_t)(((int64_t)dt * Q_ONE) / tut);
        if (isi_q < 66) isi_q = 66;                       // clip ISI/T̂u >= ~1e-3 (matches ff_features)
        logisi[i] = fxlog(isi_q);
        g_bit[i] = ch[i + 1] ? (int32_t)Q_ONE : 0;
    }
    return ng;
}

static void gap_feats(int g, int ng) {
    int W = CLF_W;
    for (int w = -W; w <= W; w++) {
        int idx = g + w, p = w + W;
        int32_t lx = (idx < 0 || idx >= ng) ? (int32_t)CLF_PAD_LOGX_Q : logisi[idx];
        int32_t bt = (idx < 0 || idx >= ng) ? 0 : g_bit[idx];
        feat_raw[p] = lx;
        feat_raw[(2 * W + 1) + p] = bt;
    }
    for (int j = 0; j < CLF_NFEAT; j++)                    // standardize: fxmul_round(insd, raw - mu)
        feat[j] = (int32_t)fxmul_round(CLF_INSD_Q[j], (int64_t)feat_raw[j] - CLF_MU_Q[j]);
}

// Exp 208 on-device posterior temperature: lp3 *= T as one fxmul_round per value (the seedsweep/scale.py
// device op, Exp 207). gen_clf_golden.py emits CLF_TEMP_Q into clf_weights.h only when it is not 2^16;
// absent -> Q_ONE, where fxmul_round is the identity.
#ifndef CLF_TEMP_Q
#define CLF_TEMP_Q Q_ONE
#endif

static void softmax_lp3(int32_t *out3) {
    int32_t mx = logits[0];
    if (logits[1] > mx) mx = logits[1];
    if (logits[2] > mx) mx = logits[2];
    int64_t e[3], S = 0;
    for (int i = 0; i < 3; i++) {
        int32_t d = logits[i] - mx;
        e[i] = (d == 0) ? (int64_t)Q_ONE : (int64_t)exp_q(d);
        S += e[i];
    }
    for (int i = 0; i < 3; i++)
        out3[i] = (int32_t)fxmul_round(CLF_TEMP_Q, fxlog((int32_t)((e[i] << 16) / S)));
}

int clf_run_marks(const int32_t *ticks, const int32_t *ch, int ns, int32_t tut, int32_t *lp3_out) {
    int ng = build_feats(ticks, ch, ns, tut);
    for (int g = 0; g < ng; g++) {
        gap_feats(g, ng);
        clf_forward();                                    // module-specific: feat[] -> logits[]
        softmax_lp3(lp3_out + (int)g * 3);
    }
    return ng;
}

int clf_run(int vi, int32_t *lp3_out) {
    int base = CLF_VEC_OFF[vi];
    return clf_run_marks(&CLF_VEC_TICKS[base], &CLF_VEC_CH[base], CLF_VEC_NSP[vi], CLF_VEC_TU[vi], lp3_out);
}
#endif  // CLF_CORE_IMPL
