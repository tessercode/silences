// gap9_3c fixed-point segmenter — A2.1 port of the 3-class head onto the UNCHANGED phase2b
// Q16.16 datapath. BIT-EXACT mirror of a2_gap9_3c_golden/fixedpoint_3c.py (which is itself gated
// bit-exact against phase2a/fixedpoint.py). Derived from phase2b/firmware/gap9.c (sha
// 7d44e6dc...); the diff is: 3-row fc head, the binary output block REPLACED by the 3-way
// log-softmax block (exp LUT + per-class Q16.16 divide + prob-mean + fxlog), per-table POP access
// (corrected per-rate domains, pop_luts_3c.h), and D4 argmax-free 3xi32 output. Feature math,
// matvec, GRU, LUT interp, fxlog are byte-identical to the copied original.
#include "gap9_3c.h"
#include "luts.h"          /* SIGMOID/TANH/MANT LUTs + feature constants (POP_* grid UNUSED here) */
#include "exp_lut.h"       /* A2.0b golden exp LUT [-16,0], 4097 x 1/256 */
#include "pop_luts_3c.h"   /* A2.0b golden corrected POP tables (PER-TABLE step/hi) */
#include "weights_3c.h"    /* 5 seeds, order [42,47,48,49,43]; fc 3-row */
#include "vectors_3c.h"    /* 97 eval vectors (47 baked + 50 IW-bearing) */

/* A2 = single seed (index 0 = seed 42); A3 builds the SAME code with -DGAP9_3C_N_SEEDS=5. */
#ifndef GAP9_3C_N_SEEDS
#define GAP9_3C_N_SEEDS 1
#endif
_Static_assert(GAP9_3C_N_SEEDS >= 1 && GAP9_3C_N_SEEDS <= N_SEEDS, "seed count > baked weights");

/* Exp 208 on-device posterior temperature, per build: lp3 *= T as one round-to-nearest Q16.16 multiply
 * after fxlog (== clf_core.h fxmul_round == seedsweep/scale.py). gen_a2_golden.py emits the two values into
 * weights_3c.h when device_temperature.json exists; absent -> Q_ONE (identity). */
#ifndef GAP9_3C_TEMP_Q_N1
#define GAP9_3C_TEMP_Q_N1 Q_ONE
#endif
#ifndef GAP9_3C_TEMP_Q_N5
#define GAP9_3C_TEMP_Q_N5 Q_ONE
#endif
#if GAP9_3C_N_SEEDS == 1
#define GAP9_3C_TEMP_Q GAP9_3C_TEMP_Q_N1
#elif GAP9_3C_N_SEEDS == 5
#define GAP9_3C_TEMP_Q GAP9_3C_TEMP_Q_N5
#else
#error "GAP9_3C_N_SEEDS has no calibrated temperature (1 or 5 only)"
#endif
_Static_assert(N_CLASSES == 3, "3-way head expected");
/* SEMANTIC feature constants must equal the read-only originals (the LUT grid is deliberately
 * different — do NOT assert POP_STEP_Q/POP_HI_Q here). */
_Static_assert(TAU_DUR_Q == 229376 && TAU_DOT_Q == 196608 && TAU_DASH_Q == 327680
               && INV_GAP_S_Q == 109227, "feature constants drifted from luts.h originals");
_Static_assert(sizeof(EXP_LUT) / sizeof(EXP_LUT[0]) == 4097, "exp LUT length");
_Static_assert(EXP_LUT_LO_Q == -1048576 && EXP_LUT_STEP_Q == 256, "exp LUT grid");
_Static_assert(sizeof(POP3C_LUTS) / sizeof(POP3C_LUTS[0]) == 6 * (POP3C_N + 1), "POP3C length");
_Static_assert(sizeof(POP3C_STEP_Q) / sizeof(POP3C_STEP_Q[0]) == 6, "POP3C per-table steps");

// ---- Q16.16 multiply: ((int64)a*b) >> FRAC. Arithmetic shift (GCC: signed >> is arithmetic). ----
static inline int32_t fxmul(int32_t a, int32_t b) {
    return (int32_t)(((int64_t)a * (int64_t)b) >> Q_FRAC);
}

// ---- generic LUT with integer linear interpolation (mirrors FxLUT.__call__) ----
static int32_t lut_interp(const int32_t *t, int32_t lo_q, int32_t step_q, int n, int32_t x_q) {
    int32_t hi_q = lo_q + (int32_t)n * step_q;          // exact by construction (power-of-two ranges)
    if (x_q <= lo_q) return t[0];
    if (x_q >= hi_q) return t[n];
    int32_t off = x_q - lo_q;                            // >= 0
    int32_t idx = off / step_q;                          // floor (positive operands)
    int32_t rem = off - idx * step_q;
    int32_t frac = (int32_t)(((int64_t)rem << Q_FRAC) / step_q);  // Q16.16 in [0,1)
    int32_t t0 = t[idx], t1 = t[idx + 1];
    return t0 + (int32_t)(((int64_t)(t1 - t0) * (int64_t)frac) >> Q_FRAC);
}

static inline int32_t sigmoid_q(int32_t x) {
    return lut_interp(SIGMOID_LUT, SIGMOID_LUT_LO_Q, SIGMOID_LUT_STEP_Q, 4096, x);
}
static inline int32_t tanh_q(int32_t x) {
    return lut_interp(TANH_LUT, TANH_LUT_LO_Q, TANH_LUT_STEP_Q, 4096, x);
}
static inline int32_t pop_q(int j, int32_t x) {
    /* PER-TABLE constants (corrected per-rate domains) — the shared POP_STEP_Q/POP_HI_Q of
     * luts.h:801-804 NO LONGER APPLY (a2_golden_reference.md §6.2 port note). */
    return lut_interp(&POP3C_LUTS[j * (POP3C_N + 1)], POP3C_LO_Q, POP3C_STEP_Q[j], POP3C_N, x);
}
static inline int32_t exp_q(int32_t x) {
    return lut_interp(EXP_LUT, EXP_LUT_LO_Q, EXP_LUT_STEP_Q, 4096, x);
}

// ---- log(p) for p_q in Q16.16, p in (0,1]: frexp + mantissa LUT (mirrors fxlog) ----
static int32_t fxlog(int32_t p_q) {
    if (p_q <= 0) p_q = 1;
    int e = 0;
    int32_t x = p_q;
    while (x < Q_ONE) { x <<= 1; e--; }
    while (x >= (Q_ONE << 1)) { x >>= 1; e++; }
    int32_t logm = lut_interp(MANT_LUT, MANT_LUT_LO_Q, MANT_LUT_STEP_Q, 1024, x);
    return (int32_t)(e * LN2_Q) + logm;
}

// ---- working buffers (static -> .bss, single-threaded; keeps stack tiny) ----
static int32_t feats[MAX_GAPS][9];
static int32_t fwd_h[MAX_GAPS][H];
static int32_t rev_h[MAX_GAPS][H];
static int64_t psum3[MAX_GAPS][3];    // per-class sum of per-seed Q16.16 posteriors

// ---- integer feature rows (mirror fx_feature_rows) ----
// INPUT SOURCE re-parameterized (process-on-arrival): the marks come from caller-supplied buffers
// (absolute completion ticks[], next-channel ch[], n_spikes ns) + T̂u-ticks (tut), instead of the
// baked VEC_* arrays. The FEATURE MATH BELOW IS UNCHANGED — byte-for-byte the same ISI/pop/sigmoid
// computation — so the SAME validated compute runs on either baked vectors (gap9_run) or live 0x06
// marks (gap9_run_marks). ISIs are differences (ticks[i+1]-ticks[i]) so the absolute tick ORIGIN is
// irrelevant: baked-absolute and front-end-relative marks with identical ISIs yield identical features.
static int build_features(const int32_t *ticks, const int32_t *ch, int ns, int32_t tut) {
    int ng = ns - 1;
    for (int i = 0; i < ng; i++) {
        int32_t dt = ticks[i + 1] - ticks[i];
        int32_t isi_q = (int32_t)(((int64_t)dt * Q_ONE) / tut);    // floor, positive
        int32_t ch_next = ch[i + 1];
        for (int j = 0; j < N_POP; j++) feats[i][j] = pop_q(j, isi_q);
        feats[i][6] = ch_next ? (int32_t)Q_ONE : 0;
        feats[i][7] = sigmoid_q(fxmul(isi_q - TAU_DUR_Q, INV_GAP_S_Q));
        int32_t tau = ch_next ? TAU_DASH_Q : TAU_DOT_Q;
        feats[i][8] = sigmoid_q(fxmul(isi_q - tau, INV_GAP_S_Q));
    }
    return ng;
}

// out[k] = (sum_j W[k*ncol+j]*x[j]) >> FRAC, int64 accumulation (mirror _matvec_q16)
static void matvec_q16(const int32_t *W, int rows, int ncol, const int32_t *x, int32_t *out) {
    for (int k = 0; k < rows; k++) {
        int64_t acc = 0;
        const int32_t *Wr = W + (int)k * ncol;
        for (int j = 0; j < ncol; j++) acc += (int64_t)Wr[j] * (int64_t)x[j];
        out[k] = (int32_t)(acc >> Q_FRAC);
    }
}

// One GRU direction for one seed. dir=0 forward, dir=1 reverse. Writes into hstore[gap][H].
static void gru_dir(int seed, int dir, int ng, int32_t hstore[][H]) {
    const int32_t *gih = (dir == 0 ? w_ih : w_ihr) + (int)seed * 96 * 9;
    const int32_t *ghh = (dir == 0 ? w_hh : w_hhr) + (int)seed * 96 * 32;
    const int32_t *gbih = (dir == 0 ? b_ih : b_ihr) + (int)seed * 96;
    const int32_t *gbhh = (dir == 0 ? b_hh : b_hhr) + (int)seed * 96;
    int32_t h[H];
    for (int k = 0; k < H; k++) h[k] = 0;
    int32_t ir[H], hr[H], iz[H], hz[H], inn[H], hn[H];
    for (int s = 0; s < ng; s++) {
        int t = (dir == 0) ? s : (ng - 1 - s);          // input index; rev writes realigned
        const int32_t *x = feats[t];
        // gate rows: r=0..31, z=32..63, n=64..95
        matvec_q16(gih + 0 * 32 * 9,  H, 9,  x, ir);
        matvec_q16(ghh + 0 * 32 * 32, H, 32, h, hr);
        matvec_q16(gih + 1 * 32 * 9,  H, 9,  x, iz);
        matvec_q16(ghh + 1 * 32 * 32, H, 32, h, hz);
        matvec_q16(gih + 2 * 32 * 9,  H, 9,  x, inn);
        matvec_q16(ghh + 2 * 32 * 32, H, 32, h, hn);
        int32_t nh[H];
        for (int k = 0; k < H; k++) {
            int32_t pre_r = ir[k] + gbih[k]        + hr[k] + gbhh[k];
            int32_t pre_z = iz[k] + gbih[32 + k]   + hz[k] + gbhh[32 + k];
            int32_t r_q = sigmoid_q(pre_r);
            int32_t z_q = sigmoid_q(pre_z);
            int32_t inner_h = hn[k] + gbhh[64 + k];                 // W_hn h + b_hn
            int32_t pre_n = inn[k] + gbih[64 + k] + fxmul(r_q, inner_h);
            int32_t n_q = tanh_q(pre_n);
            nh[k] = fxmul((int32_t)Q_ONE - z_q, n_q) + fxmul(z_q, h[k]);
        }
        for (int k = 0; k < H; k++) h[k] = nh[k];
        for (int k = 0; k < H; k++) hstore[t][k] = h[k];
    }
}

// The ENSEMBLE compute (BiGRU × seeds → per-gap 3-way log-softmax posteriors → prob-mean →
// fxlog). The binary output block (Δlogit → sigmoid → psum-mean → fxlog → 1-bit argmax) is
// REPLACED, mirroring fixedpoint_3c.py::fx_seed_p3 / fx_log_p3_ticks exactly; the seed loop and
// integer mean stay (gap9.c:129,151 structure). D4: argmax-free — 3×i32 log-posteriors per gap.
static int gap9_finish(int ng, int32_t *lp3_out) {
    for (int t = 0; t < ng; t++)
        for (int i = 0; i < 3; i++) psum3[t][i] = 0;
    for (int seed = 0; seed < GAP9_3C_N_SEEDS; seed++) {
        gru_dir(seed, 0, ng, fwd_h);
        gru_dir(seed, 1, ng, rev_h);
        const int32_t *fcw = fc_w + (int)seed * 3 * 64;
        const int32_t *fcb = fc_b + (int)seed * 3;
        for (int t = 0; t < ng; t++) {
            int32_t x[3];
            for (int r = 0; r < 3; r++) {                      // int64 dot >>16 + Q16.16 bias
                int64_t acc = 0;
                for (int j = 0; j < H; j++) acc += (int64_t)fcw[r * 64 + j] * (int64_t)fwd_h[t][j];
                for (int j = 0; j < H; j++) acc += (int64_t)fcw[r * 64 + (H + j)] * (int64_t)rev_h[t][j];
                x[r] = (int32_t)(acc >> Q_FRAC) + fcb[r];
            }
            int32_t m = x[0];                                  // 3-way max
            if (x[1] > m) m = x[1];
            if (x[2] > m) m = x[2];
            int64_t e[3], S = 0;
            for (int i = 0; i < 3; i++) {
                int32_t d = x[i] - m;                          // d <= 0; d == 0 at (every) argmax
                e[i] = (d == 0) ? (int64_t)Q_ONE : (int64_t)exp_q(d);   // ties: BOTH get Q_ONE
                S += e[i];                                     // S in [Q_ONE, 3*Q_ONE]
            }
            for (int i = 0; i < 3; i++)                        // Q16.16 posterior; positive ⇒
                psum3[t][i] += (int32_t)((e[i] << 16) / S);    // C trunc == floor == python //
        }
    }
    for (int t = 0; t < ng; t++)
        for (int i = 0; i < 3; i++)                            // integer mean, ONE fxlog per class
            lp3_out[t * 3 + i] = (int32_t)((((int64_t)GAP9_3C_TEMP_Q *         /* Exp 208 temperature */
                                 fxlog((int32_t)(psum3[t][i] / GAP9_3C_N_SEEDS))) + (Q_ONE >> 1)) >> Q_FRAC);
    return ng;
}

// Baked-vector entry — features from the baked VEC3_* arrays, then the shared 3-way compute.
int gap9_run(int vi, int32_t *lp3_out) {
    int base = VEC3_OFFSET[vi];
    int ng = build_features(&SPIKE3_TICKS[base], &SPIKE3_CH[base], VEC3_NSPIKE[vi], VEC3_TU_TICKS[vi]);
    return gap9_finish(ng, lp3_out);
}

// Live process-on-arrival entry — the SAME compute on caller-supplied whole-transmission marks +
// T̂u (a front-end 0x06 frame). Only the feature INPUT SOURCE differs; the compute is identical.
int gap9_run_marks(const int32_t *ticks, const int32_t *ch, int ns, int32_t tut, int32_t *lp3_out) {
    int ng = build_features(ticks, ch, ns, tut);
    return gap9_finish(ng, lp3_out);
}
