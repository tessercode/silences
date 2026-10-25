/*
 * fists.h — Exp-186 representative operator "fists" (styles) for the on-device encoder.
 *
 * SOURCE (verify-before-baking): analysis/operator_style_fits.csv (1164 per-segment fits of the 44
 * real Exp-4 recordings; columns wpm,r,sigma_elem,omega,group — NO representative flag). The K=3
 * representative fists are the per-stratum MEDIANS under Exp-186's OWN style_stratum rule
 * (experiment186_operator_style_generator.ipynb cell 8):
 *     r >= 4.0            -> high_r_bug
 *     wpm >= 26.0 (r<4)   -> fast_normalr
 *     else                -> slow_normalr
 * computed over the trimmed style cloud (WPM in [6,60], r in [2,7]) — the SAME 1101 segments Exp-186
 * used ("n_style_segments_used": 1101, exp186_results.json). sigma_gap = sigma_elem * 1.15 (Exp-186
 * SIGMA_GAP_RATIO; the corpus per-segment ratio, NOT the 0.89 pooled ratio).
 *
 * INCREMENT 2 (JITTER): wpm, r AND sigma_elem/sigma_gap/omega are now all APPLIED. The style-only path
 * (encoder_render_styled, sigma=omega=0) is unchanged and remains the boot default; the jittered path
 * (encoder_render_styled_jit, via the shared seeded PRNG enc_rng.h) reads sigma_elem/sigma_gap/omega
 * when :jit is on. Under jitter a perturbed element/gap duration CAN straddle the channel threshold, so
 * the render uses T_thresh = 1.92 (the device value); the matched host reference mirrors 1.92.
 */
#ifndef ENCODER_FISTS_H
#define ENCODER_FISTS_H

typedef struct {
    const char *name;
    double wpm;          /* APPLIED */
    double r;            /* APPLIED (dash ratio) */
    double sigma_elem;   /* RECORDED-BUT-UNUSED this increment (future jitter) */
    double sigma_gap;    /* RECORDED-BUT-UNUSED */
    double omega;        /* RECORDED-BUT-UNUSED */
} enc_fist_t;

/* CSV-derived per-stratum medians (see report Step 1b). Literal decimals: gcc, atof, and Python float()
 * all parse these to the identical IEEE-754 double, so the device render and the host reference agree
 * bit-for-bit. */
static const enc_fist_t ENC_FISTS[] = {
    /* name            wpm      r        sigma_elem  sigma_gap   omega   */
    { "fast_normalr",  35.87,   3.0302,  0.1005,     0.1156,     0.1081 },
    { "slow_normalr",  17.496,  3.1453,  0.1304,     0.1500,     0.0760 },
    { "high_r_bug",    29.083,  4.7748,  0.3041,     0.3497,     0.3049 },
    /* Cootie-class high-jitter representative (Exp-190 source of record, 2026-07-10). Same recipe as
     * fists 1-3: per-set median + D5 rounding, sigma_gap = rounded_sigma x 1.15. Member set = 13
     * speed-consistent segments of the two recordings with speed-consistent pooled sigma >= 0.40 T_u
     * over >= 50 residuals (IK5TTA-COOTIE 0.694, IU8GVN-MONARCH 0.467). The genuine sigma-extreme
     * surviving T_u-misfit exclusion; also plugs the mid-speed coverage gap. omega is the corpus's
     * lowest-confidence axis; the renderer floors the per-word multiplier at 0.2. Jittered runs are
     * validated by correct decode / distribution, not frozen bit-exact. Appended index 3 -> :fist 4. */
    { "cootie_high_jitter", 22.116,  3.6653,  0.5191,     0.5970,     0.7122 },
    /* Corrected cootie representative (Phase 2, 2026-07-13). IDENTICAL to cootie_high_jitter except
     * omega 0.7122 -> 0.3774: the baked omega was inflated by decode/noise-blip artifacts in the fit
     * (30/31 extreme per-word ovals demonstrated artifacts; analysis/cootie_omega_correction_phase1.md).
     * 0.3774 = artifacts-out, no leakage subtraction — the conservative/robust Phase-1 choice, -47%
     * vs entry 3. Entry 3 (:fist 4) is RETAINED as the silicon-validated (Exp-191/192) conservative
     * stress case; this entry (:fist 5) is the honest REPRESENTATIVE operator. wpm/r/sigma unchanged
     * (fit axes not implicated). Appended index 4 -> :fist 5. */
    { "cootie_corrected",   22.116,  3.6653,  0.5191,     0.5970,     0.3774 },
};
#define ENC_N_FISTS ((int)(sizeof(ENC_FISTS) / sizeof(ENC_FISTS[0])))

#endif /* ENCODER_FISTS_H */
