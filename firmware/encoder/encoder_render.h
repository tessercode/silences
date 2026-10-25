/*
 * encoder_render.h — deterministic pure-ITU TEMPO render (increment 1), shared by the RP2040
 * firmware (encoder_hw.c) and the host self-check (encoder_host.c) so BOTH run byte-identical logic.
 *
 * Reproduces float_reference.render_word_style (deploy/rp2040/phase2a/float_reference.py:157-187) —
 * the exact function that rendered the frozen phase2a vectors — for the pure-ITU regime
 * (sigma_elem=0, sigma_gap=0, omega=0, r=3). Under that regime the reference is RNG-FREE:
 *   - omega_word = max(0.2, normal(1.0, 0.0)) == 1.0            => t_u = t_u_base = 1.2/wpm
 *   - element noise / gap noise are guarded by `if sigma > 0`  => never drawn
 * so no random generator is needed here. (Increments 2+ add jitter; deferred.)
 *
 * ARITHMETIC ORDER (critical, per the Stage-3 lesson). The reference ACCUMULATES spike times in
 * double SECONDS (running cursor `t_cursor += ...`) and only converts each spike to integer us-ticks
 * at the very end via round(t_s * 1e6). We mirror that EXACTLY:
 *   - accumulate t_cursor / t_spike in double seconds (IEEE-754 double is identical on x86 host and
 *     the M0+ soft-float toolchain), THEN
 *   - tick = lrint(t_spike * 1e6).  lrint honours the default FE_TONEAREST rounding mode = round-half-
 *     to-EVEN, which is exactly Python's round(). (round-then-accumulate — e.g. rounding t_u to an
 *     integer tick first — DIVERGES: it puts spike 6 of "LUKE" at 634505 vs the reference 634498.)
 */
#ifndef ENCODER_RENDER_H
#define ENCODER_RENDER_H
#include <stdint.h>
#include <math.h>     /* lrint */
#include "enc_rng.h"  /* increment 2: shared seedable PRNG + Box-Muller (jittered render only) */

/* Morse table, A-Z (idx 0..25) then 0-9 (idx 26..35). Byte-identical to
 * tempo/dataset/generate_dataset.py:21-32 (imported verbatim by float_reference.py). */
static const char *const ENC_MORSE[36] = {
    ".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---",
    "-.-", ".-..", "--", "-.", "---", ".--.", "--.-", ".-.", "...", "-",
    "..-", "...-", ".--", "-..-", "-.--", "--..",
    "-----", ".----", "..---", "...--", "....-", ".....", "-....", "--...", "---..", "----."
};

static inline const char *enc_morse_lookup(char c) {
    if (c >= 'A' && c <= 'Z') return ENC_MORSE[c - 'A'];
    if (c >= '0' && c <= '9') return ENC_MORSE[26 + (c - '0')];
    return 0;                                  /* unknown -> skip (mirror `if ch not in MORSE_TABLE`) */
}

/* pure-ITU render constants */
#define ENC_R_VAL      3.0     /* dash ratio r (dash duration = r * T_u) */
#define ENC_INTER_GAP  3.0     /* inter-letter gap = 3 * T_u (fixed word/letter spacing; NOT r) */
#define ENC_T_THRESH   1.92    /* CANONICAL generate_dataset.py:54 (t_thresh = 1.92*t_u); was 2.10
                                * (E50 drift). channel = (dur < T_thresh*T_u) ? 0 : 1. Invisible at
                                * deterministic r=3 (dot=1<1.92, dash=3>=1.92 -> identical channels). */

/* STYLE-parameterized render: adds `r` (dash ratio) as a PARAMETER (was the ENC_R_VAL constant). This
 * is the SIMPLE operator-styling increment — STYLE-ONLY: only wpm and r vary per operator fist; jitter
 * (sigma_elem/sigma_gap/omega) is NOT applied here (style-only, sigma=omega=0), so the render stays
 * RNG-free and bit-exact. The timing MATH is otherwise IDENTICAL to the deterministic renderer — the
 * ONLY change is `ENC_R_VAL` -> `r`. Passing r == ENC_R_VAL reproduces the pure-ITU render exactly.
 * (Threshold note: at sigma=0 with r >= ~2.64 every dash dur = r*t_u sits above BOTH 1.92 and 2.10 and
 *  every dot dur = 1*t_u below both, so the 1.92 device / 2.10 reference threshold gives identical
 *  channels — the fists are all r >= 3.03, so this holds.) */
static inline int encoder_render_styled(const char *word, double wpm, double r,
                                        uint32_t *tick_us, uint8_t *ch, int cap) {
    const double t_u       = 1.2 / wpm;              /* t_u_base; omega_word == 1.0 (style-only, no drift) */
    const double t_thresh  = ENC_T_THRESH * t_u;
    const double floor_dur = 0.1 * t_u;              /* 0.1*T_u floor (kept for fidelity) */
    double t_cursor = 0.0;
    int n = 0;
    for (const char *wp = word; *wp; ++wp) {
        const char *els = enc_morse_lookup(*wp);
        if (!els) continue;
        int L = 0; while (els[L]) ++L;               /* element count for this letter */
        for (int e = 0; e < L; ++e) {
            double t_ideal = (els[e] == '.') ? (1.0 * t_u) : (r * t_u);   /* r PARAM (was ENC_R_VAL const) */
            double t_dur   = t_ideal + 0.0;          /* element noise == 0 (style-only) */
            if (t_dur < floor_dur) t_dur = floor_dur;                 /* max(t_ideal, 0.1*T_u) */
            double t_spike = t_cursor + t_dur;
            if (n < cap) {
                tick_us[n] = (uint32_t)lrint(t_spike * 1e6);          /* round-half-to-even == round() */
                ch[n]      = (t_dur < t_thresh) ? 0u : 1u;
                ++n;
            }
            if (e < L - 1) {                          /* intra-element gap = 1*T_u (render_word_style:180-182) */
                double g = 1.0 * t_u + 0.0;           /* gap noise == 0 (style-only) */
                if (g < floor_dur) g = floor_dur;
                t_cursor = t_spike + g;
            } else {                                  /* last element: NO intra gap (the eidx<len-1 guard) */
                t_cursor = t_spike;
            }
        }
        double gl = ENC_INTER_GAP * t_u + 0.0;        /* inter-letter gap = 3*T_u (render_word_style:185-186) */
        if (gl < floor_dur) gl = floor_dur;
        t_cursor += gl;
    }
    return n;
}

/* JITTERED render (increment 2) — mirrors float_reference.render_word_style (float_reference.py:157-187)
 * EXACTLY: application, draw order, and the max()/floor clamps. The ONLY intentional deviations, both
 * flagged in the report:
 *   (1) T_thresh = 1.92 (ENC_T_THRESH, the DEVICE value) for the channel decision, NOT render_word_
 *       style's 2.10 — decided for this increment because under jitter a perturbed dur can straddle the
 *       threshold, so the channel now depends on it. The matched host reference (enc_rng_ref.py) uses
 *       1.92 too, so device==host.
 *   (2) the omega speed-drift draw is GUARDED on omega>0 (render_word_style always draws it). This makes
 *       sigma_elem=sigma_gap=omega=0 consume ZERO uniforms, so the jittered path with jitter OFF reduces
 *       BYTE-IDENTICALLY to encoder_render_styled above (and thus nominal still reproduces frozen vec-30).
 *       For the fists (all omega>0) the guard is inactive, so jittered renders are unaffected by it.
 *
 * DRAW ORDER (must match host for bit-exactness), all via the shared enc_rng Box-Muller N(0,1):
 *   draw #0  omega_word = max(0.2, 1 + omega*z)              [once per word, if omega>0]
 *   then in spike order, per element: t_dur += (sigma_elem*t_u)*z   [if sigma_elem>0]
 *     per intra-element gap (eidx<len-1): gap += (sigma_gap*t_u)*z  [if sigma_gap>0]
 *     per inter-letter gap (after each letter): gap += (sigma_gap*t_u)*z [if sigma_gap>0]
 * NOTE sigma_e/sigma_g/t_thresh are scaled by the DRIFTED t_u (= t_u_base*omega_word), mirroring the
 * reference. `rng` must be pre-seeded by the caller; it is never dereferenced when all of
 * sigma_elem/sigma_gap/omega are 0 (so a NULL rng is safe in the jitter-off special case). */
/* ---------------------------------------------------------------------------------------------
 * E.1 change #3 — the SHARED per-word core.
 *
 * ⚠️ THIS IS A HOIST, NOT A REWRITE. The body below is encoder_render_styled_jit's ORIGINAL body,
 * statement for statement. Only three things are parameterized, all of them things a SENTENCE needs
 * and a single word cannot express:
 *    (a) `t_cursor` is in/out (a sentence continues it across words; a word starts at 0.0),
 *    (b) `last_gap_tu` is the gap nominal after the word's LAST letter (ENC_INTER_GAP == 3.0 for a
 *        single word == the original's unconditional constant; word_gap_tu for a sentence),
 *    (c) the word is a [word, word_end) RANGE, so a sentence needs no per-word copy buffer.
 *
 * BOTH paths call this — reuse, don't fork. A forked per-word math is a second implementation that
 * can drift, which is exactly what the enc_rng.h hash rule and B2b.2's router_core.h hoist exist to
 * prevent. The single-word anchor (E.0's 20/20 vs the frozen enc_render_jit) is what PROVES the
 * hoist neutral — the same way B2b.2 proved its hoist neutral by re-running gap9's gates.
 *
 * MIRRORS convergence_golden/enc_rng_ref_sentence.py:81-129 (E.0's SPEC) statement for statement.
 * --------------------------------------------------------------------------------------------- */
static inline int enc_render_word_core_jit(const char *word, const char *word_end,
                                           double wpm, double r,
                                           double sigma_elem, double sigma_gap, double omega,
                                           enc_rng_t *rng, double *t_cursor, double last_gap_tu,
                                           uint32_t *tick_us, uint8_t *ch, int n, int cap) {
    const double t_u_base = 1.2 / wpm;                         /* PARIS T_u; omega_word applied next */
    double omega_word;
    if (omega > 0.0) {                                         /* draw #0: per-word speed drift */
        double w = 1.0 + omega * enc_rng_normal(rng);         /* == rng.normal(1.0, omega) */
        omega_word = (w > 0.2) ? w : 0.2;                     /* max(0.2, .) */
    } else {
        omega_word = 1.0;                                     /* no draw (guard); == max(0.2,1.0) */
    }
    const double t_u       = t_u_base * omega_word;
    const double t_thresh  = ENC_T_THRESH * t_u;              /* 1.92 (device); matters under jitter */
    const double floor_dur = 0.1 * t_u;
    const double sigma_e   = sigma_elem * t_u;                /* scaled by DRIFTED t_u (mirror ref) */
    const double sigma_g   = sigma_gap  * t_u;
    for (const char *wp = word; wp < word_end; ++wp) {
        const char *els = enc_morse_lookup(*wp);
        if (!els) continue;
        int L = 0; while (els[L]) ++L;
        for (int e = 0; e < L; ++e) {
            double t_ideal = (els[e] == '.') ? (1.0 * t_u) : (r * t_u);
            double noise   = (sigma_e > 0.0) ? (sigma_e * enc_rng_normal(rng)) : 0.0;  /* per element */
            double t_dur   = t_ideal + noise;
            if (t_dur < floor_dur) t_dur = floor_dur;         /* max(t_ideal+noise, 0.1*T_u) */
            double t_spike = *t_cursor + t_dur;
            if (n < cap) {
                tick_us[n] = (uint32_t)lrint(t_spike * 1e6);  /* round-half-to-even == round() */
                ch[n]      = (t_dur < t_thresh) ? 0u : 1u;
                ++n;
            }
            if (e < L - 1) {                                  /* intra-element gap = 1*T_u + noise */
                double g_noise = (sigma_g > 0.0) ? (sigma_g * enc_rng_normal(rng)) : 0.0;
                double g = 1.0 * t_u + g_noise;
                if (g < floor_dur) g = floor_dur;
                *t_cursor = t_spike + g;
            } else {
                *t_cursor = t_spike;
            }
        }
        /* The gap after EVERY letter. The draw happens either way, so the RNG stream position does
         * not depend on the nominal — only the constant does (spec:119-129). `wp + 1 == word_end`
         * is the spec's `ci == len(word) - 1`. */
        double gl_noise = (sigma_g > 0.0) ? (sigma_g * enc_rng_normal(rng)) : 0.0;
        double gl = ((wp + 1 == word_end) ? last_gap_tu : ENC_INTER_GAP) * t_u + gl_noise;
        if (gl < floor_dur) gl = floor_dur;
        *t_cursor += gl;
    }
    return n;
}

static inline int encoder_render_styled_jit(const char *word, double wpm, double r,
                                            double sigma_elem, double sigma_gap, double omega,
                                            enc_rng_t *rng,
                                            uint32_t *tick_us, uint8_t *ch, int cap) {
    /* UNCHANGED BEHAVIOUR: t_cursor starts at 0.0 and the last letter's gap nominal is
     * ENC_INTER_GAP — exactly the original's unconditional `ENC_INTER_GAP * t_u + gl_noise`. */
    double t_cursor = 0.0;
    const char *end = word; while (*end) ++end;
    return enc_render_word_core_jit(word, end, wpm, r, sigma_elem, sigma_gap, omega, rng,
                                    &t_cursor, ENC_INTER_GAP, tick_us, ch, 0, cap);
}

/* E.1 change #3 — the SENTENCE render. Mirrors enc_rng_ref_sentence.enc_render_sentence_jit
 * (spec:68-130): ONE rng for the whole sentence (the caller seeds it ONCE — change #2), a t_cursor
 * that CONTINUES across words, and a `word_gap_tu`*t_u gap after each word's last letter, drifted by
 * that (OUTGOING) word's t_u because the gap is emitted inside the outgoing word's core call.
 * `text` must already be cleaned (single spaces, no leading/trailing) — enc_clean_text does that. */
static inline int encoder_render_sentence_jit(const char *text, double wpm, double r,
                                              double sigma_elem, double sigma_gap, double omega,
                                              enc_rng_t *rng, double word_gap_tu,
                                              uint32_t *tick_us, uint8_t *ch, int cap) {
    double t_cursor = 0.0;
    int n = 0;
    const char *p = text;
    while (*p) {
        while (*p == ' ') ++p;                                /* spec: [w for w in split(" ") if w] */
        if (!*p) break;
        const char *ws = p;
        while (*p && *p != ' ') ++p;
        n = enc_render_word_core_jit(ws, p, wpm, r, sigma_elem, sigma_gap, omega, rng,
                                     &t_cursor, word_gap_tu, tick_us, ch, n, cap);
    }
    return n;
}

/* Deterministic pure-ITU render (UNCHANGED public entry): r fixed to ENC_R_VAL = 3.0. Byte-identical in
 * behaviour to the previously-validated renderer, so encoder_hw / encoder_gpio / encoder_host and every
 * frozen bit-exact check stay exactly as before — the styling increment adds a path, it changes none. */
static inline int encoder_render(const char *word, double wpm,
                                 uint32_t *tick_us, uint8_t *ch, int cap) {
    return encoder_render_styled(word, wpm, ENC_R_VAL, tick_us, ch, cap);
}
#endif /* ENCODER_RENDER_H */
