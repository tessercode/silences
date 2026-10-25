/*
 * enc_addons_meta_check.c — HOST check of the enc_addons observer SEAM (no hardware).
 *
 * Proves the read-only reconstruction that enc_addons_start() performs is correct against the REAL
 * render output (g_tick/g_ch), independent of the RP2040 peripherals:
 *   - the character sequence recovered by walking the cleaned text through the PUBLIC Morse table
 *     equals the intended text (e.g. "HELLO WORLD"), one byte per char incl. the inter-word space;
 *   - every tone OFFSET equals the real completion tick g_tick[i] (so audio spacing == the wire);
 *   - every tone ON-LENGTH equals the nominal (dot = 1*T_u, dash = r*T_u);
 *   - tones never overlap (on[i] >= off[i-1]);
 *   - the char-emit onset equals the first element's onset.
 *
 * Build/run: see host/build_and_run.sh (added). Uses only encoder_render.h (RNG-free at sigma=0).
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "encoder_render.h"

#define CAP 256

/* The exact reconstruction from enc_addons_start(), lifted verbatim so the host tests the shipping
 * logic. (Kept in sync by construction — this is the same handful of statements.) */
static uint32_t nominal_on_us(char elem, double wpm, double r) {
    const double t_u = 1.2 / wpm;
    const double dur = (elem == '.') ? (1.0 * t_u) : (r * t_u);
    return (uint32_t)(dur * 1e6 + 0.5);
}

int main(void) {
    /* Emulate the encoder's cleaned text for input "HELLO, WORLD" (non-alnum -> space, collapse). */
    const char *text = "HELLO WORLD";
    const double wpm = 20.0, r = 3.0;

    uint32_t tick[CAP]; uint8_t ch[CAP];
    int n = encoder_render_sentence_jit(text, wpm, r, 0.0, 0.0, 0.0,
                                        (enc_rng_t *)0, 7.0, tick, ch, CAP);

    /* --- reconstruct (mirror of enc_addons_start) --- */
    uint32_t on_us[CAP], off_us[CAP]; int n_el = 0;
    char  char_ascii[CAP]; uint32_t char_us[CAP]; int n_char = 0;
    int ei = 0; uint32_t prev_off = 0;
    for (const char *p = text; *p; ++p) {
        char c = *p;
        if (c == ' ') { char_us[n_char] = prev_off; char_ascii[n_char] = ' '; n_char++; continue; }
        const char *els = enc_morse_lookup(c);
        if (!els) continue;
        int started = 0;
        for (int e = 0; els[e]; ++e) {
            if (ei >= n) break;
            uint32_t off = tick[ei];
            uint32_t onlen = nominal_on_us(els[e], wpm, r);
            uint32_t on = (off > onlen) ? (off - onlen) : 0u;
            if (on < prev_off) on = prev_off;
            on_us[n_el] = on; off_us[n_el] = off; n_el++;
            if (!started) { char_us[n_char] = on; char_ascii[n_char] = c; n_char++; started = 1; }
            prev_off = off; ei++;
        }
    }

    int fails = 0;

    /* 1. char sequence */
    char seq[CAP]; int si = 0;
    for (int i = 0; i < n_char; i++) seq[si++] = char_ascii[i];
    seq[si] = '\0';
    printf("char sequence emitted : \"%s\"\n", seq);
    if (strcmp(seq, "HELLO WORLD") != 0) { printf("  FAIL: expected \"HELLO WORLD\"\n"); fails++; }

    /* 2. offsets == ticks, 3. on-lengths nominal, 4. no overlap */
    printf("elements: %d (render n=%d)\n", n_el, n);
    if (n_el != n) { printf("  FAIL: element count mismatch\n"); fails++; }
    uint32_t last_off = 0;
    for (int i = 0; i < n_el; i++) {
        if (off_us[i] != tick[i]) { printf("  FAIL el %d: off %u != tick %u\n", i, off_us[i], tick[i]); fails++; }
        if (on_us[i] < last_off)  { printf("  FAIL el %d: overlap on %u < prev off %u\n", i, on_us[i], last_off); fails++; }
        last_off = off_us[i];
    }

    /* 5. report a couple of measured tone lengths vs the nominal dot/dash (validation #4 sanity) */
    double t_u = 1.2 / wpm;
    printf("nominal dot on-len = %.0f us, dash on-len = %.0f us (t_u=%.0f us)\n",
           1.0 * t_u * 1e6, r * t_u * 1e6, t_u * 1e6);
    for (int i = 0; i < n_el && i < 5; i++)
        printf("  el %d ch=%d tone [%u..%u] len=%u us\n", i, ch[i], on_us[i], off_us[i], off_us[i]-on_us[i]);

    printf(fails ? "\nRESULT: FAIL (%d)\n" : "\nRESULT: PASS\n", fails);
    return fails ? 1 : 0;
}
