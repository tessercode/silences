/*
 * encoder_gpio.c — encoder emitting the CANONICAL completion-spike GPIO signal (Change 1).
 *
 * E.1: SENTENCE-keying port of deploy/rp2040/encoder/encoder_gpio.c (READ-ONLY original). Same
 * USB-CDC console (Enter = GO) and the same renderer core — but the unit of a transmission is now a
 * SENTENCE, not a word. The six E.1 changes are marked "E.1 change #N" throughout; every other line
 * is the original. THE SPEC is convergence_golden/enc_rng_ref_sentence.py, mirrored statement for
 * statement; the E.0 golden defines the expected waveform.
 *
 * Same USB-CDC typed-text input as encoder_interactive.c (Enter = GO), same UNCHANGED renderer
 * (encoder_render.h) — but the OUTPUT is now two GPIO lines carrying BRIEF completion spikes, not a
 * 0x04 UART frame. The falling/rising edge of each brief pulse marks the canonical completion time;
 * the channel (which line) is the encoder's duration->threshold decision (T_thresh = 1.92*T_u, fixed
 * in encoder_render.h). Held-duration is GONE. Antimode/dur_ticks are out of scope (Paper 3).
 *
 * Pins:  DOT-out = GP6 -> front-end GP2 (dot),  DASH-out = GP7 -> front-end GP3 (dash), common ground.
 *   (Free pins; no conflict with GP4/GP5/GP2/GP0-1/GP23. GP4=0x04 TX and GP2=GO are unused here.)
 * A leading 400 ms HELD CAL pulse on the DOT line is the count->tick calibration reference the
 * front-end expects (it is NOT a completion spike) — ONE per TRANSMISSION (E.1 change #1), not per
 * word. USB = typed-text console; GP23 = green LED (pio1).
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>            /* atoi, strtoull */
#include <string.h>           /* strncmp */
#include "pico/stdlib.h"
#include "encoder_render.h"    /* renderer: encoder_render_styled + encoder_render_styled_jit; pulls enc_rng.h */
#include "ref_vector.h"        /* REF_WPM (the nominal/frozen wpm) */
#include "fists.h"             /* Exp-186 representative operator fists (style axes wpm, r) */
#if defined(ENC_AUDIO) || defined(ENC_CHAR_UART)
#include "enc_addons.h"        /* OUTPUT-ONLY observers (sidetone / char UART); OFF by default.
                                * Guarded so the shipping image is bit-identical when neither is defined. */
#endif

void board_identity_led(void);

#define PIN_DOT      6u        /* -> front-end GP2 (dot line) */
#define PIN_DASH     7u        /* -> front-end GP3 (dash line) */
#define MAX_SPIKES   256       /* E.1 change #6: was 64. +960 B .bss (g_tick 256->1024, g_ch 64->256).
                                * The over-cap guard (process_line) still REJECTS; it never truncates. */
#define ENC_WPM      REF_WPM   /* NOMINAL wpm (the V1-V5 frozen operator); T_thresh 1.92 in encoder_render.h */

/* E.1 change #3 — the inter-word gap nominal. gen_corpus.py:30 (WORD_GAP_TU); the encoder had NO
 * inter-word concept at all (ENC_INTER_GAP was a flat 3.0 after every letter, encoder_render.h:44).
 * Applied by encoder_render_sentence_jit after each word's LAST letter, drifted by the OUTGOING
 * word's t_u and noised with sigma_gap, exactly as enc_rng_ref_sentence.py:119-129 specifies. */
#define ENC_WORD_GAP_TU  7.0

/* E.1 change #5 — the console line bound, DERIVED.
 *
 * The old bound was MAX_WORD_LEN = 20 applied at the CONSOLE (original :209): `else if (len <
 * MAX_WORD_LEN)` silently dropped the 21st keystroke WITHOUT even echoing it, so clean_word's own
 * cap never bound and a sentence simply could not be typed.
 *
 * DERIVATION (not a guess). The fewest elements any character contributes is 1 ('E' = ".", 'T' =
 * "-"), so a text with more than MAX_SPIKES alnum characters is over-cap no matter what it holds.
 * The longest line that could still be ACCEPTED therefore interleaves MAX_SPIKES single-char words
 * with one separator each:
 *      MAX_SPIKES alnum + (MAX_SPIKES - 1) separators = 2*MAX_SPIKES - 1
 * Sizing the buffer to that means it can never truncate a line that count_spikes might accept, so
 * count_spikes stays the SINGLE authoritative guard (change #6's reject-don't-truncate behaviour is
 * preserved) and an over-length line is reported, never silently swallowed. */
#define ENC_MAX_TEXT_LEN  (2 * MAX_SPIKES - 1)          /* 511 */

/* The exp198 SHAPE the E.0 golden is drawn from (convergence_golden.md §4): 2-6 words x 2-8 letters.
 * Digits are the 5-element characters, so the shape's ceilings are DERIVED, and both are asserted
 * against the caps below — an in-shape sentence provably cannot trip either guard. */
#define ENC_SHAPE_MAX_WORDS     6
#define ENC_SHAPE_MAX_WORD_LEN  8
#define ENC_SHAPE_MAX_ELEMS     5
_Static_assert(ENC_SHAPE_MAX_WORDS * ENC_SHAPE_MAX_WORD_LEN * ENC_SHAPE_MAX_ELEMS <= MAX_SPIKES,
               "exp198 shape's mark ceiling (240) must fit MAX_SPIKES");
_Static_assert(ENC_SHAPE_MAX_WORDS * ENC_SHAPE_MAX_WORD_LEN + (ENC_SHAPE_MAX_WORDS - 1)
               <= ENC_MAX_TEXT_LEN, "exp198 shape's text ceiling (53) must fit the line buffer");
_Static_assert(ENC_MAX_TEXT_LEN >= MAX_SPIKES,
               "line buffer must never truncate a line count_spikes could accept");

/* Active operator style. g_fist == -1 -> NOMINAL (REF_WPM / r=ENC_R_VAL=3.0), which reproduces the
 * validated V1-V5 completion-spike output BYTE-FOR-BYTE (encoder_render_styled at those params == the
 * old encoder_render call). g_fist 0..ENC_N_FISTS-1 -> a representative fist (ENC_FISTS). Default is
 * NOMINAL so the live pipeline is unchanged unless the operator explicitly picks a style. STYLE-ONLY:
 * only wpm and r are applied; sigma/omega are never read; no RNG. The emit path (emit_gpio) is untouched. */
static int g_fist = -1;
static double      active_wpm(void)  { return g_fist < 0 ? (double)ENC_WPM   : ENC_FISTS[g_fist].wpm; }
static double      active_r(void)    { return g_fist < 0 ? (double)ENC_R_VAL : ENC_FISTS[g_fist].r; }
static const char *active_name(void) { return g_fist < 0 ? "nominal"         : ENC_FISTS[g_fist].name; }

/* Increment 2 — JITTER. g_jit toggles operator jitter (sigma_elem/sigma_gap/omega) via the shared
 * seedable PRNG (enc_rng.h). DEFAULT OFF, so the boot state (nominal fist, jit off) reproduces the
 * validated V1-V5 completion-spike output BYTE-FOR-BYTE. Nominal has no sigma/omega (returns 0), so
 * jitter only takes effect once a representative fist is selected AND :jit 1. Each render RE-SEEDS a
 * fresh RNG from g_seed, so (fist, word, seed) -> the same spike train (reproducible). The emit path
 * (emit_gpio), CAL, and pins are UNTOUCHED. */
static int      g_jit  = 0;
static uint64_t g_seed = 1;
static double active_se(void) { return g_fist < 0 ? 0.0 : ENC_FISTS[g_fist].sigma_elem; }
static double active_sg(void) { return g_fist < 0 ? 0.0 : ENC_FISTS[g_fist].sigma_gap; }
static double active_om(void) { return g_fist < 0 ? 0.0 : ENC_FISTS[g_fist].omega; }

/* Brief completion-spike width. Must be << the smallest gap (intra = 1*T_u ~= 33 ms at wpm 35.9) so
 * spikes never merge, and >> a DAQ sample (1 us at 1 MHz) so it's captured cleanly. 1 ms satisfies
 * both with huge margin, and a CONSTANT width cancels in the front-end's ISI differences (T̂u invariant). */
#define SPIKE_W_US       1000u
#define CAL_TICKS        400000u   /* held CAL pulse on DOT (matches ni_frontend_test.py) */
#define POST_CAL_GAP_US  20000u

static uint32_t g_tick[MAX_SPIKES];
static uint8_t  g_ch[MAX_SPIKES];

static int count_spikes(const char *word) {           /* exact spike count (reuses enc_morse_lookup) */
    int n = 0;
    for (const char *w = word; *w; ++w) {
        const char *els = enc_morse_lookup(*w);
        if (els) while (*els) { ++n; ++els; }
    }
    return n;
}
/* E.1 change #4 — non-alphanumerics become SPACE (they were DELETED).
 *
 * WAS (clean_word, original :72-82): uppercase; keep A-Z/0-9; EVERYTHING else dropped and counted in
 * `skipped`. `HELLO, WORLD` keyed as `HELLOWORLD` — ONE ten-letter word: the comma AND the space both
 * vanished, so the encoder could not express a word boundary at all.
 *
 * NOW: uppercase; keep A-Z/0-9; map every other character to a SPACE; collapse runs; strip the ends.
 * That is charlm.py:85-87's rule (`C2I.get(c, C2I[' '])`), so the encoder's alphabet and the decoder's
 * finally agree, and it mirrors enc_rng_ref_sentence.clean_text (spec:51-65), whose
 * `" ".join("".join(out).split())` is exactly collapse-runs-and-strip.
 *
 * Returns the cleaned length, or -1 if the line does not fit `outcap` (REJECT, never truncate).
 * `*mapped` counts the non-space characters that became spaces (typed spaces are not "mapped"). */
static int enc_clean_text(const char *in, int inlen, char *out, int outcap, int *mapped) {
    int w = 0, pending_space = 0, started = 0;
    *mapped = 0;
    for (int i = 0; i < inlen; i++) {
        char c = in[i];
        if (c >= 'a' && c <= 'z') c = (char)(c - 'a' + 'A');
        if (!((c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'))) {
            if (c != ' ') (*mapped)++;             /* punctuation -> SPACE (was: deleted) */
            pending_space = started;               /* leading runs never become a space (strip) */
            continue;
        }
        if (pending_space) {                       /* one separator per run (collapse) */
            if (w >= outcap) return -1;
            out[w++] = ' ';
            pending_space = 0;
        }
        if (w >= outcap) return -1;
        out[w++] = c;
        started = 1;
    }
    out[w] = '\0';                                 /* trailing run left as pending -> stripped */
    return w;
}

/* Emit the canonical signal: ONE held CAL pulse, then a brief W-wide completion spike at each
 * element's completion time (tick_us) on its channel's line. Blocks for the whole transmission;
 * lines rest LOW after.
 *
 * E.1 change #1 — THE CAL HOIST. This function ALREADY emitted exactly one CAL per call; the defect
 * was that process_line called it ONCE PER WORD, so a multi-word transmission would have carried a
 * CAL between every pair of words. The hoist is therefore STRUCTURAL: the whole sentence is now
 * rendered into ONE tick array and emitted with ONE call, so one CAL per TRANSMISSION falls out and
 * this function's body is unchanged.
 *
 * ⚠️ WHY IT IS LOAD-BEARING: the CAL is in-band and per-capture. A second CAL mid-sentence would make
 * the front end take the NEXT fragment's first ~54 ms dot as its 400 ms count->tick reference — a ~7x
 * tick-scale inflation on everything after it. Same mechanism that made B0b's mid-sentence splits
 * catastrophic. The structure matches b1_2_replay.py:20's contract exactly:
 *      [PRE_IDLE][400 ms CAL on dot/GP2][POST_CAL 20 ms][waveform][TAIL]
 * and b1_golden_reference.md:101-103 ("one 400 ms CAL pulse per transmission ... CAL is consumed by
 * reconstruction and is not a mark") — which is why E.0's Task 0 could treat the CAL as outside the
 * golden waveform. */
static void emit_gpio(const uint32_t *tick, const uint8_t *ch, int n) {
    gpio_put(PIN_DOT, 1); busy_wait_us(CAL_TICKS); gpio_put(PIN_DOT, 0);   /* HELD 400 ms CAL on dot */
    busy_wait_us(POST_CAL_GAP_US);
    uint32_t prev = 0;
    for (int i = 0; i < n; i++) {
        busy_wait_us((uint64_t)(tick[i] - prev));                          /* wait to the completion time */
        uint gp = ch[i] ? PIN_DASH : PIN_DOT;
        gpio_put(gp, 1); busy_wait_us(SPIKE_W_US); gpio_put(gp, 0);        /* brief completion spike */
        prev = tick[i] + SPIKE_W_US;
    }
    /* trailing idle: both lines LOW; the wait before the next transmission exceeds the front-end's
     * end-of-transmission threshold, so the front end sees the SENTENCE as complete. That threshold
     * is now the FIXED theta = 1.75 s (idle_clamp_split_analysis.md; B0b retired the old adaptive
     * IDLE_DONE = 500 ms this comment used to cite). E.0 measured the worst in-corpus inter-mark
     * silence at 1.095 s against it, and integrated the tail: :fist 5 splits with p = 6.0e-15 per
     * word boundary. */
}

static void print_active(void) {
    if (g_jit && g_fist >= 0)
        printf("active fist: %s wpm=%.4f r=%.4f JIT ON (sigma_e=%.4f sigma_g=%.4f omega=%.4f seed=%llu)\n",
               active_name(), active_wpm(), active_r(), active_se(), active_sg(), active_om(),
               (unsigned long long)g_seed);
    else
        printf("active fist: %s wpm=%.4f r=%.4f JIT OFF (style-only; sigma=omega=0; emit path unchanged)\n",
               active_name(), active_wpm(), active_r());
}

/* Dump the first N uint32 xoshiro outputs + N standard normals for the current seed — a SILICON check
 * that the M0+ RNG core matches the host (verify_jitter V1). Read-only; does not disturb render state. */
static void rng_dump(int N) {
    if (N < 1) N = 1; if (N > 64) N = 64;
    enc_rng_t rng; enc_rng_seed(&rng, g_seed);
    printf("RNGDUMP seed=%llu N=%d\n", (unsigned long long)g_seed, N);
    for (int i = 0; i < N; i++) printf("u32 %u\n", (unsigned)enc_rng_next(&rng));
    /* Dump raw IEEE-754 bit patterns (integer printf only, NO float formatting — avoids any
     * printf %a / float-print limitation; gives an exact bitwise device-vs-host comparison). */
    enc_rng_seed(&rng, g_seed);
    for (int i = 0; i < N; i++) {
        double u = enc_rng_unit(&rng); uint64_t b; memcpy(&b, &u, sizeof b);
        printf("uni %08lx%08lx\n", (unsigned long)(uint32_t)(b >> 32), (unsigned long)(uint32_t)b);
    }
    enc_rng_seed(&rng, g_seed);
    for (int i = 0; i < N; i++) {
        double z = enc_rng_normal(&rng); uint64_t b; memcpy(&b, &z, sizeof b);
        printf("nrm %08lx%08lx\n", (unsigned long)(uint32_t)(b >> 32), (unsigned long)(uint32_t)b);
    }
    printf("RNGDUMP END\n");
}

/* Console commands. ':fist N' selects style (0=nominal, 1..N=fist). ':jit 0|1' toggles jitter.
 * ':seed X' sets the PRNG seed (decimal or 0x-hex). ':rngdump N' dumps the RNG core for silicon V1.
 * Returns 1 if the line was a command (any ':'-prefixed line). */
static int handle_command(const char *buf, int len) {
    if (len == 0 || buf[0] != ':') return 0;
    if (len >= 5 && strncmp(buf, ":fist", 5) == 0) {
        int N = atoi(buf + 5);
        if (N == 0)                          { g_fist = -1;    printf("selected "); print_active(); }
        else if (N >= 1 && N <= ENC_N_FISTS) { g_fist = N - 1; printf("selected "); print_active(); }
        else printf("usage: :fist 0..%d  (0 = nominal, 1..%d = representative styles)\n", ENC_N_FISTS, ENC_N_FISTS);
    } else if (len >= 4 && strncmp(buf, ":jit", 4) == 0) {
        int on = atoi(buf + 4);
        g_jit = (on != 0);
        if (g_jit && g_fist < 0)
            printf("note: jitter ON but fist=nominal (sigma=omega=0) -> render stays deterministic; pick :fist 1..%d\n",
                   ENC_N_FISTS);
        printf("jitter %s. ", g_jit ? "ON" : "OFF"); print_active();
    } else if (len >= 5 && strncmp(buf, ":seed", 5) == 0) {
        g_seed = (uint64_t)strtoull(buf + 5, NULL, 0);   /* 0 base: accepts decimal or 0x-hex */
        printf("seed = %llu. ", (unsigned long long)g_seed); print_active();
    } else if (len >= 8 && strncmp(buf, ":rngdump", 8) == 0) {
        rng_dump(atoi(buf + 8));
    } else {
        printf("commands: :fist 0..%d (0=nominal) | :jit 0|1 | :seed X | :rngdump N\n", ENC_N_FISTS);
    }
    return 1;
}

/* Render a cleaned SENTENCE into g_tick/g_ch. Split out of process_line so the host harness can
 * drive the exact shipping path (clean -> guard -> render) without a console. Returns the spike
 * count, or -1 if the over-cap guard rejected it. */
static int enc_render_text(const char *text, int *n_spikes_observed) {
    int nsp = count_spikes(text);                            /* EXACT count, BEFORE render (:167) */
    *n_spikes_observed = nsp;
    if (nsp > MAX_SPIKES) return -1;                         /* change #6: REJECT, never truncate */
    double wpm = active_wpm(), r = active_r();               /* active operator STYLE (nominal or a fist) */
    if (g_jit && g_fist >= 0) {
        /* E.1 change #2 — ONE seed per SENTENCE. The original re-seeded inside the per-WORD path
         * (original :170-171), so every word of a sentence would have drawn the IDENTICAL omega and
         * the identical noise sequence: all words drifting in lockstep, unlike every golden and
         * unlike any operator. One stream per sentence == gen_corpus == enc_rng_ref_sentence:74. */
        enc_rng_t rng; enc_rng_seed(&rng, g_seed);
        return encoder_render_sentence_jit(text, wpm, r, active_se(), active_sg(), active_om(),
                                           &rng, ENC_WORD_GAP_TU, g_tick, g_ch, MAX_SPIKES);
    }
    /* jit off: sigma_elem = sigma_gap = omega = 0 -> every draw is guarded off, so the render is
     * RNG-FREE and `rng` is never dereferenced (encoder_render.h:112 states NULL is safe in exactly
     * this case). Reduces to the old style-only render, plus the 7*t_u word gaps. */
    return encoder_render_sentence_jit(text, wpm, r, 0.0, 0.0, 0.0,
                                       (enc_rng_t *)0, ENC_WORD_GAP_TU, g_tick, g_ch, MAX_SPIKES);
}

static void process_line(const char *buf, int len) {
    if (handle_command(buf, len)) return;                    /* ':' commands (e.g. :fist N) — not text */
    static char text[ENC_MAX_TEXT_LEN + 1];                  /* static: 512 B belongs in .bss, not the stack */
    int mapped;
    int tlen = enc_clean_text(buf, len, text, ENC_MAX_TEXT_LEN, &mapped);
    if (tlen < 0) { printf("line too long: > %d cleaned chars; rejected\n", ENC_MAX_TEXT_LEN); return; }
    if (mapped) printf("(mapped %d non-alnum char(s) -> SPACE)\n", mapped);
    if (tlen == 0) { printf("no valid A-Z/0-9 chars; ignored\n"); return; }
    int nsp;
    int n = enc_render_text(text, &nsp);
    if (n < 0) {
        printf("sentence too long: %s -> %d spikes > %d; rejected\n", text, nsp, MAX_SPIKES);
        return;
    }
    if (g_jit && g_fist >= 0)
        printf("emitting GPIO: %s (n_spikes=%d, fist=%s wpm=%.4f r=%.4f JIT seed=%llu "
               "sig_e=%.4f sig_g=%.4f om=%.4f, W=%uus, CAL=%ums, word_gap=%.1fTu)\n",
               text, n, active_name(), active_wpm(), active_r(), (unsigned long long)g_seed,
               active_se(), active_sg(), active_om(), (unsigned)SPIKE_W_US,
               (unsigned)(CAL_TICKS / 1000), ENC_WORD_GAP_TU);
    else
        printf("emitting GPIO: %s (n_spikes=%d, fist=%s wpm=%.4f r=%.4f JIT off, W=%uus, CAL=%ums, "
               "word_gap=%.1fTu)\n",
               text, n, active_name(), active_wpm(), active_r(), (unsigned)SPIKE_W_US,
               (unsigned)(CAL_TICKS / 1000), ENC_WORD_GAP_TU);
#if defined(ENC_AUDIO) || defined(ENC_CHAR_UART)
    /* OUTPUT-ONLY tap: hand the just-rendered schedule to the core1 observer, THEN emit. This runs
     * BEFORE the timed spike loop (never between spikes); emit_gpio below is byte-for-byte unchanged
     * and core0 walks its busy_wait chain alone. The observer reads g_tick/g_ch/text read-only. */
    enc_addons_start(text, g_tick, g_ch, n, active_wpm(), active_r());
#endif
    emit_gpio(g_tick, g_ch, n);                              /* ONE call -> ONE CAL (change #1) */
    printf("done: %s\n", text);
}

int main(void) {
    board_identity_led();                                /* green identity (pio1) */
    stdio_init_all();                                    /* USB-CDC typed-text console */

    gpio_init(PIN_DOT);  gpio_set_dir(PIN_DOT, GPIO_OUT);  gpio_put(PIN_DOT, 0);
    gpio_init(PIN_DASH); gpio_set_dir(PIN_DASH, GPIO_OUT); gpio_put(PIN_DASH, 0);

#if defined(ENC_AUDIO) || defined(ENC_CHAR_UART)
    enc_addons_init();   /* bring up the sidetone/char-UART peripherals + launch the core1 observer */
#endif

    sleep_ms(100);
    printf("RP2040_ENCODER_GPIO styled+jitter dot=GP%u dash=GP%u (wpm,r + sigma/omega jitter via seeded PRNG)\n",
           PIN_DOT, PIN_DASH);
    printf("fists: 0=nominal(REF_WPM/r=3)");
    for (int i = 0; i < ENC_N_FISTS; i++) printf(" %d=%s", i + 1, ENC_FISTS[i].name);
    printf("\n"); print_active();
    printf("commands: :fist N (0=nominal) | :jit 0|1 | :seed X | :rngdump N ; type a SENTENCE + Enter to emit\n");
    printf("(sentence: words separated by spaces; non-alnum -> SPACE; one CAL + one RNG stream per transmission;\n"
           " inter-word gap %.1f*Tu; max %d cleaned chars, max %d spikes)\n",
           ENC_WORD_GAP_TU, ENC_MAX_TEXT_LEN, MAX_SPIKES);
    printf("(boot default = nominal + jit OFF = V1-V5 completion-spike output byte-for-byte)\n");

    /* E.1 change #5 — the console gate. `b` is now ENC_MAX_TEXT_LEN (511, derived above), not
     * MAX_WORD_LEN (20), so a SENTENCE can be typed at all. static: 512 B belongs in .bss. */
    static char b[ENC_MAX_TEXT_LEN + 1];
    int len = 0, over = 0;
    while (true) {
        int c = getchar_timeout_us(1000);
        if (c == PICO_ERROR_TIMEOUT) { tight_loop_contents(); continue; }
        if (c == '\r' || c == '\n') {
            putchar('\n');
            if (over) {                                  /* NOT silent — the old code just dropped it */
                printf("line too long: > %d chars typed; rejected (nothing emitted)\n",
                       ENC_MAX_TEXT_LEN);
                over = 0; len = 0; continue;
            }
            if (len > 0) { b[len] = '\0'; process_line(b, len); len = 0; }  /* NUL-terminate: atoi/strtoull in handle_command must not read stale trailing bytes from a previous longer line */
        }
        else if (c == 8 || c == 127) { if (len > 0) { len--; printf("\b \b"); } else over = 0; }
        else if (len < ENC_MAX_TEXT_LEN) { b[len++] = (char)c; putchar(c); }
        else { over = 1; }   /* overflow is REMEMBERED and reported on Enter, never silently dropped */
    }
}
