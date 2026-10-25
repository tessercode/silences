/*
 * enc_host_check.c — E.1 HOST GATE for the sentence-keying encoder port.
 *
 * Compiles the SHIPPING firmware itself (same-TU include, the A2.1/A4.1/B1.1 pattern) with mocked
 * pico-sdk headers (mock/), so what is gated is the actual encoder_gpio.c + encoder_render.h +
 * enc_rng.h + enc_clean_text that go to the RP2040 — not a mirror of them.
 *
 * THE GATE (Task 3): for every E.0 golden entry, feed (seed, fist, text) through the shipping path
 *   enc_clean_text -> count_spikes guard -> encoder_render_sentence_jit
 * and compare the rendered array against the expected waveform: us-tick AND channel, EXACT, no
 * tolerance. If the waveform matches, every downstream byte in the golden (0x06/0x03/0x05/decode)
 * follows by construction — E.0 composed them through the frozen mirrors from exactly this input.
 *
 * ⚠️ PROVENANCE (see e1_port.md §3): 36 of the 64 golden entries were rendered by E.0 from fist
 * parameters that match NO device and no other artifact (E.0's table is the Exp-52 calibration
 * distribution, not fists.h). gen_cases.py tags each case:
 *    GOLDEN (28) — expected waveform is the FROZEN golden's, asserted bit-identical. The real gate.
 *    SPEC   (36) — expected waveform re-derived from E.0's own spec at the CORRECT fists.h params.
 * Both are reported separately. The headline claim rests on GOLDEN only.
 *
 *   [T1 WAVEFORM]  all 64 cases: tick + channel exact.
 *   [T2 STRESS]    the four stress picks called out by name.
 *   [T3 ANCHOR]    the ported single-word render == the FROZEN enc_render_jit (E.0's 20/20). A break
 *                  here means change #3 leaked into the per-word math.
 *   [T4 RNGDUMP]   the COPIED enc_rng.h stream == enc_rng_ref's, bitwise (u32 + unit + normal).
 *                  This is copy==original at the behavioural level, complementing the sha256.
 *   [T5 GUARDS]    over-cap rejects (never truncates); clean_text maps/collapses/strips; the console
 *                  line bound never silently drops.
 *   [T6 EMIT]      the emit schedule is monotonic and never underflows busy_wait_us.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>   /* fabs — T4b's toleranced r_max check */

/* ---- include the FIRMWARE UNDER TEST (same TU; its main() renamed, never called) ---- */
#define main encoder_gpio_fw_main
#include "../encoder_gpio.c"
#undef main

/* ================================ mock definitions ================================ */
static uint64_t g_sim_us;          /* busy_wait_us accumulates here instead of blocking */
static int g_emit_underflow;       /* set if a wait was computed from an out-of-order tick */

void gpio_init(uint pin) { (void)pin; }
void gpio_set_dir(uint pin, int dir) { (void)pin; (void)dir; }
void gpio_put(uint pin, int value) { (void)pin; (void)value; }
void stdio_init_all(void) {}
void sleep_ms(uint32_t ms) { (void)ms; }
void tight_loop_contents(void) {}
void busy_wait_us(uint64_t us) {
    if (us > (uint64_t)1e12) g_emit_underflow = 1;   /* a wrapped uint32 subtraction lands here */
    g_sim_us += us;
}
int getchar_timeout_us(uint32_t us) { (void)us; return PICO_ERROR_TIMEOUT; }
void board_identity_led(void) {}

/* ================================ case loading ================================ */
#define MAXN 512
typedef struct {
    int vid, fist_index, console_fist, n;
    unsigned long long seed;
    char kind[32], tag[16], text[600];
    uint32_t tick[MAXN];
    uint8_t  ch[MAXN];
} case_t;

static int n_fail_total;

static void die(const char *m) { fprintf(stderr, "FATAL: %s\n", m); exit(2); }

static char *slurp(const char *p) {
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p); exit(2); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc(sz + 1);
    if (fread(b, 1, sz, f) != (size_t)sz) die("short read");
    b[sz] = 0; fclose(f);
    return b;
}

/* ================================ T1/T2: the waveform gate ================================ */
static int run_case(const case_t *c, int verbose) {
    /* drive the SHIPPING globals exactly as the console would: :fist N, :jit 1, :seed X */
    g_fist = c->fist_index;
    g_jit = 1;
    g_seed = c->seed;

    static char cleaned[ENC_MAX_TEXT_LEN + 1];
    int mapped;
    int tlen = enc_clean_text(c->text, (int)strlen(c->text), cleaned, ENC_MAX_TEXT_LEN, &mapped);
    if (tlen < 0) { printf("  vid %d: clean_text REJECTED (line too long)\n", c->vid); return 0; }

    int nsp = 0;
    int n = enc_render_text(cleaned, &nsp);

    if (c->n > MAX_SPIKES) {                     /* the over-cap case: the guard must REJECT */
        if (n >= 0) { printf("  vid %d: over-cap (%d spikes) NOT rejected\n", c->vid, nsp); return 0; }
        if (nsp != c->n) { printf("  vid %d: guard counted %d, expected %d\n", c->vid, nsp, c->n); return 0; }
        if (verbose) printf("  vid %-3d %-20s REJECTED at %d spikes > %d (no truncation, nothing emitted)\n",
                            c->vid, c->kind, nsp, MAX_SPIKES);
        return 1;
    }
    if (n < 0) { printf("  vid %d: rejected but expected %d spikes\n", c->vid, c->n); return 0; }
    if (n != c->n) { printf("  vid %d: n_spikes %d != expected %d\n", c->vid, n, c->n); return 0; }

    for (int i = 0; i < n; i++) {
        if (g_tick[i] != c->tick[i] || g_ch[i] != c->ch[i]) {
            printf("  vid %-3d %-8s MISMATCH at spike %d: got (%" PRIu32 ",%u) want (%" PRIu32 ",%u)\n",
                   c->vid, c->tag, i, g_tick[i], g_ch[i], c->tick[i], c->ch[i]);
            return 0;
        }
    }
    /* T6: the emit schedule must be monotonic — run the real emit_gpio against the mocked clock. */
    g_sim_us = 0; g_emit_underflow = 0;
    emit_gpio(g_tick, g_ch, n);
    if (g_emit_underflow) { printf("  vid %d: emit_gpio wait UNDERFLOW (non-monotonic ticks)\n", c->vid); return 0; }
    if (verbose) printf("  vid %-3d %-20s %-6s n=%-3d OK\n", c->vid, c->kind, c->tag, n);
    return 1;
}

static void gate_waveform(void) {
    char *b = slurp("cases.txt");
    char *s = b;
    int N = (int)strtol(s, &s, 10);
    int okg = 0, ng = 0, oks = 0, ns = 0;
    case_t c;
    printf("\n[T1 WAVEFORM] %d golden entries: tick + channel EXACT, no tolerance\n", N);
    for (int k = 0; k < N; k++) {
        memset(&c, 0, sizeof c);
        c.vid = (int)strtol(s, &s, 10);
        sscanf(s, "%31s %15s %d %d %llu %d", c.kind, c.tag, &c.fist_index, &c.console_fist,
               &c.seed, &c.n);
        for (int f = 0; f < 6; f++) { while (*s == ' ') s++; while (*s && *s != ' ' && *s != '|') s++; }
        while (*s && *s != '|') s++;
        s++;                                          /* opening | */
        char *e = strchr(s, '|');
        if (!e) die("bad text field");
        memcpy(c.text, s, e - s); c.text[e - s] = 0;
        s = e + 1;
        if (c.n > MAXN) die("case exceeds MAXN");
        for (int i = 0; i < c.n; i++) c.tick[i] = (uint32_t)strtoul(s, &s, 10);
        for (int i = 0; i < c.n; i++) c.ch[i] = (uint8_t)strtol(s, &s, 10);

        int ok = run_case(&c, 0);
        int is_g = (strcmp(c.tag, "GOLDEN") == 0);
        if (is_g) { ng++; okg += ok; } else { ns++; oks += ok; }
        if (!ok) n_fail_total++;
    }
    printf("  GOLDEN (frozen E.0 waveform, params == fists.h) : %d/%d\n", okg, ng);
    printf("  SPEC   (re-derived at fists.h params; E.0 golden unreachable — e1_port.md §3): %d/%d\n",
           oks, ns);
    printf("  TOTAL  : %d/%d\n", okg + oks, ng + ns);
    free(b);
}

static void gate_stress(void) {
    char *b = slurp("cases.txt");
    char *s = b;
    int N = (int)strtol(s, &s, 10);
    case_t c;
    printf("\n[T2 STRESS] the four deterministic stress picks\n");
    for (int k = 0; k < N; k++) {
        memset(&c, 0, sizeof c);
        c.vid = (int)strtol(s, &s, 10);
        sscanf(s, "%31s %15s %d %d %llu %d", c.kind, c.tag, &c.fist_index, &c.console_fist,
               &c.seed, &c.n);
        while (*s && *s != '|') s++;
        s++;
        char *e = strchr(s, '|');
        memcpy(c.text, s, e - s); c.text[e - s] = 0;
        s = e + 1;
        for (int i = 0; i < c.n; i++) c.tick[i] = (uint32_t)strtoul(s, &s, 10);
        for (int i = 0; i < c.n; i++) c.ch[i] = (uint8_t)strtol(s, &s, 10);
        if (strncmp(c.kind, "stress", 6) == 0) {
            printf("  %-22s %-6s text=%-30s ", c.kind, c.tag, c.text);
            int ok = run_case(&c, 0);
            printf("%s\n", ok ? "PASS" : "FAIL");
            if (!ok) n_fail_total++;
        }
    }
    free(b);
}

/* ================================ T3: the single-word anchor ================================ */
static void gate_anchor(void) {
    FILE *f = fopen("anchor_cases.txt", "r");
    if (!f) die("cannot open anchor_cases.txt");
    int N, fi;
    if (fscanf(f, "%d %d", &N, &fi) != 2) die("bad anchor header");
    int ok = 0, checked = 0;
    static uint32_t tk[MAXN]; static uint8_t ch[MAXN];
    printf("\n[T3 ANCHOR] ported single-word render == FROZEN enc_render_jit (change #3 must not leak)\n");
    for (int k = 0; k < N; k++) {
        char word[64]; unsigned long long sd; int n;
        if (fscanf(f, "%63s %llu %d", word, &sd, &n) != 3) die("bad anchor case");
        if (n > MAXN) die("anchor case exceeds MAXN");
        enc_rng_t rng; enc_rng_seed(&rng, sd);
        int got = encoder_render_styled_jit(word, ENC_FISTS[fi].wpm, ENC_FISTS[fi].r,
                                            ENC_FISTS[fi].sigma_elem, ENC_FISTS[fi].sigma_gap,
                                            ENC_FISTS[fi].omega, &rng, tk, ch, MAXN);
        int good = (got == n);
        for (int i = 0; i < n; i++) {
            unsigned long wt;
            if (fscanf(f, "%lu", &wt) != 1) die("bad anchor tick");
            if (i < got && tk[i] != (uint32_t)wt) good = 0;
        }
        for (int i = 0; i < n; i++) {
            int wc;
            if (fscanf(f, "%d", &wc) != 1) die("bad anchor ch");
            if (i < got && ch[i] != (uint8_t)wc) good = 0;
        }
        if (!good) printf("  %s seed=%llu MISMATCH vs frozen enc_render_jit (n got %d want %d)\n",
                          word, sd, got, n);
        ok += good; checked++;
    }
    fclose(f);
    if (checked != N) die("anchor: not every case was read");
    printf("  %d/%d single-word renders bit-identical to the frozen reference (fist %s)\n",
           ok, N, ENC_FISTS[fi].name);
    if (ok != N) n_fail_total++;
}

/* ================================ T4: :rngdump parity ================================ */
static void gate_rng(void) {
    char *b = slurp("rng_cases.txt");
    char *s = b;
    int NS = (int)strtol(s, &s, 10);
    int N = (int)strtol(s, &s, 10);
    int ok_u32 = 0, ok_uni = 0, ok_nrm = 0, tot = 0;
    printf("\n[T4 RNGDUMP] the COPIED enc_rng.h stream vs enc_rng_ref, BITWISE (copy == original)\n");
    for (int k = 0; k < NS; k++) {
        unsigned long long sd = strtoull(s, &s, 10);
        enc_rng_t rng;
        enc_rng_seed(&rng, sd);
        for (int i = 0; i < N; i++) {
            uint32_t want = (uint32_t)strtoul(s, &s, 10);
            ok_u32 += (enc_rng_next(&rng) == want);
        }
        enc_rng_seed(&rng, sd);
        for (int i = 0; i < N; i++) {
            while (*s == ' ' || *s == '\n') s++;
            char hex[17]; memcpy(hex, s, 16); hex[16] = 0; s += 16;
            uint64_t want = strtoull(hex, NULL, 16);
            double u = enc_rng_unit(&rng); uint64_t got; memcpy(&got, &u, sizeof got);
            ok_uni += (got == want);
        }
        enc_rng_seed(&rng, sd);
        for (int i = 0; i < N; i++) {
            while (*s == ' ' || *s == '\n') s++;
            char hex[17]; memcpy(hex, s, 16); hex[16] = 0; s += 16;
            uint64_t want = strtoull(hex, NULL, 16);
            double z = enc_rng_normal(&rng); uint64_t got; memcpy(&got, &z, sizeof got);
            ok_nrm += (got == want);
        }
        tot += N;
    }
    printf("  u32    %d/%d\n  unit   %d/%d (IEEE-754 bit patterns)\n  normal %d/%d (Box-Muller)\n",
           ok_u32, tot, ok_uni, tot, ok_nrm, tot);
    if (ok_u32 != tot || ok_uni != tot || ok_nrm != tot) n_fail_total++;
    free(b);
}

/* ================================ T4b: the EXTERNAL anchor ================================
 * T4 above proves the COPIED enc_rng.h matches enc_rng_ref.py. But those two were written in one
 * commit, one a stated line-for-line port of the other (enc_rng_audit.md §1) — so T4, E.2's 8/8 on
 * silicon and the 64/64 render all compare a file to its own port, and would stay green even if both
 * had been written from a shared misreading of the algorithm.
 *
 * This gate asserts the SHIPPING enc_rng.h against vectors derived from VIGNA'S PUBLISHED
 * xoshiro128** 1.1 (enc_rng_kat/, transcribed from the sha256-pinned vendored source).
 * >>> A FAILURE HERE MEANS THE IMPLEMENTATION IS WRONG. <<<
 *
 * KAT_NORMAL_BITS is deliberately NOT asserted as an anchor: no published vectors exist for the
 * Box-Muller, so it is a REGRESSION FREEZE (a failure means "something changed", never "wrong").
 * It is checked as such, and labelled as such, below.
 */
#include "../../tests/kat/enc_rng_kat.h"

static void gate_kat(void) {
    printf("\n[T4b KAT] the SHIPPING enc_rng.h vs VIGNA'S PUBLISHED algorithm (external anchor)\n");
    int n_state = 0, n_stream = 0, n_fold = 0, fails = 0;
    for (int i = 0; i < KAT_N_SEEDS; i++) {
        enc_rng_t r;
        enc_rng_seed(&r, KAT_SEED[i]);
        /* the published source: "The state must be seeded so that it is not everywhere zero." */
        if (!(r.s[0] | r.s[1] | r.s[2] | r.s[3])) {
            printf("  KAT[seed=%llu]: ALL-ZERO xoshiro state\n", (unsigned long long)KAT_SEED[i]);
            fails++; continue;
        }
        for (int k = 0; k < 4; k++) {
            if (r.s[k] != KAT_STATE[i][k]) {
                printf("  KAT_STATE[seed=%llu][%d]: SplitMix64 seed expansion != the published "
                       "algorithm — got 0x%08X, want 0x%08X\n",
                       (unsigned long long)KAT_SEED[i], k, r.s[k], KAT_STATE[i][k]);
                fails++;
            }
        }
        n_state++;
        for (int j = 0; j < KAT_N_STREAM; j++) {
            uint32_t got = enc_rng_next(&r);
            if (got != KAT_STREAM[i][j]) {
                printf("  KAT_STREAM[seed=%llu][%d]: xoshiro128** stream != the published algorithm "
                       "— got 0x%08X, want 0x%08X\n",
                       (unsigned long long)KAT_SEED[i], j, got, KAT_STREAM[i][j]);
                fails++; break;
            }
        }
        n_stream++;
        enc_rng_seed(&r, KAT_SEED[i]);
        uint32_t fs = 0, fx = 0;
        for (int j = 0; j < KAT_N_FOLD; j++) { uint32_t v = enc_rng_next(&r); fs += v; fx ^= v; }
        if (fs != KAT_FOLD_SUM[i] || fx != KAT_FOLD_XOR[i]) {
            printf("  KAT_FOLD[seed=%llu]: the %d-draw fold diverges (the first %d draws matched, so "
                   "the stream diverges LATER) — sum 0x%08X/0x%08X xor 0x%08X/0x%08X\n",
                   (unsigned long long)KAT_SEED[i], KAT_N_FOLD, KAT_N_STREAM,
                   fs, KAT_FOLD_SUM[i], fx, KAT_FOLD_XOR[i]);
            fails++;
        }
        n_fold++;
    }
    printf("  ANCHOR : %d seed states, %dx%d u32, %dx%d-draw folds (%d draws) — %s\n",
           n_state, n_stream, KAT_N_STREAM, n_fold, KAT_N_FOLD, n_fold * KAT_N_FOLD,
           fails ? "*** FAIL ***" : "EXACT vs the published algorithm");

    /* ---- REGRESSION FREEZE: internal truth only. A failure means SOMETHING CHANGED, not "wrong". */
    int n_nrm = 0;
    for (int i = 0; i < KAT_N_SEEDS; i++) {
        enc_rng_t r;
        enc_rng_seed(&r, KAT_SEED[i]);
        for (int j = 0; j < KAT_N_NORMAL; j++) {
            double z = enc_rng_normal(&r);
            uint64_t bits;
            memcpy(&bits, &z, sizeof bits);
            if (bits != KAT_NORMAL_BITS[i][j]) {
                printf("  KAT_NORMAL_BITS[seed=%llu][%d]: the Box-Muller output CHANGED (freeze, "
                       "NOT an anchor — this means something changed, not that it is wrong)\n",
                       (unsigned long long)KAT_SEED[i], j);
                fails++; break;
            }
            n_nrm++;
        }
    }
    printf("  FREEZE : %d Box-Muller normals bit-identical to the 2026-07-17 freeze "
           "(internal truth; no published vectors exist)\n", n_nrm);

    /* ---- DERIVED: the support bound, vs its derivation sqrt(2*33*ln 2). Toleranced, not
     * bit-compared — the closed form and libm's path differ by 1 ULP. */
    double u1_min = 0.5 / 4294967296.0;
    double r_tree = enc_sqrt(-2.0 * enc_log(u1_min));
    double rmax;
    uint64_t rb = KAT_RMAX_BITS;
    memcpy(&rmax, &rb, sizeof rmax);
    if (!(fabs(r_tree - rmax) <= 2.0 * 8.881784197001252e-16)) {
        printf("  KAT_RMAX: support bound moved — got %.17g, want %.17g (= sqrt(2*33*ln 2))\n",
               r_tree, rmax);
        fails++;
    }
    printf("  DERIVED: support bound |z| <= %.6f = sqrt(2*33*ln 2) <- u1_min = 0.5/2^32 = 2^-33\n", rmax);
    if (fails) n_fail_total++;
}

/* ================================ T5: the guards ================================ */
static void gate_guards(void) {
    static char out[ENC_MAX_TEXT_LEN + 1];
    int mapped, fails = 0;
    printf("\n[T5 GUARDS] clean_text + the over-cap guard\n");

    struct { const char *in, *want; int mapped; } tv[] = {
        { "HELLO, WORLD",     "HELLO WORLD", 1 },   /* change #4: the comma becomes the boundary */
        { "hello world",      "HELLO WORLD", 0 },
        { "  CQ   DE  LUKE ", "CQ DE LUKE",  0 },   /* collapse + strip */
        { "A--B",             "A B",         2 },   /* a RUN of punctuation is ONE separator */
        { "...",              "",            3 },
        { "K7QX/P",           "K7QX P",      1 },
        { "",                 "",            0 },
    };
    for (size_t i = 0; i < sizeof tv / sizeof tv[0]; i++) {
        int n = enc_clean_text(tv[i].in, (int)strlen(tv[i].in), out, ENC_MAX_TEXT_LEN, &mapped);
        int ok = (n >= 0 && strcmp(out, tv[i].want) == 0 && mapped == tv[i].mapped);
        printf("  %-20s -> %-14s mapped=%d  %s\n", tv[i].in, out, mapped, ok ? "OK" : "FAIL");
        fails += !ok;
    }
    /* the over-cap guard rejects and does NOT truncate */
    static char big[ENC_MAX_TEXT_LEN + 1];
    memset(big, 'E', 300); big[300] = 0;          /* 300 x 'E' = 300 spikes > MAX_SPIKES */
    int nsp = 0;
    int n = enc_render_text(big, &nsp);
    int ok = (n < 0 && nsp == 300);
    printf("  over-cap 300 x 'E': n=%d nsp=%d -> %s (reject, no truncation)\n", n, nsp,
           ok ? "OK" : "FAIL");
    fails += !ok;

    /* a line longer than the buffer is REJECTED, not truncated into a valid-looking one */
    static char huge[4096];
    memset(huge, 'E', 4000); huge[4000] = 0;
    n = enc_clean_text(huge, 4000, out, ENC_MAX_TEXT_LEN, &mapped);
    ok = (n == -1);
    printf("  4000-char line: clean_text -> %d -> %s (rejected, never silently truncated)\n", n,
           ok ? "OK" : "FAIL");
    fails += !ok;
    if (fails) n_fail_total++;
}

/* ================================ T0: the negative control ================================
 * The E.0 lesson: a gate that cannot fail proves nothing. Perturb each input the waveform gate
 * depends on by the smallest amount that matters and demand a MISMATCH. If any of these "passes",
 * T1's 64/64 is decoration. */
static int nc_render_matches(int fist_index, unsigned long long seed, const char *text,
                             const uint32_t *want_tick, const uint8_t *want_ch, int want_n) {
    g_fist = fist_index; g_jit = 1; g_seed = seed;
    int nsp = 0;
    int n = enc_render_text(text, &nsp);
    if (n != want_n) return 0;
    for (int i = 0; i < n; i++)
        if (g_tick[i] != want_tick[i] || g_ch[i] != want_ch[i]) return 0;
    return 1;
}

static void gate_negative_control(void) {
    /* take a known-good GOLDEN case: the single-word stress pick, :fist 5 (index 4), seed 9001 */
    const int fi = 4;
    const char *text = "PARAMOUNT";
    static uint32_t tick[MAXN]; static uint8_t ch[MAXN];
    g_fist = fi; g_jit = 1; g_seed = 9001;
    int nsp = 0;
    int n = enc_render_text(text, &nsp);
    memcpy(tick, g_tick, sizeof(uint32_t) * n);
    memcpy(ch, g_ch, sizeof(uint8_t) * n);

    printf("\n[T0 NEGATIVE CONTROL] the waveform gate must FAIL when its input is perturbed\n");
    int bad = 0;
    struct { const char *what; int matches; } t[] = {
        { "baseline (unperturbed) must MATCH",   nc_render_matches(fi, 9001, text, tick, ch, n) },
        { "seed 9001 -> 9002",                   nc_render_matches(fi, 9002, text, tick, ch, n) },
        { "fist 5 -> fist 4 (cootie_high_jitter, omega 0.3774 -> 0.7122)",
                                                 nc_render_matches(3, 9001, text, tick, ch, n) },
        { "text PARAMOUNT -> PARAMOUNS",         nc_render_matches(fi, 9001, "PARAMOUNS", tick, ch, n) },
        { "one word -> two (PARA MOUNT: the 7*Tu gap must move every later tick)",
                                                 nc_render_matches(fi, 9001, "PARA MOUNT", tick, ch, n) },
    };
    for (size_t i = 0; i < sizeof t / sizeof t[0]; i++) {
        int want_match = (i == 0);
        int ok = (t[i].matches == want_match);
        printf("  %-62s %-9s %s\n", t[i].what, t[i].matches ? "MATCHES" : "differs",
               ok ? "OK" : (want_match ? "*** BASELINE BROKEN ***" : "*** VACUOUS GATE ***"));
        bad += !ok;
    }
    if (bad) n_fail_total++;
}

int main(void) {
    printf("========================================================================\n");
    printf("E.1 HOST GATE — sentence-keying encoder, SHIPPING TU compiled in-harness\n");
    printf("========================================================================\n");
    gate_negative_control();
    gate_rng();          /* copy == original FIRST: everything below rests on it */
    gate_kat();          /* ...and copy == THE PUBLISHED ALGORITHM: the external anchor (T4b) */
    gate_anchor();
    gate_waveform();
    gate_stress();
    gate_guards();
    printf("\n%s\n", n_fail_total == 0 ? "ALL GATES PASS" : "*** FAILURES ***");
    return n_fail_total ? 1 : 0;
}
