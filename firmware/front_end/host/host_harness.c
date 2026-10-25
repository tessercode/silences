/*
 * host_harness.c — B1.1 HOST SELF-CHECK for the sentence-scale front-end port.
 *
 * Compiles the MODIFIED firmware itself (same-TU include, the anchor_vs_host_verify30 pattern) with
 * mocked pico-sdk headers (mock/), so the emit + delimiter + overflow code under test is the actual
 * frontend_stage3.c that ships to the RP2040 — not a mirror. Build with
 * -fsanitize=address,undefined (build_and_run.sh); any overrun aborts the run.
 *
 * Tests, per b1_golden_reference.md and the B1.1 gate:
 *   [T1 FRAMING]   for every golden transmission, feed the golden's integer-us edges straight into
 *                  process_and_emit() as capture counts (identity cal: 1 count = 1 us, cal_counts =
 *                  400000, so counts_to_ticks is the identity and the marks entering the framing path
 *                  ARE the golden values per golden SS6) -> byte-compare the emitted 0x06 against
 *                  expected_frame_hex. Strict: any mismatch is a port bug.
 *   [T1b RATIO]    same, at 25 counts/us (cal_counts = 10,000,000): the exact-rational counts->ticks
 *                  must reproduce identical bytes through a non-trivial CAL ratio.
 *   [T2 DELIMITER] replay each fist's full 24-transmission timeline (waveform + pacing_after_us +
 *                  trailing idle) through arm_capture/capture_vector with the mocked clock: exactly
 *                  24 captures, 1:1, every 0x06 byte-exact, all edges consumed (no orphan capture).
 *   [T2b BOOT]     boot init is IDLE_SENTENCE_US: T2 on slow_normalr already proves sentence 0 is
 *                  intact from a boot-fresh state. Negative control: force the OLD 500 ms boot init
 *                  and confirm slow_normalr transmission 0 SPLITS (>1 capture).
 *   [T3 OVERFLOW]  replay overflow_marker_case (257-mark train): exactly one 0x06-module frame, equal
 *                  to the golden KIND 0x7F marker (11 B on-wire); no data frame; sanitizers clean.
 *
 * Case files are the golden JSONs flattened to text by gen_cases.py (read-only transform; no golden
 * bytes are regenerated).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ---- include the FIRMWARE UNDER TEST (same TU; its main() renamed, never called) ---- */
#define main fe_firmware_main
#include "../frontend_stage3.c"
#include "../frontend.c"
#undef main

/* ================================ mock definitions ================================ */
struct mock_pio_inst { int dummy; } mock_pio0;
struct mock_uart_inst { int dummy; } mock_uart1;
const int lvldur_program = 0;

void gpio_init(uint pin) { (void)pin; }
void gpio_set_dir(uint pin, int dir) { (void)pin; (void)dir; }
void gpio_pull_down(uint pin) { (void)pin; }
void gpio_set_function(uint pin, int fn) { (void)pin; (void)fn; }
void stdio_init_all(void) {}
void sleep_ms(uint32_t ms) { (void)ms; }
void tight_loop_contents(void) {}
void uart_init(uart_inst_t *u, uint32_t baud) { (void)u; (void)baud; }
void uart_set_format(uart_inst_t *u, int d, int s, int p) { (void)u; (void)d; (void)s; (void)p; }
uint32_t clock_get_hz(int clk) { (void)clk; return 125000000u; }
void mock_hard_assert(int ok, const char *expr) {
    if (!ok) { fprintf(stderr, "hard_assert FAILED: %s\n", expr); exit(1); }
}
void lvldur_program_init(PIO p, uint sm, uint offset, uint pin) { (void)p; (void)sm; (void)offset; (void)pin; }
uint pio_add_program(PIO p, const void *prog) { (void)p; (void)prog; return 0; }
uint pio_claim_unused_sm(PIO p, bool required) { (void)p; (void)required; static uint next = 0; return next++; }
void pio_set_sm_mask_enabled(PIO p, uint32_t m, bool e) { (void)p; (void)m; (void)e; }
void pio_sm_clear_fifos(PIO p, uint sm) { (void)p; (void)sm; }
void pio_restart_sm_mask(PIO p, uint32_t m) { (void)p; (void)m; }
void pio_sm_exec(PIO p, uint sm, uint32_t i) { (void)p; (void)sm; (void)i; }
uint pio_encode_jmp(uint a) { (void)a; return 0; }
void board_identity_led(void) {}

/* ---- mocked clock ---- */
#define TIME_STEP_US 100
static int64_t sim_now = 0;
static int64_t sim_deadline = 0;      /* watchdog against a delimiter that never fires */
uint32_t time_us_32(void) {
    uint32_t t = (uint32_t)sim_now;
    sim_now += TIME_STEP_US;
    if (sim_deadline && sim_now > sim_deadline) {
        fprintf(stderr, "SIM DEADLINE EXCEEDED (idle break never fired?)\n"); exit(1);
    }
    return t;
}

/* ---- simulated capture SMs: per-channel edge timeline (absolute us, alternating rise/fall) ---- */
typedef struct {
    int64_t *edges; int n_edges;
    int cursor;                 /* next edge (its push) not yet consumed */
    int64_t prev_boundary;      /* start of the level run in progress */
} sm_sim_t;
static sm_sim_t sim_sm[2];      /* index == sm id: 0 = dot, 1 = dash (claim order in firmware main) */
static int64_t sim_scale = 1;   /* counts per us (identity 1; ratio test 25) */

void pio_enable_sm_mask_in_sync(PIO p, uint32_t mask) {   /* == arm: restart level runs at "now" */
    (void)p; (void)mask;
    for (int s = 0; s < 2; s++) {
        sm_sim_t *m = &sim_sm[s];
        while (m->cursor < m->n_edges && m->edges[m->cursor] <= sim_now) m->cursor++;
        if (m->cursor % 2 != 0) { fprintf(stderr, "SIM ERROR: armed mid-pulse (sm %d)\n", s); exit(1); }
        m->prev_boundary = sim_now;
    }
}
bool pio_sm_is_rx_fifo_empty(PIO p, uint sm) {
    (void)p;
    sm_sim_t *m = &sim_sm[sm];
    return !(m->cursor < m->n_edges && m->edges[m->cursor] <= sim_now);
}
uint32_t pio_sm_get(PIO p, uint sm) {
    (void)p;
    sm_sim_t *m = &sim_sm[sm];
    if (m->cursor >= m->n_edges) { fprintf(stderr, "SIM ERROR: get on empty stream\n"); exit(1); }
    int64_t v = (m->edges[m->cursor] - m->prev_boundary) * sim_scale;
    m->prev_boundary = m->edges[m->cursor];
    m->cursor++;
    return (uint32_t)v;
}

/* ---- captured UART bytes ---- */
static uint8_t ubuf[4 << 20];
static size_t ulen = 0;
void uart_write_blocking(uart_inst_t *u, const uint8_t *src, size_t len) {
    (void)u;
    if (ulen + len > sizeof ubuf) { fprintf(stderr, "uart capture overflow\n"); exit(1); }
    memcpy(ubuf + ulen, src, len); ulen += len;
}

/* ================================ case files ================================ */
#define MAX_TX 32
#define MAX_MARKS_CASE 512
typedef struct {
    int replay_seq, n_marks;
    long long pacing_after_us;
    long long rise[MAX_MARKS_CASE], fall[MAX_MARKS_CASE];
    int ch[MAX_MARKS_CASE];
    char frame_hex[8192];
} tx_case_t;
typedef struct {
    char name[64];
    int ntx;
    tx_case_t tx[MAX_TX];
} case_set_t;

static void load_cases(const char *path, case_set_t *cs) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    if (fscanf(f, "NAME %63s NTX %d", cs->name, &cs->ntx) != 2 || cs->ntx > MAX_TX) {
        fprintf(stderr, "bad case header in %s\n", path); exit(1);
    }
    for (int t = 0; t < cs->ntx; t++) {
        tx_case_t *x = &cs->tx[t];
        if (fscanf(f, " TX %d %d %lld", &x->replay_seq, &x->n_marks, &x->pacing_after_us) != 3 ||
            x->n_marks > MAX_MARKS_CASE) { fprintf(stderr, "bad TX in %s\n", path); exit(1); }
        for (int i = 0; i < x->n_marks; i++) {
            if (fscanf(f, " %lld %lld %d", &x->rise[i], &x->fall[i], &x->ch[i]) != 3) {
                fprintf(stderr, "bad edge in %s\n", path); exit(1);
            }
        }
        if (fscanf(f, " FRAME %8191s", x->frame_hex) != 1) { fprintf(stderr, "bad FRAME in %s\n", path); exit(1); }
    }
    fclose(f);
}

/* ================================ frame extraction ================================ */
/* deframe the WIDENED envelope [A5 5A][mid][len u16 BE][payload][crc8 over mid+len(2)+payload] */
typedef struct { uint8_t mid; int frame_len; size_t off; } fr_ref_t;
static int deframe_all(const uint8_t *b, size_t n, fr_ref_t *out, int max_out) {
    int cnt = 0; size_t i = 0;
    while (i + 6 <= n) {
        if (b[i] == 0xA5 && b[i + 1] == 0x5A) {
            int len = (b[i + 3] << 8) | b[i + 4];
            size_t end = i + 5 + (size_t)len;
            if (end < n + 1 && end + 1 <= n) {
                uint8_t c = 0;
                for (size_t k = i + 2; k < end; k++) {
                    c ^= b[k];
                    for (int bit = 0; bit < 8; bit++)
                        c = (c & 0x80) ? (uint8_t)((c << 1) ^ 0x07) : (uint8_t)(c << 1);
                }
                if (c == b[end]) {
                    if (cnt < max_out) { out[cnt].mid = b[i + 2]; out[cnt].frame_len = (int)(end + 1 - i); out[cnt].off = i; }
                    cnt++;
                    i = end + 1;
                    continue;
                }
            }
        }
        i++;
    }
    return cnt;
}
static void hexify(const uint8_t *b, int n, char *out) {
    for (int i = 0; i < n; i++) sprintf(out + 2 * i, "%02x", b[i]);
    out[2 * n] = 0;
}
/* return 0 on match; on mismatch print context */
static int compare_0x06(const char *label, const char *expected_hex) {
    fr_ref_t frs[64];
    int n = deframe_all(ubuf, ulen, frs, 64);
    int found = 0; static char got[8192];
    for (int i = 0; i < n; i++) {
        if (frs[i].mid != 0x06) continue;
        found++;
        hexify(ubuf + frs[i].off, frs[i].frame_len, got);
    }
    if (found != 1) { printf("FAIL %s: %d 0x06 frames (want 1)\n", label, found); return 1; }
    if (strcmp(got, expected_hex) != 0) {
        printf("FAIL %s: byte mismatch\n  want %.96s...\n  got  %.96s...\n", label, expected_hex, got);
        return 1;
    }
    return 0;
}

/* ================================ T1: direct framing ================================ */
/* Build the per-channel capture count streams for ONE transmission exactly as the capture PIO would
 * deliver them (identity origin at arm=0): dot = [idle][CAL] then (low,high) per dot mark; dash =
 * (low,high) per dash mark from t=0. Counts = us * sim_scale. Then call process_and_emit directly. */
#define T1_PRE_IDLE 20000LL
#define T1_POST_CAL 20000LL
static int t1_nd, t1_na;
static void build_counts_direct(const tx_case_t *x) {
    long long cal_rise = T1_PRE_IDLE, cal_fall = cal_rise + 400000LL;
    long long origin = cal_fall + T1_POST_CAL;      /* golden waveform t=0 in absolute us */
    int nd = 0, na = 0;
    long long prev_fall_dot = cal_fall, prev_fall_dash = 0;
    c_dot[nd++] = (uint32_t)(cal_rise * sim_scale);                 /* idle low before CAL */
    c_dot[nd++] = (uint32_t)((cal_fall - cal_rise) * sim_scale);    /* CAL high */
    for (int i = 0; i < x->n_marks; i++) {
        long long r = origin + x->rise[i], f = origin + x->fall[i];
        if (x->ch[i] == 0) {
            c_dot[nd++] = (uint32_t)((r - prev_fall_dot) * sim_scale);
            c_dot[nd++] = (uint32_t)((f - r) * sim_scale);
            prev_fall_dot = f;
        } else {
            c_dash[na++] = (uint32_t)((r - prev_fall_dash) * sim_scale);
            c_dash[na++] = (uint32_t)((f - r) * sim_scale);
            prev_fall_dash = f;
        }
        if (nd > MAXP || na > MAXP) { fprintf(stderr, "T1 case exceeds MAXP\n"); exit(1); }
    }
    t1_nd = nd; t1_na = na;
}

static int run_framing_direct(const case_set_t *cs, int scale, const char *tag) {
    int fails = 0;
    sim_scale = scale;
    for (int t = 0; t < cs->ntx; t++) {
        ulen = 0;
        build_counts_direct(&cs->tx[t]);
        process_and_emit(t1_nd, t1_na, cs->tx[t].replay_seq);
        char label[128];
        snprintf(label, sizeof label, "%s[%s] tx %d (n_marks %d)", cs->name, tag, t, cs->tx[t].n_marks);
        fails += compare_0x06(label, cs->tx[t].frame_hex);
    }
    sim_scale = 1;
    return fails;
}

/* ================================ T2: delimiter replay ================================ */
static int64_t *tl_edges[2];
static int tl_n[2];
static int64_t timeline_end;

/* build the full replay timeline: per tx, CAL on dot then the marks; next tx CAL rise =
 * previous tx last fall + pacing_after_us. First CAL rise at T1_PRE_IDLE. */
static void build_timeline(const case_set_t *cs, int ntx) {
    int cap = 4 * MAX_TX * MAX_MARKS_CASE;
    for (int s = 0; s < 2; s++) { tl_edges[s] = malloc(sizeof(int64_t) * cap); tl_n[s] = 0; }
    long long t = 0;
    for (int x = 0; x < ntx; x++) {
        const tx_case_t *tx = &cs->tx[x];
        long long cal_rise = t + T1_PRE_IDLE, cal_fall = cal_rise + 400000LL;
        long long origin = cal_fall + T1_POST_CAL;
        tl_edges[0][tl_n[0]++] = cal_rise; tl_edges[0][tl_n[0]++] = cal_fall;
        long long last_fall = cal_fall;
        for (int i = 0; i < tx->n_marks; i++) {
            int s = tx->ch[i];
            tl_edges[s][tl_n[s]++] = origin + tx->rise[i];
            tl_edges[s][tl_n[s]++] = origin + tx->fall[i];
            if (origin + tx->fall[i] > last_fall) last_fall = origin + tx->fall[i];
        }
        t = last_fall + tx->pacing_after_us;   /* driven idle, then the next transmission's CAL */
    }
    timeline_end = t;                          /* == last fall + trailing pacing (>= 2.0 s) */
}
static void free_timeline(void) { free(tl_edges[0]); free(tl_edges[1]); }

static void sim_reset(void) {
    /* mirror the firmware main()'s PIO init (main is never called on host): dot SM claimed first */
    sm_dot = 0; sm_dash = 1; pio_off = 0;
    sim_now = 0; ulen = 0;
    for (int s = 0; s < 2; s++) { sim_sm[s].edges = tl_edges[s]; sim_sm[s].n_edges = tl_n[s]; sim_sm[s].cursor = 0; sim_sm[s].prev_boundary = 0; }
    sim_deadline = timeline_end + 60 * 1000000LL;
    g_idle_us = IDLE_SENTENCE_US;              /* boot-fresh delimiter state */
}

/* the firmware main loop body, verbatim shape (arm -> capture -> tick-test on seq 0 -> process) */
static int run_replay(const case_set_t *cs, int *captures_out) {
    int fails = 0, seq = 0;
    sim_reset();
    while (seq < cs->ntx) {
        size_t ulen_before = ulen;
        arm_capture();
        int nd = 0, na = 0;
        capture_vector(&nd, &na);
        if (seq == 0) emit_tick_test();
        process_and_emit(nd, na, seq);
        /* compare only the bytes THIS capture emitted */
        size_t save = ulen; const tx_case_t *tx = &cs->tx[seq];
        memmove(ubuf, ubuf + ulen_before, ulen - ulen_before); ulen -= ulen_before;
        char label[128];
        snprintf(label, sizeof label, "%s[replay] tx %d (seq %d, n_marks %d)", cs->name, seq, tx->replay_seq, tx->n_marks);
        fails += compare_0x06(label, tx->frame_hex);
        (void)save;
        seq++;
    }
    *captures_out = seq;
    for (int s = 0; s < 2; s++) {
        if (sim_sm[s].cursor != sim_sm[s].n_edges) {
            printf("FAIL %s: channel %d has %d unconsumed edges (orphan capture material)\n",
                   cs->name, s, sim_sm[s].n_edges - sim_sm[s].cursor);
            fails++;
        }
    }
    return fails;
}

/* T2b negative control: OLD 500 ms boot init must SPLIT slow_normalr transmission 0 */
static int run_boot_negative(const case_set_t *cs) {
    build_timeline(cs, 1);                     /* timeline = transmission 0 + its pacing */
    sim_reset();
    g_idle_us = 500000;                        /* the word-scoped firmware's boot init */
    int captures = 0;
    while (sim_sm[0].cursor != tl_n[0] || sim_sm[1].cursor != tl_n[1]) {
        arm_capture();
        int nd = 0, na = 0;
        capture_vector(&nd, &na);
        process_and_emit(nd, na, captures);
        captures++;
        if (captures > 64) break;
    }
    free_timeline();
    g_idle_us = IDLE_SENTENCE_US;
    printf("  boot negative control: slow_normalr tx 0 at 500 ms init -> %d captures (expect >1: split)\n", captures);
    return captures > 1 ? 0 : 1;
}

/* ================================ main ================================ */
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <cases_dir>\n", argv[0]); return 2; }
    const char *dir = argv[1];
    static const char *FISTS[5] = { "fast_normalr", "slow_normalr", "high_r_bug",
                                    "cootie_high_jitter", "cootie_corrected" };
    static case_set_t cs;      /* large; static */
    int fails = 0, frames_exact = 0, frames_total = 0;
    char path[512];

    /* boot assertion parity with the target: exercise the sysclk hard_assert path */
    hard_assert(clock_get_hz(clk_sys) == 125000000u);

    /* T1 + T1b + T2 per fist */
    for (int f = 0; f < 5; f++) {
        snprintf(path, sizeof path, "%s/%s.txt", dir, FISTS[f]);
        load_cases(path, &cs);
        sim_deadline = 0;
        int r1 = run_framing_direct(&cs, 1, "direct");
        int r1b = run_framing_direct(&cs, 25, "ratio25");
        build_timeline(&cs, cs.ntx);
        int captures = 0;
        int r2 = run_replay(&cs, &captures);
        free_timeline();
        frames_total += 3 * cs.ntx;
        frames_exact += 3 * cs.ntx - r1 - r1b - r2;
        fails += r1 + r1b + r2;
        printf("%-20s direct %2d/%d  ratio25 %2d/%d  replay %2d/%d (captures %d/%d)\n",
               cs.name, cs.ntx - r1, cs.ntx, cs.ntx - r1b, cs.ntx, cs.ntx - r2, cs.ntx, captures, cs.ntx);
        if (strcmp(cs.name, "slow_normalr") == 0) {
            sim_deadline = 0;
            fails += run_boot_negative(&cs);
        }
    }

    /* T3: overflow marker */
    snprintf(path, sizeof path, "%s/overflow_marker_case.txt", dir);
    load_cases(path, &cs);
    sim_deadline = 0;
    build_timeline(&cs, cs.ntx);
    sim_reset();
    arm_capture();
    int nd = 0, na = 0;
    capture_vector(&nd, &na);
    process_and_emit(nd, na, 0);
    int r3 = compare_0x06("overflow_marker_case", cs.tx[0].frame_hex);
    /* the single 0x06 frame must be the 11-byte marker, i.e. no KIND 0x01 data frame at all */
    fails += r3;
    free_timeline();
    printf("overflow (257 marks): %s (marker-only, 11 B on-wire)\n", r3 ? "FAIL" : "OK");

    printf("\n%s — framing %d/%d byte-exact (per-fist direct + ratio25 + replay), fails %d\n",
           fails == 0 ? "B1.1 HOST SELF-CHECK PASS" : "B1.1 HOST SELF-CHECK **FAIL**",
           frames_exact, frames_total, fails);
    return fails == 0 ? 0 : 1;
}
