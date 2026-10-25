/*
 * B1.1 SENTENCE-SCALE PORT (Track B) of the stage-3 front-end. COPY of
 * deploy/rp2040/stage3_frontend/firmware/frontend_stage3.c (base sha256 a13e10d8…7739518);
 * the original is READ-ONLY and untouched. Change surface (b1_golden_reference.md §1-§3 +
 * transport_layer_inventory.md; full line-referenced diff in b1_1_port.md):
 *   1. DELIMITER: fixed IDLE_SENTENCE_US = 1.75 s replaces clamp(16·T̂u, 500 ms, 1.5 s);
 *      carry-forward g_idle update DISABLED (T̂u still computed + emitted per capture).
 *   2. CAPACITY: FE_MAXN 64→256 (frontend.h), MAXP 300→600; scaled buffers sized to the cap.
 *   3. FRAME WIDENING: envelope LENGTH u8→u16 BE (CRC-8 spans BOTH length bytes);
 *      0x06 n_marks u8→u16 BE. Payload max = 9 + 5·256 = 1289 B; on-wire +6.
 *   4. OVERFLOW FAIL-SAFE: >N_MARKS_MAX(256) observed marks → KIND 0x7F marker frame
 *      [0x7F][seq u16][n_marks_observed u16 sat] and the transmission is DROPPED.
 * The reconstruction, fall-time stable sort, and CAL counts→µs reference are VERBATIM from the
 * copy — the §6 count-exactness argument depends on that code being byte-for-byte unchanged.
 * No set_sys_clock* call exists here (sysclk = pico-sdk default 125 MHz; hard-asserted in main).
 *
 * ORIGINAL HEADER (word-scoped stage 3) follows:
 *
 * Stage 3 front-end test-harness firmware (THIN I/O WRAPPER around the UNCHANGED front-end).
 *
 * Receives a spike-stream vector as timed edges on GP2 (dot) / GP3 (dash), reconstructs the raw
 * per-mark timing (end_ticks, dur_ticks, ch), runs the validated front-end computation exactly as in
 * emulation (both T_u-hat estimators + framer + tick conversion, frontend.c UNCHANGED), and emits the
 * results as RESULT_FRAMEs over 8N1 UART on GP4 (uart1). Nothing in frontend.c is modified.
 *
 * Timing: two PIO state machines measure per-pin low/high run lengths at ~16 ns (2 sysclk/count),
 * started in sync so both channels share t=0. A known-width calibration pulse (the FIRST high pulse on
 * GP2) recovers counts->ticks, cancelling the 2-cycle scale AND the RP2040-vs-DAQ clock-rate mismatch.
 * The first inter-letter gap is presented stretched (~6 T_u) so the first letter segments unambiguously.
 *
 * Protocol constants are shared with the DAQ harness (ni_frontend_test.py).
 */
#include <stdio.h>
#include <stdint.h>
#include "pico/stdlib.h"
#include "hardware/pio.h"
#include "hardware/uart.h"
#include "hardware/clocks.h"   // B1.1: clock_get_hz for the boot-time sysclk assertion
#include "capture.pio.h"
#include "frontend.h"

// ---- pins ----
#define PIN_DOT   2u          // GP2  <- Dev1/port0/line0
#define PIN_DASH  3u          // GP3  <- Dev1/port0/line1
#define PIN_RESULT 4u         // GP4  -> Dev1/port1/line0 (uart1 TX)
#define RESULT_UART uart1
#define RESULT_BAUD 9600

// ---- shared protocol constants (must match the DAQ harness) ----
#define CAL_TICKS     400000  // calibration pulse width in ticks(us); first high pulse on GP2
// B1.1 END-OF-TRANSMISSION delimiter — FIXED constant (golden §1, from the clamp feasibility window
// [1.33 s, 2.0 s): max rendered inter-word gap 1.476 s < θ < T_ITX 2.0 s, margin both sides). This
// REPLACES the word-scoped clamp(16·T̂u, 500 ms, 1.5 s): that clamp's ceiling was designed so "words
// never merge" — the exact opposite of sentence scope, where a transmission's 7·T_u inter-word gaps
// must stay INSIDE one capture. The T̂u carry-forward update is disabled below; T̂u is still computed
// every capture and still emitted in the 0x06 (gap9 feature) — only the delimiter stopped consuming it.
// Boot init is the same constant (removes the slow_normalr boot transient of the old 500 ms init).
#define IDLE_SENTENCE_US 1750000
#define SEG_NUM        4      // first-letter gap threshold = SEG_NUM * tu_hat
// B1.1 capacity: >= 2·N_MARKS_MAX+2 level counts/channel (514) so a full 256-mark single-channel
// transmission fits with its (idle,CAL) pair; 600 adds headroom so the 0x7F overflow path can still
// OBSERVE a modestly-over-cap train (257 marks = 516 counts) instead of silently truncating at the cap.
#define MAXP           600    // max low/high counts captured per channel (was 300, word-scoped)
#define TICK_SCALE_CYC 1      // (documentation) counts are already the calibrated unit
// B1.1 overflow fail-safe cap (golden §3). KIND 0x7F is free in the MODULE 0x06 namespace (0x06 has
// only KIND 0x01; KINDs are per-module — 0x7F/0x7E exist only as MODULE 0x01 debug-frame kinds below).
#define N_MARKS_MAX    256
#define FE_PAYLOAD_MAX (9 + 5 * N_MARKS_MAX)   // widened 0x06: KIND+seq+tu+n_marks(u16) + 5 B/mark = 1289
_Static_assert(FE_MAXN >= N_MARKS_MAX, "mark buffers (frontend.h FE_MAXN) must hold the transport cap");
_Static_assert(MAXP >= 2 * N_MARKS_MAX + 2, "capture buffers must hold a full-cap single-channel transmission");

// tick-test inputs (Q16.16 seconds) from vectors/v1/MANIFEST.json
static const int64_t TICK_SQ[8] = {0, 39, 3932, 3965, 8091, 32768, 104890, 0};
#define N_TICK 8

static PIO pio = pio0;
static uint sm_dot, sm_dash, pio_off;
static int64_t g_idle_us = IDLE_SENTENCE_US;   // B1.1: FIXED end-of-transmission idle; never updated

// ---------------- RESULT_FRAME emit (see common/RESULT_FRAME.md) ----------------
static uint8_t crc8(const uint8_t *d, int n) {
    uint8_t c = 0;
    for (int i = 0; i < n; i++) { c ^= d[i];
        for (int b = 0; b < 8; b++) c = (c & 0x80) ? (uint8_t)((c << 1) ^ 0x07) : (uint8_t)(c << 1); }
    return c;
}
// B1.1 WIDENED envelope: [A5 5A][MODULE u8][LENGTH u16 BE][PAYLOAD][CRC8]; the CRC-8 spans
// MODULE_ID ++ BOTH LENGTH BYTES ++ PAYLOAD (golden §2 scope — a one-length-byte CRC mismatches
// every golden frame). Staging is static and sized to the cap, retiring the word-scoped firmware's
// reachable tmp[2+260] stack overflow (inventory §2.3 item 3); the guard below makes an over-cap
// payload structurally unemittable rather than silently truncated (the 0x7F path handles overflow).
static void emit_frame(uint8_t module_id, const uint8_t *payload, int len) {
    if (len < 0 || len > FE_PAYLOAD_MAX) return;       // never a wrapped LENGTH / oversize write
    uint8_t hdr[3] = {module_id, (uint8_t)(len >> 8), (uint8_t)len};
    uint8_t marker[2] = {0xA5, 0x5A};
    uart_write_blocking(RESULT_UART, marker, 2);
    uart_write_blocking(RESULT_UART, hdr, 3);
    if (len) uart_write_blocking(RESULT_UART, payload, len);
    // CRC over module_id, len_hi, len_lo, payload
    static uint8_t tmp[3 + FE_PAYLOAD_MAX];
    tmp[0] = module_id; tmp[1] = (uint8_t)(len >> 8); tmp[2] = (uint8_t)len;
    for (int i = 0; i < len; i++) tmp[3 + i] = payload[i];
    uint8_t c = crc8(tmp, 3 + len);
    uart_write_blocking(RESULT_UART, &c, 1);
}
static int put_u16(uint8_t *p, int v) { p[0] = (uint8_t)(v >> 8); p[1] = (uint8_t)v; return 2; }
static int put_i32(uint8_t *p, int32_t v) { uint32_t u = (uint32_t)v;
    p[0]=u>>24; p[1]=u>>16; p[2]=u>>8; p[3]=u; return 4; }

// ---------------- capture ----------------
static uint32_t c_dot[MAXP], c_dash[MAXP];

static void arm_capture(void) {
    uint32_t mask = (1u << sm_dot) | (1u << sm_dash);
    pio_set_sm_mask_enabled(pio, mask, false);
    pio_sm_clear_fifos(pio, sm_dot);
    pio_sm_clear_fifos(pio, sm_dash);
    pio_restart_sm_mask(pio, mask);
    pio_sm_exec(pio, sm_dot,  pio_encode_jmp(pio_off));
    pio_sm_exec(pio, sm_dash, pio_encode_jmp(pio_off));
    pio_enable_sm_mask_in_sync(pio, mask);   // both start on the same cycle -> shared t=0
}

// capture one vector's low/high counts; blocks until an idle gap marks end-of-vector.
static void capture_vector(int *n_dot, int *n_dash) {
    int nd = 0, na = 0; bool started = false; uint32_t last = time_us_32();
    while (true) {
        bool got = false;
        while (!pio_sm_is_rx_fifo_empty(pio, sm_dot) && nd < MAXP) { c_dot[nd++] = pio_sm_get(pio, sm_dot); got = true; }
        while (!pio_sm_is_rx_fifo_empty(pio, sm_dash) && na < MAXP) { c_dash[na++] = pio_sm_get(pio, sm_dash); got = true; }
        uint32_t now = time_us_32();
        if (got) { started = true; last = now; }
        else if (started && (uint32_t)(now - last) > (uint32_t)g_idle_us) break;   // T̂u-relative idle
    }
    *n_dot = nd; *n_dash = na;
}

// ---------------- reconstruct marks + run front-end + emit ----------------
static int64_t end_t[FE_MAXN], dur_t[FE_MAXN], rise_t[FE_MAXN];
static int8_t  chan[FE_MAXN];

static int64_t counts_to_ticks(int64_t counts, int64_t cal_counts) {
    // ticks = round(counts * CAL_TICKS / cal_counts)
    return (counts * (int64_t)CAL_TICKS + cal_counts / 2) / cal_counts;
}

static void process_and_emit(int n_dot, int n_dash, int seq) {
    // raw-capture debug frame FIRST (before any early return): n_dot,n_dash + first 6 raw counts each
    {
        uint8_t d[5 + 4 * 12]; int o = 0; d[o++] = 0x7F;
        o += put_u16(d + o, n_dot); o += put_u16(d + o, n_dash);
        for (int i = 0; i < 6; i++) o += put_i32(d + o, (int32_t)(i < n_dot ? c_dot[i] : -1));
        for (int i = 0; i < 6; i++) o += put_i32(d + o, (int32_t)(i < n_dash ? c_dash[i] : -1));
        emit_frame(0x01, d, o);
    }
    // calibration pulse = first HIGH on the dot channel = c_dot[1]
    if (n_dot < 2) return;
    int64_t cal_counts = c_dot[1];
    if (cal_counts <= 0) return;

    // Build marks in COUNT space: absolute falling-edge cumulative count + duration count + channel.
    // Keeping counts (not per-mark ticks) lets us round INTERVALS instead of absolute positions below.
    static int64_t cumc[FE_MAXN], durc[FE_MAXN];
    int nm = 0;
    int nm_observed = 0;    // B1.1: EVERY mark seen is counted; only the first N_MARKS_MAX are stored
    int64_t cc = 0;
    for (int k = 0; 2 * k + 1 < n_dot; k++) {          // dot channel; k=0 is (idle,cal) -> skip
        cc += c_dot[2 * k] + c_dot[2 * k + 1];          // cc = falling-edge cumulative count
        if (k == 0) continue;                           // calibration pulse (not a mark)
        nm_observed++;
        if (nm < N_MARKS_MAX) { cumc[nm]=cc; durc[nm]=c_dot[2*k+1]; chan[nm]=0; nm++; }
    }
    cc = 0;
    for (int k = 0; 2 * k + 1 < n_dash; k++) {          // dash channel; every pulse is a mark
        cc += c_dash[2 * k] + c_dash[2 * k + 1];
        nm_observed++;
        if (nm < N_MARKS_MAX) { cumc[nm]=cc; durc[nm]=c_dash[2*k+1]; chan[nm]=1; nm++; }
    }
    // B1.1 overflow fail-safe (golden §3): transmission too long -> emit the KIND 0x7F marker and DROP
    // the transmission. Never a wrapped LENGTH, never a write past the mark buffer. Also fires when the
    // raw capture saturated MAXP: counts were dropped, so the mark set is unreliable — and a saturated
    // channel implies >= (MAXP-2)/2 = 299 marks on that channel alone, over the cap regardless.
    if (nm_observed > N_MARKS_MAX || n_dot >= MAXP || n_dash >= MAXP) {
        uint8_t mpl[5]; int mo = 0;
        mpl[mo++] = 0x7F;                                          // KIND: transmission-too-long
        mo += put_u16(mpl + mo, seq);
        mo += put_u16(mpl + mo, nm_observed > 0xFFFF ? 0xFFFF : nm_observed);   // saturating u16
        emit_frame(0x06, mpl, mo);                                 // 5 B payload; 11 B on-wire
        return;
    }
    if (nm < 2) return;

    // sort marks by absolute cumulative count (falling time)
    for (int i = 1; i < nm; i++) {
        int64_t cv=cumc[i], dv=durc[i]; int8_t c=chan[i]; int j=i-1;
        while (j>=0 && cumc[j]>cv) { cumc[j+1]=cumc[j]; durc[j+1]=durc[j]; chan[j+1]=chan[j]; j--; }
        cumc[j+1]=cv; durc[j+1]=dv; chan[j+1]=c;
    }

    // Reconstruct ticks by ACCUMULATING ROUNDED INTER-MARK INTERVALS (count-differences), not by
    // rounding absolute positions. round(interval_counts * k) recovers each interval bit-exactly, so
    // the shortest-ISI estimator's input ISIs are exact and never flip the last ULP at a rounding
    // boundary (independent absolute rounding of two edges could differ by 1). Origin: first mark
    // rises at t=0 (end[0]=dur[0]) -> matches emulation, which also fixes the framer's absolute clamp.
    dur_t[0] = counts_to_ticks(durc[0], cal_counts);
    end_t[0] = dur_t[0];
    rise_t[0] = 0;
    for (int i = 1; i < nm; i++) {
        end_t[i]  = end_t[i-1] + counts_to_ticks(cumc[i] - cumc[i-1], cal_counts);
        dur_t[i]  = counts_to_ticks(durc[i], cal_counts);
        rise_t[i] = end_t[i] - dur_t[i];
    }

    // reconstructed-marks debug frame (KIND 0x7E): zeroed end/dur/ch for up to 16 marks (diagnostics)
    {
        uint8_t d[6 + 9 * 16]; int o = 0;
        d[o++] = 0x7E; o += put_i32(d + o, (int32_t)cal_counts); d[o++] = (uint8_t)nm;
        int show = nm < 16 ? nm : 16;
        for (int i = 0; i < show; i++) { o += put_i32(d+o,(int32_t)end_t[i]); o += put_i32(d+o,(int32_t)dur_t[i]); d[o++]=(uint8_t)chan[i]; }
        emit_frame(0x01, d, o);
    }

    timing_t t = { end_t, dur_t, chan, nm };
    int64_t tua = estimate_tu(FE_EST_SHORTEST_ISI, &t);   // shortest-ISI T̂u — estimator UNCHANGED

    // B1.1: the Exp-191 T̂u-relative carry-forward update of g_idle_us is DISABLED on the sentence
    // path — the delimiter is the fixed IDLE_SENTENCE_US (golden §1). tua is still computed above and
    // still emitted in the 0x06 below (gap9 consumes it); only the delimiter stopped consuming it.
    // NB: framing/segmentation DROPPED (gap9 owns it). antimode/dur_ticks are out of scope (Paper 3);
    // est_antimode is no longer called and dur_ticks is not forwarded. end_t reconstruction above is
    // UNCHANGED, so tua stays bit-exact.

    // FORWARD (MODULE_ID 0x06): raw whole-TRANSMISSION marks (completion_tick, channel) + T̂u.
    // B1.1 widened: [0x01][seq:u16][tu_shortest_isi:i32][n_marks:u16][(end_tick:i32, channel:u8) × n]
    static uint8_t pl[FE_PAYLOAD_MAX]; int o = 0;         // 1289 B, static (was 329 B on stack)
    pl[o++] = 0x01;                                       // KIND: per-capture forward record
    o += put_u16(pl + o, seq);
    o += put_i32(pl + o, (int32_t)tua);                  // shortest-ISI T̂u (µs-ticks)
    o += put_u16(pl + o, nm);                            // n marks, u16 BE (B1.1 widening; was u8)
    for (int i = 0; i < nm; i++) { o += put_i32(pl + o, (int32_t)end_t[i]); pl[o++] = (uint8_t)chan[i]; }
    emit_frame(0x06, pl, o);                             // front-end -> gap9 contract (marks + T̂u)
}

static void emit_tick_test(void) {
    uint8_t pl[2 + 4 * N_TICK]; int o = 0;
    pl[o++] = 0x02; pl[o++] = (uint8_t)N_TICK;
    for (int i = 0; i < N_TICK; i++) o += put_i32(pl + o, (int32_t)fe_tick_convert(TICK_SQ[i]));
    emit_frame(0x01, pl, o);
}

/* Identity LED (blue). Defined in frontend_led.c — the ONLY TU that includes status_led.h, kept
 * separate so the shared MODULE_* enum can never collide with any frame-id macro, and so this
 * validated compute file is only touched to ADD the one boot-time call below. */
void board_identity_led(void);

int main(void) {
    stdio_init_all();                                 // uart0 on GP0/GP1 (debug sentinel only)

    // B1.1 sysclk assertion: the golden's count-exactness argument (b1_golden_reference.md §6)
    // assumes the pico-sdk default 125 MHz sysclk (16 ns per 2-cycle capture-PIO count at clkdiv 1.0).
    // This firmware makes no set_sys_clock* call; assert the default actually holds at boot.
    hard_assert(clock_get_hz(clk_sys) == 125000000u);

    // identity LED: blue = front-end. One-time boot write on GP23 via pio1 (the capture PIO is pio0
    // SM_dot/SM_dash; no overlap), BEFORE the DAQ-synchronised capture loop -> not in the timed path.
    board_identity_led();

    // result UART on GP4 = uart1 TX
    uart_init(RESULT_UART, RESULT_BAUD);
    gpio_set_function(PIN_RESULT, GPIO_FUNC_UART);
    uart_set_format(RESULT_UART, 8, 1, UART_PARITY_NONE);

    // pull-downs so idle (undriven-between-stimuli) reads low -> no phantom marks from floating pins;
    // the DAQ's push-pull 3.3 V actively overrides the weak pull while presenting a vector.
    gpio_init(PIN_DOT);  gpio_set_dir(PIN_DOT, GPIO_IN);  gpio_pull_down(PIN_DOT);
    gpio_init(PIN_DASH); gpio_set_dir(PIN_DASH, GPIO_IN); gpio_pull_down(PIN_DASH);

    pio_off = pio_add_program(pio, &lvldur_program);
    sm_dot  = pio_claim_unused_sm(pio, true);
    sm_dash = pio_claim_unused_sm(pio, true);
    lvldur_program_init(pio, sm_dot,  pio_off, PIN_DOT);
    lvldur_program_init(pio, sm_dash, pio_off, PIN_DASH);

    sleep_ms(100);
    printf("RP2040_B1_FE_SENTENCE_OK\n");             // debug sentinel on uart0 (B1.1 port identity)

    int seq = 0;
    while (true) {
        arm_capture();
        int nd = 0, na = 0;
        capture_vector(&nd, &na);
        if (seq == 0) emit_tick_test();               // first vector's capture carries the tick-test frame
        process_and_emit(nd, na, seq);
        seq++;
    }
}
