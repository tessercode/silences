/*
 * gap9_hw.c — B2b.1 SENTENCE-TREE RX wrapper (THIN I/O around the UNCHANGED 3-class segmenter).
 *
 * COPIED from deploy/rp2040/gap9_hardware/firmware/gap9_hw.c (sha256 94d5c474ec806881...) and
 * modified; the read-only original is untouched. The module it wraps (gap9_3c.c/.h + its LUTs and
 * weights) is A2's, silicon-validated 97/97 at n1 and n5 — NOT touched here. B2b.1's surface is
 * only the transport around it: RX assembly, the 0x06 parse, and the widened 0x03 emit.
 *
 * DIFF vs the copied original (every change is forced by a decided constant; nothing else moved):
 *   1. module: gap9_mod.h/vectors.h  -> gap9_3c.h/vectors_3c.h; the 3c entries are argmax-free
 *      (`clf_run(vi, lp3)` / `clf_run_marks(ticks, ch, ns, tut, lp3)`) so `arg[]` is GONE.
 *   2. envelope: LENGTH u8 -> u16 BE; the CRC spans MODULE_ID ++ BOTH LENGTH BYTES ++ PAYLOAD
 *      (:69-79 -> emit_frame below). CRC staging is static and cap-sized (was `tmp[2+200]`).
 *   3. 0x03: n_gaps u8 -> u16; DECISION-BITS FIELD DELETED (D4) and with it the `+4` in the pl
 *      declaration (see emit_result_0x03); per-gap 1 x i32 -> 3 x i32 BE.
 *   4. 0x06 parse: n_marks u8 -> u16 (B1 emits u16; the shipped u8 read would desync).
 *   5. INT8 variant dropped: D1 pinned Q16.16; the phase2c build has no 3-class module.
 *
 * The BYTES are defined by the FROZEN B2.0 golden (b2_frame_golden/golden_frames_n{1,5}.json);
 * this file mirrors b2_frame_golden/frames_b2.py statement-for-statement.
 *
 * Wiring (unchanged from the original):
 *   GP2 <- GO trigger   GP3 <- 0x06 RX (PIO)   GP4 -> uart1 TX 0x03   GP5 -> latency marker
 *   GP0/GP1 = uart0 debug sentinel.  GP23/GP24 = board RGB/button, DO NOT TOUCH.
 */
#include <stdio.h>
#include <stdint.h>
#include "pico/stdlib.h"
#include "hardware/uart.h"
#include "hardware/pio.h"
#include "hardware/irq.h"   /* B2a: PIO RX IRQ */
#include "clf_core.h"          /* B2b.1: was gap9_mod.h (phase2b/2c selector; D1 pinned Q16.16) */
#include "clf_vectors.h"       /* B2b.1: was vectors.h */
#include "uart_rx.pio.h"      /* copied verbatim from router/ : 8N1 PIO receiver for the 0x06 line */
#include "router_core.h"      /* B2b.2: SHARED, firmware/common/ (one copy for gap9 + router +
                              * the decode board) — was a private copy here at B2b.1. fa_feed/CRC. */
#include "rx_ring.h"          /* B2a: SHARED SPSC byte ring (firmware/common/) */

/* Identity LED (magenta). Defined in gap9_led.c — the ONLY TU that includes status_led.h, so the
 * shared enum's MODULE_GAP9 cannot collide with this file's MODULE_GAP9 (0x03 frame id). */
void board_identity_led(void);

// ---- pins ----
#define PIN_GO      2u
#define PIN_MARK    5u
#define PIN_RX      3u          // 0x06 RX from front-end GP4
/* The 0x03 result line. DEFAULTS UNCHANGED from the copied original (uart1 / GP4) — the #ifndef
 * guards exist only so the Renode runner can re-point them: the emulator's RP2040 platform models
 * ONLY uart0 (repos/Renode_RP2040/cores/rp2040.repl:47), so uart1's bytes are uncapturable there.
 * TRANSPORT/INSTRUMENTATION ONLY — the frame CONTENT is byte-identical either way, and it is the
 * content the comparator gates. Precedent: A2.2/A4.2 re-pinned their stdio UART for exactly this
 * class of rig/emulator constraint (a4_2_silicon.md:26-27,43). */
#ifndef PIN_RESULT
#define PIN_RESULT  4u
#endif
#ifndef RESULT_UART
#define RESULT_UART uart1
#endif
#define RESULT_BAUD 9600

#define MODULE_GAP9 0x03
#define KIND_RESULT 0x01
#define MODULE_FE_FWD 0x06      // front-end -> gap9 forward frame (raw marks + T̂u)

// B2b.1 build-time pins (the project's assert-at-build pattern): tie this wrapper to the decided
// constants and to the module's own shape. A drift here is a compile error, not a wrong frame.
_Static_assert(MAX_SPIKES == 256, "B2b.1: N_MARKS_MAX cap");
_Static_assert(MAX_GAPS == 255, "B2b.1: MAX_GAPS = MAX_SPIKES - 1");
_Static_assert(ROUTER_MAX_PL >= 9 + 5 * MAX_SPIKES, "fa_t.pl must hold the widest 0x06 (1289 B)");
_Static_assert(sizeof(((fa_t *)0)->len) == 2 && sizeof(((fa_t *)0)->idx) == 2,
               "B2b.1: fa_t len/idx must be u16 (a u8 idx wraps inside a 1289 B payload)");
_Static_assert(N_VEC3 > 0 && N_SPIKE3_TOTAL > 0, "baked vector set present");

// Live-interconnect single-vector select (bench GO path only; selection, not compute).
#define GAP9_REF_VECTOR_ID 30

#define CAL_TICKS   400000u     // marker CAL pulse width (RP2040 us), drift calibration
#define PRE_PASS_US 5000u
#define GAP_US      2000u

// ---------------- RESULT_FRAME emit (see common/RESULT_FRAME.md; B2.0 golden defines the bytes) --
static uint8_t crc8(const uint8_t *d, int n) {
    uint8_t c = 0;
    for (int i = 0; i < n; i++) { c ^= d[i];
        for (int b = 0; b < 8; b++) c = (c & 0x80) ? (uint8_t)((c << 1) ^ 0x07) : (uint8_t)(c << 1); }
    return c;
}

// Widest payload this board can EMIT: the 0x03 at the cap = 5 + 12*MAX_GAPS (D4: no decision bits).
// Derived from MAX_GAPS, never a literal (b2_frame_golden.md 2.5).
#define PL_0x03_MAX  (5 + 12 * MAX_GAPS)                /* 3065 */
#define TMP_0x03_MAX (3 + PL_0x03_MAX)                  /* 3068 — `3 +`: mid + BOTH length bytes */
_Static_assert(PL_0x03_MAX == 3065 && TMP_0x03_MAX == 3068, "B2.0 golden buffer sizes");

/* B2b.1: LENGTH u8 -> u16 BE. The CRC spans MODULE_ID ++ len_hi ++ len_lo ++ PAYLOAD — a
 * one-length-byte CRC mismatches every frame (the B1.1 trap, fe/frontend_stage3.c:87-89). The
 * staging buffer is STATIC and cap-sized (was `static tmp[2+200]`, which the widened 0x03
 * overflows 15x). */
static void emit_frame(uint8_t module_id, const uint8_t *payload, int len) {
    if (len < 0 || len > PL_0x03_MAX) return;        // structurally unemittable, never truncated
    uint8_t marker[2] = {0xA5, 0x5A};
    uint8_t hdr[3] = {module_id, (uint8_t)(len >> 8), (uint8_t)len};   // WIDENED: u16 BE LENGTH
    uart_write_blocking(RESULT_UART, marker, 2);
    uart_write_blocking(RESULT_UART, hdr, 3);
    if (len) uart_write_blocking(RESULT_UART, payload, len);
    static uint8_t tmp[TMP_0x03_MAX];
    tmp[0] = module_id; tmp[1] = hdr[1]; tmp[2] = hdr[2];
    for (int i = 0; i < len; i++) tmp[3 + i] = payload[i];
    uint8_t c = crc8(tmp, 3 + len);
    uart_write_blocking(RESULT_UART, &c, 1);
}
static int put_u16(uint8_t *p, int v) { p[0] = (uint8_t)(v >> 8); p[1] = (uint8_t)v; return 2; }
static int put_i32(uint8_t *p, int32_t v) { uint32_t u = (uint32_t)v;
    p[0] = (uint8_t)(u >> 24); p[1] = (uint8_t)(u >> 16); p[2] = (uint8_t)(u >> 8); p[3] = (uint8_t)u; return 4; }

// per-gap outputs — D4 argmax-free: 3 x Q16.16 per gap, gap-major. (The original's `arg[MAX_GAPS]`
// decision array is GONE: the 3c module has no argmax output.)
static int32_t lp3[3 * MAX_GAPS];

/* The 3-class contract, pinned WRAPPER-SIDE. `N_CLASSES` itself lives in weights_3c.h — which
 * carries 169 KB of `static const` tables, so including it HERE would hand this TU a private copy
 * (a1_golden_reference.md 6: `static const` in a header = one copy per including TU). gap9_3c.c is
 * the single include site and already asserts `N_CLASSES == 3` (:20). These two assertions tie the
 * wrapper's own buffer and payload formula to 3 classes x 4 B = 12 B/gap without importing it. */
_Static_assert(sizeof(lp3) / sizeof(lp3[0]) == 3 * MAX_GAPS, "D4: 3 log-posteriors per gap");
_Static_assert(PL_0x03_MAX == 5 + 12 * MAX_GAPS, "D4: 12 B/gap = 3 classes x i32, no decision bits");

/* Build + emit the widened, argmax-free 0x03 from lp3[]. `vid` is echoed: the baked VEC3_ID for the
 * GO/replay path, or the front-end's forwarded seq for process-on-arrival — the SINGLE 0x03
 * construction site for BOTH paths, so their frames are byte-identical.
 *
 * Mirrors b2_frame_golden/frames_b2.py:build_0x03_payload (which mirrors the copied original's
 * :91-102). Layout: [KIND 0x01][vector_id u16 BE][n_gaps u16 BE][3 x i32 BE per gap];
 * payload = 5 + 12*ng.
 *
 * D4 DELETES the decision-bits block (original :96-99) AND the `+4` term in the original's
 * `pl[1+2+1+ 4 +4*MAX_GAPS]` declaration (:92). That `4` was a decision-bits allowance HARDCODED
 * for MAX_GAPS=31 (ceil(31/8)=4 — margin exactly +0 as shipped); at MAX_GAPS=255 it needs 32 and
 * would overflow by 28 B (b2_frame_golden.md 2.3 — the router's `3 +` defect's twin). The term is
 * dissolved, not carried forward in any form. */
static void emit_result_0x03(int vid, int ng) {
    /* B2b.1: STATIC, not a stack local. The original's `uint8_t pl[132]` was a stack array — safe
     * at MAX_GAPS=31, but at 255 the payload is 3065 B and the declaration alone overruns the
     * 2048 B default stack. Measured, not assumed: -fstack-usage reported a 3,104 B frame here
     * before this change (b2b_1_port.md 5). Same class as B1.1's tmp[2+260] and the router's
     * main-locals — the cap raise arms it. */
    static uint8_t pl[PL_0x03_MAX];
    int o = 0;
    pl[o++] = KIND_RESULT;                             // :93
    o += put_u16(pl + o, vid);                         // :94
    o += put_u16(pl + o, ng);                          // :95 WIDENED (was `pl[o++] = (uint8_t)ng`)
                                                       // :96-99 decision_bits -> DELETED (D4)
    for (int t = 0; t < ng; t++)                       // :100 WIDENED (was 1 x i32 per gap)
        for (int i = 0; i < 3; i++) o += put_i32(pl + o, lp3[t * 3 + i]);
    emit_frame(MODULE_GAP9, pl, o);                    // :101
}

// run one baked vector (GO/replay path): bracket gap9_run with the GP5 marker, emit the 0x03.
static void run_one(int vi) {
    gpio_put(PIN_MARK, 1);
    int ng = clf_run(vi, lp3);              // UNCHANGED segmenter (A2, argmax-free entry)
    gpio_put(PIN_MARK, 0);
    emit_result_0x03((int)VEC3_ID[vi], ng);  // emit AFTER lowering marker (UART time not counted)
}

// ---- PROCESS-ON-ARRIVAL: consume a front-end 0x06 forward frame, run the UNCHANGED compute ----
static int32_t rd_i32(const uint8_t *p) {
    return (int32_t)(((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3]);
}
static int32_t rx_ticks[MAX_SPIKES];
static int32_t rx_ch[MAX_SPIKES];

/* 0x06 payload (the B1-frozen FE contract, fe/frontend_stage3.c:234-241 / b2_frame_golden.md 3):
 *   [0x01][seq u16][tu i32][n_marks u16][(end_tick i32, ch u8) x n]
 * B2b.1 WIDENING: n_marks u8 -> u16. The original read `ns = pl[7]` — a u8 — which against B1's
 * u16 emitter reads the HIGH byte: it mis-sizes every frame (n_marks < 256 -> ns = 0 -> dropped)
 * and the mark records start one byte late. Header is now 9 B, not 8.
 * Transmission-complete = this frame received (a 0x06 IS a complete transmission; no silence
 * detection). The front-end owns T̂u (gap9 does NOT estimate it). */
static void process_marks_frame(const uint8_t *pl, int len) {
    if (len < 9 || pl[0] != KIND_RESULT) return;    // WIDENED header (was 8); non-result -> ignore
    int vid = (pl[1] << 8) | pl[2];                 // forwarded seq -> 0x03 vector_id (pass-through)
    int32_t tut = rd_i32(pl + 3);
    int ns = (pl[7] << 8) | pl[8];                  // WIDENED: n_marks u16 BE (was `pl[7]`, u8)
    /* Over-cap behavior KEPT from the original (:128): silently drop. This is what orphans a
     * router slot forever (b2_router_decode_inventory.md 1.3) — a report path is B3's, not this
     * step's. Unreachable from the FE, which emits KIND 0x7F and drops above 256 marks. */
    if (ns < 2 || ns > MAX_SPIKES) return;          // need >=1 gap; bound the buffer
    if (9 + ns * 5 > len) return;                   // WIDENED offset; truncated/malformed payload
    if (tut <= 0) return;                           // guard the ISI divide (build_features needs >0)
    for (int i = 0; i < ns; i++) {
        rx_ticks[i] = rd_i32(pl + 9 + i * 5);       // WIDENED base offset (was 8)
        rx_ch[i]    = pl[9 + i * 5 + 4];
    }
    gpio_put(PIN_MARK, 1);
    int ng = clf_run_marks(rx_ticks, rx_ch, ns, tut, lp3);   // SAME compute, live marks + T̂u
    gpio_put(PIN_MARK, 0);
    emit_result_0x03(vid, ng);                      // SAME 0x03 construction as the replay path
}

/* ---- the RX byte path, in ONE place so the host harness drives the SHIPPING code ----
 * B2b.1 refactor (the only structural change to the copy): `fa` and the RX payload buffer move to
 * file scope (both were locals in main) and the drain body becomes rx_feed_byte(). main's loop is
 * unchanged in behavior — it just calls this per byte. This is what lets the host self-check feed
 * the golden's 0x06 BYTES through the real fa_feed + the real MODULE dispatch + the real
 * process_marks_frame, with no re-implementation of the path under test (the A1.1 "zero mocks of
 * the code under test" principle). Both buffers are STATIC: at ROUTER_MAX_PL = 3321, `pl` as a
 * stack local would smash the 2 KB default stack on the declaration alone. */
static fa_t    g_fa;
static uint8_t g_pl[ROUTER_MAX_PL];

/* ================================ B2a — IRQ + ring RX buffering ================================
 * WHY: this board is deaf for C(255) + E3(255) ~= 9.05 s at the cap while it computes and emits the
 * 0x03. Its PIO RX FIFO holds 8 B ~= 8.3 ms. Without buffering, everything the FE sends in that
 * window is lost. Derived backlog for this line: 595 B (b2a_sizing.py) -> a 1024 B ring.
 *
 * INVISIBLE TO THE FRAME LAYER: this changes where a byte comes FROM, not what it is. The B2b.1
 * gate (117/117 x2) is the transparency proof and is re-run unchanged.
 *
 * PIO0_IRQ_0 with the SM's rx_fifo_not_empty source. Nothing else on this board arms a PIO IRQ (the
 * LED is a one-shot pio1 write at boot); the only other IRQ in the image is the SDK's TIMER_IRQ_0
 * alarm pool, pulled in by the one boot sleep_ms. The handler is the SOLE producer. */
static rx_ring_t g_ring;
static PIO       g_rxpio;
static uint      g_rxsm;

static void gap9_rx_isr(void) {
    /* Drain the whole FIFO: the IRQ is level-sensitive on "not empty", so leaving a byte behind
     * would re-enter immediately. ~1 byte/ms at 9600 -> this fires ~960/s and costs ~1 us. */
    uint8_t b;
    while (uart_rx_getc_nonblock(g_rxpio, g_rxsm, &b)) rx_ring_push(&g_ring, b);
}

static void rx_feed_byte(uint8_t b) {
    uint8_t mid;
    uint16_t plen;                                  // WIDENED (was u8)
    if (fa_feed(&g_fa, b, &mid, g_pl, &plen) && mid == MODULE_FE_FWD)
        process_marks_frame(g_pl, (int)plen);
    // Any other MODULE (e.g. the FE's 0x01 debug frames) is framed and DISCARDED here; an unknown
    // KIND (e.g. the FE's 0x7F overflow marker) is rejected inside process_marks_frame by its
    // `pl[0] != KIND_RESULT` test. Both drop silently — no deframer on any board logs unknown
    // KINDs (b2_router_decode_inventory.md 5.2).
}

// Find the baked index whose stamped vector_id == id. -1 if absent.
static int find_vec_index(int vector_id) {
    for (int vi = 0; vi < N_VEC3; vi++) if ((int)VEC3_ID[vi] == vector_id) return vi;
    return -1;
}

static void run_pass(void) {
    gpio_put(PIN_MARK, 0); busy_wait_us(PRE_PASS_US);
    gpio_put(PIN_MARK, 1); busy_wait_us(CAL_TICKS);          // CAL pulse (once per pass)
    gpio_put(PIN_MARK, 0); busy_wait_us(GAP_US);
    int vi = find_vec_index(GAP9_REF_VECTOR_ID);
    if (vi >= 0) { run_one(vi); busy_wait_us(GAP_US); }
    emit_frame(MODULE_GAP9, (const uint8_t[]){0x02}, 1);     // end-of-pass sentinel
}

int main(void) {
    stdio_init_all();                         // uart0 on GP0/GP1 (debug sentinel only)
    board_identity_led();

    uart_init(RESULT_UART, RESULT_BAUD);
    gpio_set_function(PIN_RESULT, GPIO_FUNC_UART);
    uart_set_format(RESULT_UART, 8, 1, UART_PARITY_NONE);

    gpio_init(PIN_MARK); gpio_set_dir(PIN_MARK, GPIO_OUT); gpio_put(PIN_MARK, 0);
    gpio_init(PIN_GO);   gpio_set_dir(PIN_GO, GPIO_IN);   gpio_pull_down(PIN_GO);

    // 0x06 RX: PIO 8N1 receiver on GP3. The identity LED used pio1, so pio0 is free.
    PIO  rxpio = pio0;
    uint rxsm  = pio_claim_unused_sm(rxpio, true);
    uint rxoff = pio_add_program(rxpio, &uart_rx_program);
    uart_rx_program_init(rxpio, rxsm, rxoff, PIN_RX, RESULT_BAUD);
    fa_reset(&g_fa);

    /* B2a: arm the PIO RX IRQ. The ring must be live BEFORE the IRQ is enabled. */
    g_rxpio = rxpio; g_rxsm = rxsm;
    rx_ring_init(&g_ring);
    pio_set_irq0_source_enabled(rxpio, (enum pio_interrupt_source)(pis_sm0_rx_fifo_not_empty + rxsm),
                                true);
    irq_set_exclusive_handler(PIO0_IRQ_0, gap9_rx_isr);
    irq_set_enabled(PIO0_IRQ_0, true);

    sleep_ms(100);
    printf("RP2040_GAP9_3C_SENTENCE_OK maxsp=%d\n", (int)MAX_SPIKES);

    while (true) {
        // PRIMARY: process-on-arrival. B2a: read from the RING, not the FIFO — the ISR fills it
        // while this loop is deaf computing/emitting. The byte SOURCE changed; nothing else did.
        uint8_t b;
        while (rx_ring_pop(&g_ring, &b)) rx_feed_byte(b);
        // SECONDARY: bench GO trigger runs the baked replay pass so the board can be exercised
        // standalone without a front-end upstream.
        if (gpio_get(PIN_GO)) {
            run_pass();
            while (gpio_get(PIN_GO)) tight_loop_contents();
        }
        tight_loop_contents();
    }
}
