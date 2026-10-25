/*
 * output_board.c — USB-BRIDGE / OUTPUT-BOARD firmware (increment 1: RAW passthrough).
 *
 * Reads the router's 0x05 forward-frame stream on a hardware UART RX pin (wired to router GP4) and
 * relays every byte RAW to the host over USB-CDC. The board enumerates as /dev/ttyACM* and the
 * existing output_stage LiveSource reads the identical 0x05 byte stream it would from any serial
 * source. This board is the first increment of the eventual LCD display board — the byte path funnels
 * through on_rx_bytes(), whose body swaps from "relay" to "parse + display" later WITHOUT touching how
 * bytes arrive. Yellow identity LED (MODULE_DECODE). The router is NOT modified.
 *
 * Wiring:  output_board GP5 (uart1 RX)  <-  router GP4 (uart1 TX, 0x05 forward)  |  common ground
 *          output_board USB  ->  host /dev/ttyACM*
 */
#include <stdio.h>
#include <stdint.h>
#include "pico/stdlib.h"
#include "hardware/uart.h"
#include "hardware/irq.h"    /* B2a: UART RX IRQ */
#include "rx_ring.h"           /* B2a: SHARED SPSC byte ring (firmware/common/) */
#include "status_led.h"        /* shared identity LED (common/); MODULE_DECODE = yellow (already in table) */
#ifdef CHARLM_EMBED
#include "charlm_lookup.h"     /* A4.1: Q16.16 char-LM lookup (returns the beam addend directly) */
#include "beam.h"              /* A4.1: the integer sentence beam (beam.c) */
#include "router_core.h"       /* B2b.3: SHARED (firmware/common/) — fa_feed + decode_forward + CRC */
#endif

/* ---- UART RX from the router (params MATCH router GP4 forward TX exactly) ----
 * HARDWARE UART (uart1), not PIO: the WS2812 LED owns a PIO SM on pio1, and using a hardware UART here
 * means there is NO PIO state-machine or instruction-memory contention with the LED at all. GP5 is a
 * valid uart1 RX pin and is free (not GP0/1 debug, not GP23 LED, not GP24 button). */
#define RX_UART   uart1
#define RX_PIN    5u
#define RX_BAUD   9600         /* 8N1, idle-high — identical to router_hw.c LINE_BAUD / uart_set_format */

/* ================================ B2a — IRQ + ring RX buffering ================================
 * ⚠️ THE SECOND MECHANISM. gap9 and the router use the PIO receiver (8 B FIFO); THIS board uses a
 * HARDWARE uart1 (32 B FIFO ~= 33 ms at 9600) — a different peripheral, a different IRQ, a different
 * arming call. Same ring, same discipline.
 *
 * WHY: this board's deaf window is dominated NOT by the beam (128 ms at the cap) but by the USB
 * write — PICO_STDIO_USB_STDOUT_TIMEOUT_US = 500 ms if the host is not draining. 0.628 s total =
 * 19x the 32 B FIFO's 33 ms. Derived backlog: 602 B, which is EXACTLY the line's physical ceiling
 * (0.628 s x 960 B/s) — this line is saturated for the whole window (b2a_sizing.py). 1024 B ring.
 *
 * ⚠️ PRIORITY: this board also carries USBCTRL_IRQ (TinyUSB's dcd_rp2040_irq, RAM-resident, present
 * in A4.1's silicon-validated image) plus the SDK's TIMER_IRQ_0 alarm pool. All SDK IRQs default to
 * priority 0x80; irq_set_exclusive_handler leaves UART1_IRQ at that same default, so USB and UART
 * RX are EQUAL priority and cannot preempt each other — they serialize. That is what we want: a
 * 500 ms USB write cannot starve the UART ISR, because the USB *write* is a main-loop busy-wait
 * (stdio_usb polls tud_task), not an ISR that holds the core. The USB ISR itself is microseconds.
 * Were UART RX given a LOWER priority than USB it could still not be starved (the USB ISR does not
 * run for 500 ms); equal priority is simply the SDK default and needs no override.
 *
 * INVISIBLE TO THE FRAME LAYER: byte SOURCE only. The B2b.3 gate (117/117 x2 at BOTH levels) is the
 * transparency proof and is re-run unchanged. */
static rx_ring_t g_ring;

static void decode_rx_isr(void) {
    /* Drain the FIFO: the RX/RT interrupt is level-sensitive, so a byte left behind re-enters at
     * once. ~1 byte/ms at 9600 -> ~960 fires/s, ~1 us each. The SOLE producer. */
    while (uart_is_readable(RX_UART)) rx_ring_push(&g_ring, (uint8_t)uart_getc(RX_UART));
}

/* =================================================================================================
 * THE SEAM, FILLED (B2b.3). The seam's own increment-1 comment specified this work exactly:
 *   (a) accumulate bytes into RESULT_FRAMEs   -> fa_feed          (SHARED, common/router_core.h)
 *   (b) parse the 0x05 forward payload        -> decode_forward   (SHARED, written at B2b.3)
 *   (c) extract the decoded word              -> beam_decode      (A4, silicon-validated, UNTOUCHED)
 * The board was a raw USB relay; it is now a decoder. Everything above this line (arrival, FIFO
 * drain, main loop) is unchanged, exactly as the seam promised.
 *
 * ⚠️ This board's RX is HARDWARE uart1 (32 B FIFO), NOT the PIO receiver gap9/router use. CONFIRMED
 * a plumbing difference only: fa_feed takes a byte, not a source (B2b.2 split the config table from
 * the assembler), so the identical shared assembler serves both. The byte SOURCE differs; the frame
 * machine does not.
 * ============================================================================================== */
#ifdef CHARLM_EMBED
static fa_t         g_fa;                 /* .bss — the shared u16 frame assembler */
static uint8_t      g_pl[ROUTER_MAX_PL];  /* .bss — its payload staging (3,321 B) */
static fwd_t        g_fwd;                /* .bss — the parsed 0x05 (ch[] + lp3[], 3,324 B) */
static beam_result_t g_res;               /* .bss — the beam's output */

_Static_assert(ROUTER_MAXSP == 256 && BEAM_MAX_GAPS == ROUTER_MAXGAP,
               "B2b.3: the 0x05's cap and the beam's must be the same 256/255");
_Static_assert(ROUTER_MAX_PL == ROUTER_PL_0x05_MAX, "fa_t.pl must hold the widest 0x05");
_Static_assert(sizeof(g_fwd.lp3) / sizeof(g_fwd.lp3[0]) == 3 * BEAM_MAX_GAPS,
               "D4: 3 log-posteriors per gap, and the beam's lp3 shape");

/* Emit one decoded result. putchar_raw + an integer formatter, NOT printf/fputs: pico stdio
 * translates \n -> \r\n (A1.2) and does NOT wrap fputs at all, so its bytes never reach the
 * transport (A4.1 — found only by Renode). The transport seam has bitten four times; anything
 * byte-sensitive goes out unwrapped. */
static void emit_u32(uint32_t v) {
    char b[11]; int p = 10; b[p] = 0;
    do { b[--p] = (char)('0' + (v % 10)); v /= 10; } while (v);
    while (b[p]) putchar_raw(b[p++]);
}
static void emit_i64(int64_t v) {
    if (v < 0) { putchar_raw('-'); v = -v; }
    char b[21]; int p = 20; b[p] = 0;
    uint64_t u = (uint64_t)v;
    do { b[--p] = (char)('0' + (u % 10)); u /= 10; } while (u);
    while (b[p]) putchar_raw(b[p++]);
}
static void emit_result(uint16_t vid, const beam_result_t *r) {
    /* DEC <vid> <n_gaps> <died>[ <text> <seg> <score>] — the vector_id the beam never reads
     * (a4_beam_inventory.md 2.2) is carried here, because the result line is where it is useful. */
    putchar_raw('D'); putchar_raw('E'); putchar_raw('C'); putchar_raw(' ');
    emit_u32(vid); putchar_raw(' ');
    emit_u32((uint32_t)r->n_gaps); putchar_raw(' ');
    putchar_raw(r->died ? '1' : '0');
    if (!r->died) {
        putchar_raw(' ');
        for (int i = 0; i < r->text_len; i++) putchar_raw((uint8_t)r->text[i]);
        putchar_raw(' ');
        for (int g = 0; g < r->n_gaps; g++) putchar_raw((char)('0' + beam_seg_get(r, g)));
        putchar_raw(' ');
        emit_i64(r->score_q16);
    }
    putchar_raw('\n');
    fflush(stdout);
}

/* The RX byte path, in ONE place so the host harness drives the SHIPPING code (the B2b.1 pattern):
 * bytes -> fa_feed -> decode_forward -> beam_decode -> emit. */
static void rx_feed_byte(uint8_t b) {
    uint8_t mid;
    uint16_t plen;
    if (!fa_feed(&g_fa, b, &mid, g_pl, &plen)) return;      /* not a complete CRC-valid frame yet */
    if (!decode_forward(mid, g_pl, (int)plen, &g_fwd)) return;  /* wrong module/KIND/shape -> drop */
    beam_decode(g_fwd.ch, (int)g_fwd.n_spikes, g_fwd.lp3, BEAM_WIDTH, BEAM_LM_DEPLOY, &g_res);
    emit_result(g_fwd.vector_id, &g_res);                   /* a death IS a result: died=1 */
}
#endif /* CHARLM_EMBED */

static void on_rx_bytes(const uint8_t *buf, int len) {
#ifdef CHARLM_EMBED
    for (int i = 0; i < len; i++) rx_feed_byte(buf[i]);
#else
    /* CHARLM_EMBED=OFF: the original raw relay, byte-for-byte — the size-gate baseline and the
     * built-in negative control that this TU still compiles to the copied bridge's behavior. */
    for (int i = 0; i < len; i++) putchar_raw(buf[i]);
    fflush(stdout);
#endif
}

#ifdef CHARLM_EMBED
/* ---- A4.1 test scaffolding (Q16.16 lineage of the A1.2 'P'/'D' commands; a4_1_port.md):
 * 'P' -> liveness probe (boundary rows of the Q16.16 golden, printed as int32).
 * 'D' -> drive ALL 50,653 lookups THROUGH charlm_lookup_idx in golden index order (nested c1/c2/c3
 *        loops — exercising the ARM index arithmetic, NOT a memcpy) and stream each value as
 *        4 raw bytes LITTLE-ENDIAN. Iterated in index order this must reproduce
 *        charlm_mix_q16.bin (202,612 B) EXACTLY — A4.2 verifies by SHA-256 + per-entry compare.
 * 'B' -> beam liveness probe (also run at boot): decode the 2-mark pattern ".-" with a synthetic
 *        one-gap posterior favoring intra — must print "A". Keeps beam_decode + its ~47 KB .bss
 *        + the Morse tables linked (the table's real consumer; replaces the A1.1 gc-anchor role
 *        of the probe) and gives A4.2 a one-line beam liveness check. */
static void charlm_probe(void) {
    printf("CHARLM_A4 q16[BOS,' ']=%ld q16[idx0]=%ld q16[idx50652]=%ld\n",
           (long)charlm_letter_logprob_q16(' ', (const uint8_t *)"", 0),
           (long)charlm_lookup_idx(0, 0, 0),
           (long)charlm_lookup_idx(36, 36, 36));
    fflush(stdout);
}
static void charlm_dump(void) {
    printf("CHARLM_DUMP_BEGIN\n");
    fflush(stdout);
    for (int c1 = 0; c1 < 37; c1++)
        for (int c2 = 0; c2 < 37; c2++)
            for (int c3 = 0; c3 < 37; c3++) {
                int32_t q = charlm_lookup_idx(c1, c2, c3);
                putchar_raw((uint8_t)(q & 0xff));
                putchar_raw((uint8_t)((q >> 8) & 0xff));
                putchar_raw((uint8_t)((q >> 16) & 0xff));
                putchar_raw((uint8_t)((q >> 24) & 0xff));
            }
    fflush(stdout);
    printf("\nCHARLM_DUMP_END\n");
    fflush(stdout);
}
static void beam_probe(void) {
    static const uint8_t ch2[2] = { 0, 1 };                       /* ".-" */
    static const int32_t lp1[3] = { 0, -30 * 65536, -30 * 65536 };/* one gap, intra favored */
    static beam_result_t br;                                      /* .bss, not stack */
    beam_decode(ch2, 2, lp1, BEAM_WIDTH, BEAM_LM_DEPLOY, &br);
    printf("BEAM_A4 died=%d text=%.*s\n", (int)br.died, (int)br.text_len, br.text);
    fflush(stdout);
}
#endif

int main(void) {
    /* identity LED first: yellow = decode/output board. WS2812 on GP23 via pio1; our UART RX is a
     * HARDWARE UART (no PIO) -> no PIO/pin collision with the LED. */
    status_led_init();
    status_led_set_identity(MODULE_DECODE);

    stdio_init_all();                                    /* USB-CDC per CMake (usb=1, uart-stdio=0) */

#ifdef CHARLM_EMBED
    /* A1.1/A1.2 boot self-probe (also re-printable on demand, see 'P' below — stdio_usb drops
     * writes with no terminal attached, so the boot print alone is not reliable liveness). */
    charlm_probe();
    beam_probe();
#endif

    uart_init(RX_UART, RX_BAUD);
    gpio_set_function(RX_PIN, GPIO_FUNC_UART);
    uart_set_format(RX_UART, 8, 1, UART_PARITY_NONE);    /* 8N1 — matches router GP4 forward TX */
    uart_set_fifo_enabled(RX_UART, true);

    /* B2a: arm the uart1 RX IRQ. The ring must be live BEFORE the IRQ is enabled.
     * uart_set_irq_enables(rx=true, tx=false): RX (FIFO threshold) + RT (receive timeout, which is
     * what delivers a trailing partial FIFO — without RT the last few bytes of a frame would sit in
     * the FIFO until the next byte arrived, i.e. forever at end-of-transmission). */
    rx_ring_init(&g_ring);
    irq_set_exclusive_handler(UART1_IRQ, decode_rx_isr);
    irq_set_enabled(UART1_IRQ, true);
    uart_set_irq_enables(RX_UART, true, false);

    /* B2a: read from the RING, not the FIFO — the ISR fills it while this loop is deaf in the beam
     * or, dominantly, in a USB write. The byte SOURCE changed; on_rx_bytes and everything above it
     * did not (the seam's promise: "nothing above this line changes"). */
    uint8_t buf[64];
    while (true) {
        int n = 0;
        while (n < (int)sizeof buf && rx_ring_pop(&g_ring, &buf[n])) n++;
        if (n) on_rx_bytes(buf, n);
#ifdef CHARLM_EMBED
        /* A1.2 scaffolding: non-blocking command poll on USB-CDC stdin */
        int cmd = getchar_timeout_us(0);
        if (cmd == 'P') charlm_probe();
        else if (cmd == 'D') charlm_dump();
        else if (cmd == 'B') beam_probe();
#endif
        tight_loop_contents();
    }
}
