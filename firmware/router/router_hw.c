/*
 * router_hw.c — B2b.2 SENTENCE-TREE router firmware (transmission-scale pairing + u16 frames).
 *
 * COPIED from deploy/rp2040/router/router_hw.c (sha256 aa4afa6af63b19ed...) and modified; the
 * read-only original is untouched. The pairing/frame logic lives in the SHARED
 * firmware/common/router_core.h (one copy for gap9 + this board + the decode board later).
 *
 * DIFF vs the copied original (every change forced by a decided constant or a measured defect):
 *   1. router_core.h is now ../common/ (shared) and is widened: u16 LENGTH/counts, D4 argmax-free
 *      3 x i32 per gap, ROUTER_MAXSP 64 -> 256, the `3 +` header defect FIXED, B3 eviction.
 *   2. main's `pl` / `out` / `half_t h` were STACK LOCALS (910 B at word scale) -> 9,730 B at the
 *      cap on a 2,048 B default stack. All three are now STATIC and cap-sized (B1.1's lesson;
 *      b2_router_decode_inventory.md 1.4 measured the 4.75x overrun).
 *   3. PICO_STACK_SIZE = 0x1000 (CMakeLists) — B1.1/B2b.1 precedent.
 *   4. `plen` u8 -> u16 (fa_feed's widened out_len).
 *   5. Stats are surfaced on the EXISTING uart0 debug line, rare-event-gated (see rtr_report).
 *
 * The 0x05 BYTES are defined by the FROZEN B2.0 golden (b2_frame_golden/golden_frames_n{1,5}.json).
 *
 * (original header follows)
 * router_hw.c — RP2040 interconnect-router firmware (increment 1: encoder 0x04 + gap9 0x03).
 *
 * Two PIO UART receivers (one state machine per producer line) free-run in hardware; the CPU drains
 * their FIFOs, feeds bytes to the SHARED router_core.h (same code proven host-side in Phase A), which
 * assembles RESULT_FRAMEs, pairs the two halves by vector_id (buffering the earlier-arriving encoder
 * marks across gap9's ~76ms compute skew), validates alignment, and forwards a 0x05 frame on the TX
 * line. Producer frames are consumed UNCHANGED. Integer-only -> FPU-free.
 *
 * INPUTS ARE A TABLE (rx_input_t inputs[]) — increment 1 has two entries; a third (front-end) is one
 * more row + a decode_half case. No path is hardcoded.
 *
 * Wiring (reuses the 9600-8N1 result-line convention; separate RX pins per producer):
 *   GP2  <- FRONT-END 0x06 marks+T̂u line (uart 0x06)  -- PIO0 SM0   [re-pointed: was encoder 0x04, now
 *           the front-end's forward line; the encoder emits completion-spike GPIO and no longer frames.
 *           The FE GP4 line FANS OUT to gap9 GP3 (existing) AND this router GP2 (new).]
 *   GP3  <- gap9 board result line      (uart 0x03)   -- PIO0 SM1
 *   GP4  -> forwarded 0x05 frame (uart1 TX, 8N1 9600) -> output board / host
 *   GP0/GP1 = uart0 debug sentinel.  GP23/GP24 = board RGB/button, DO NOT TOUCH.
 */
#include <stdio.h>
#include <stdint.h>
#include "pico/stdlib.h"
#include "hardware/pio.h"
#include "hardware/irq.h"   /* B2a: PIO RX IRQ */
#include "hardware/uart.h"
#include "router_core.h"   /* B2b.2: SHARED, firmware/common/ */
#include "rx_ring.h"       /* B2a: SHARED SPSC byte ring (firmware/common/) */
#include "uart_rx.pio.h"
#include "status_led.h"       /* shared identity LED (common/) */

#define PIN_FE      2u          /* front-end 0x06 marks line (re-pointed; was PIN_ENC / encoder 0x04) */
#define PIN_GAP9    3u
/* The 0x05 forward line. DEFAULTS UNCHANGED from the copied original (uart1 / GP4) — the #ifndef
 * guards exist only so the Renode runner can re-point them: the emulator's RP2040 platform models
 * ONLY uart0 (repos/Renode_RP2040/cores/rp2040.repl:47). TRANSPORT ONLY — the frame CONTENT is
 * byte-identical either way, and the content is what the comparator gates. Precedent: A2.2/A4.2 and
 * B2b.1 re-pinned for exactly this class of emulator/rig constraint. */
#ifndef PIN_FWD
#define PIN_FWD     4u
#endif
#ifndef FWD_UART
#define FWD_UART    uart1
#endif
#define LINE_BAUD   9600

/* ---- the input table (N-input extensible; add a row + decode_half case for the front-end) ----
 *
 * ⚠️ B2b.2: the `fa_t` was a MEMBER of this initialized array. That put it in `.data`, not `.bss` —
 * because the row's other fields (pio/sm/pin/module) are non-zero, the compiler must emit the WHOLE
 * struct, assembler included, as flash-resident initialized data copied to SRAM at boot. At the
 * shipped ROUTER_MAX_PL = 255 that was ~528 B and invisible; at 3,321 it is 6,696 B of needless
 * FLASH plus a 6.7 KB boot-time copy. Exactly A1.0's documented trap ("49.5 KB in .data would be
 * copied to SRAM at boot — the single most likely porting error", a1_golden_reference.md 6), and it
 * only became visible because the widening made it big. MEASURED, not assumed: `nm` put `inputs` at
 * 0x200001a4 inside .data (7,088 B) before this split; after it, .data is 392 B.
 *
 * Split: the config table stays initialized (and is now `const` -> .rodata, off SRAM entirely); the
 * assemblers live in their own all-zero array, which lands in .bss where they belong. */
typedef struct {
    PIO     pio;
    uint    sm;
    uint    pin;
    uint8_t module;          /* which producer this line carries (documentation/validation) */
} rx_input_t;

#define NIN 2
static const rx_input_t inputs[NIN] = {
    { pio0, 0, PIN_FE,   ROUTER_MOD_FE_FWD },   /* MARKS: front-end 0x06 (was encoder 0x04) */
    { pio0, 1, PIN_GAP9, ROUTER_MOD_GAP9   },   /* BOUNDARIES: gap9 0x03 (unchanged) */
};
static fa_t fas[NIN];        /* .bss (all-zero) — one frame assembler per input line */

/* ================================ B2a — IRQ + ring RX buffering ================================
 * WHY: this board is deaf for E5(256) ~= 3.47 s while it emits a cap 0x05. Each PIO RX FIFO holds
 * 8 B ~= 8.3 ms. Derived backlogs (b2a_sizing.py): line A (FE) 630 B, line B (gap9's 0x03) 746 B —
 * line B is the WORST line in the whole chain, and the least obvious, because the 0x03 for the next
 * transmission only starts after gap9's compute. Both get a 1024 B ring.
 *
 * INVISIBLE TO THE FRAME LAYER: byte SOURCE only. The B2b.2 gate (117/117 x2, both arrival orders)
 * is the transparency proof and is re-run unchanged.
 *
 * ONE handler for BOTH state machines: they share PIO0_IRQ_0, and the cheapest correct demux is to
 * drain both SMs unconditionally (a non-empty check is what uart_rx_getc_nonblock already does).
 * Nothing else on this board arms a PIO IRQ (the LED is a one-shot pio1 write at boot); the only
 * other IRQ in the image is the SDK's TIMER_IRQ_0 alarm pool. The handler is the SOLE producer. */
static rx_ring_t g_rings[NIN];

static void rtr_rx_isr(void) {
    uint8_t b;
    for (int i = 0; i < NIN; i++)
        while (uart_rx_getc_nonblock(inputs[i].pio, inputs[i].sm, &b))
            rx_ring_push(&g_rings[i], b);
}

static router_t R;

/* B2b.2 stats surfacing. The shipped code COUNTED mismatches/dropped_full/hwm and reported them
 * NOWHERE (router_hw.c had only a boot sentinel) — a silent wedge was undiagnosable. With B3's
 * eviction these are DIAGNOSTIC, not safety-critical (the table can no longer fatally fill), so the
 * bar is: cost nothing on the happy path and add no deaf window.
 *
 * Options weighed (b2b_2_port.md 5):
 *   - uart0 stdio printf on the EXISTING debug line  <- CHOSEN. Zero wire cost on the 0x05 path
 *     (different UART), zero cost when nothing anomalous happens (rare-event gated: printed only
 *     when a counter CHANGES), and the line already exists for the boot sentinel. A print is
 *     ~1-2 ms of blocking uart0 TX at 115200 for a ~40 B line — but it can only fire on an
 *     eviction/mismatch, i.e. never on the deployed cadence, and it happens AFTER the 0x05 is
 *     already emitted, inside the inter-transmission idle (>= 0.8 s of silence).
 *   - WS2812 LED: no history, no counts, and it would need a colour policy — rejected.
 *   - a stats KIND on the 0x05 line: adds bytes to the CHAIN's binding hop (the router's emit
 *     window is what binds pacing, chain_pacing_analysis.md 5) and would change the golden's byte
 *     stream — rejected outright.
 * Do NOT add a deaf window for a diagnostic. */
static void rtr_report(void) {
    static int last_mm = -1, last_ev = -1, last_hwm = -1;
    static uint32_t last_drop = 0xFFFFFFFFu;
    const uint32_t drops = g_rings[0].dropped + g_rings[1].dropped;   /* B2a: ring overflow */
    if (R.mismatches == last_mm && R.evictions == last_ev && R.hwm == last_hwm &&
        drops == last_drop) return;
    last_mm = R.mismatches; last_ev = R.evictions; last_hwm = R.hwm; last_drop = drops;
    /* B2a adds ringdrop: the ONLY evidence that the T_ITX = 3.5 s sizing assumption was violated.
     * An overflow degrades to a lost transmission (CRC fails -> fa_feed resyncs -> the half orphans
     * -> B3's eviction reclaims the slot), so it is contained — but silent without this. */
    printf("RTR paired=%d mismatch=%d evict=%d hwm=%d dropfull=%d ringdrop=%lu\n",
           R.paired, R.mismatches, R.evictions, R.hwm, R.dropped_full, (unsigned long)drops);
}

int main(void) {
    stdio_init_all();                                        /* uart0 GP0/GP1 debug sentinel */

    /* identity LED first: announce "this is the router" (cyan) at boot. Uses pio1 (a claimed unused
     * SM) — no collision with the UART receivers on pio0 below, and does not touch GP2/3/4. */
    status_led_init();
    status_led_set_identity(MODULE_ROUTER);

    /* forward line: hardware UART TX, 8N1 9600 */
    uart_init(FWD_UART, LINE_BAUD);
    gpio_set_function(PIN_FWD, GPIO_FUNC_UART);
    uart_set_format(FWD_UART, 8, 1, UART_PARITY_NONE);

    /* load the RX program once, init one SM per input line */
    uint offset = pio_add_program(pio0, &uart_rx_program);
    for (int i = 0; i < NIN; i++) {
        fa_reset(&fas[i]);
        rx_ring_init(&g_rings[i]);                       /* B2a: live BEFORE the IRQ is enabled */
        uart_rx_program_init(inputs[i].pio, inputs[i].sm, offset, inputs[i].pin, LINE_BAUD);
    }
    router_init(&R);

    /* B2a: arm ONE PIO IRQ for both lines (both SMs are on pio0). */
    for (int i = 0; i < NIN; i++)
        pio_set_irq0_source_enabled(inputs[i].pio,
            (enum pio_interrupt_source)(pis_sm0_rx_fifo_not_empty + inputs[i].sm), true);
    irq_set_exclusive_handler(PIO0_IRQ_0, rtr_rx_isr);
    irq_set_enabled(PIO0_IRQ_0, true);

    sleep_ms(100);
    printf("RP2040_ROUTER_HW_OK inputs=%d\n", NIN);

    /* B2b.2: STATIC, not main locals. At the cap these are 3,321 + 3,327 + 3,336 = 9,984 B; as
     * stack locals they overran the 2,048 B default stack by 4.75x (the shipped word-scale sizes
     * were 910 B, which fit). -fstack-usage is what proves it — b2b_1_port.md 5 caught exactly this
     * shape in gap9's writer after the host gate was already green. */
    static uint8_t mid, pl[ROUTER_MAX_PL], out[ROUTER_OUT_0x05_MAX], b;
    static uint16_t len;                                   /* WIDENED: fa_feed's out_len is u16 */
    static half_t h;
    while (true) {
        for (int i = 0; i < NIN; i++) {                      /* B2a: drain each line's RING */
            while (rx_ring_pop(&g_rings[i], &b)) {
                if (fa_feed(&fas[i], b, &mid, pl, &len)) {   /* complete CRC-valid frame */
                    if (decode_half(mid, pl, (int)len, &h)) {
                        int fl = router_ingest(&R, &h, out);
                        if (fl > 0) uart_write_blocking(FWD_UART, out, fl);   /* forward 0x05 */
                        /* fl < 0 (alignment mismatch) -> nothing forwarded (bad pair refused) */
                        rtr_report();   /* rare-event gated; AFTER the emit, in the idle window */
                    }
                }
            }
        }
        tight_loop_contents();
    }
}
