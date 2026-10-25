// B2b.1 Renode/silicon runner: feed the 117 FROZEN expected_0x06 frames through the SHIPPING RX
// path (rx_feed_byte -> fa_feed -> process_marks_frame -> gap9_run_marks -> emit_result_0x03) and
// let the REAL uart1 TX emit the 0x03 frames, which Renode captures byte-for-byte.
//
// This is the A2.1/A4.1 baked-vector pattern: the module under test is the literal shipping file,
// included in-TU with only its main() renamed. Nothing about the byte path is re-implemented —
// the only thing this file replaces is where the input bytes come from (baked, not a PIO FIFO).
#include "pico/stdlib.h"
#include "hardware/uart.h"

#define main gap9_hw_fw_main
#include "gap9_hw.c"
#undef main

#include "b2b_vectors.h"

int main(void) {
    // stdio is DISABLED in this build: RESULT_UART is re-pointed to uart0 (the only UART the Renode
    // RP2040 platform models), so uart0 must carry the 0x03 byte stream ALONE — a printf would
    // interleave into the capture. Termination is signalled IN-PROTOCOL by the wrapper's existing
    // end-of-pass sentinel frame (KIND 0x02), not by a text marker.
    board_identity_led();
    uart_init(RESULT_UART, RESULT_BAUD);   // the REAL 0x03 result line (re-pointed; see gap9_hw.c)
    gpio_set_function(PIN_RESULT, GPIO_FUNC_UART);
    uart_set_format(RESULT_UART, 8, 1, UART_PARITY_NONE);
    gpio_init(PIN_MARK); gpio_set_dir(PIN_MARK, GPIO_OUT); gpio_put(PIN_MARK, 0);
    fa_reset(&g_fa);
    rx_ring_init(&g_ring);

    sleep_ms(200);                         // let the Renode UART backend attach
    for (int v = 0; v < B2B_N_VEC; v++) {
        // B2a: inject at the RING'S PRODUCER (one layer earlier than B2b.1's runner) — the same
        // place the ISR pushes. The ring and everything above it are in the path; the ISR itself is
        // NOT (Renode cannot drive a PIO RX line). See b2a_port.md 5.
        const uint8_t *f = &B2B_F06[B2B_FOFF[v]];
        for (int i = 0; i < B2B_FLEN[v]; i++) {
            rx_ring_push(&g_ring, f[i]);                       // the ISR's job
            uint8_t x;
            while (rx_ring_pop(&g_ring, &x)) rx_feed_byte(x);  // the main loop's job, verbatim
        }
    }
    // end-of-pass sentinel — the SHIPPING emit_frame, the protocol's own KIND 0x02 (gap9_hw.c
    // run_pass()): the runner's terminator is a real frame, not instrumentation.
    emit_frame(MODULE_GAP9, (const uint8_t[]){0x02}, 1);
    while (true) tight_loop_contents();
    return 0;
}
