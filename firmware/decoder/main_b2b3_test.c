// B2b.3 Renode/silicon runner: feed the 117 FROZEN expected_0x05 frames through the SHIPPING RX path
// (rx_feed_byte -> fa_feed -> decode_forward -> beam_decode -> emit_result) and let the real
// putchar_raw emit the result lines, which Renode captures.
//
// ⚠️ WHAT THIS CANNOT COVER (b2b_3_port.md §4): Renode models only uart0 and cannot drive a UART RX
// line, so this board's HARDWARE uart1 RX (32 B FIFO) is unmodellable. The bytes are injected at
// rx_feed_byte — exactly where the FIFO drain hands them over — and stdio is pinned to uart0.
// Everything from the frame assembler onward is the literal shipping path; the uart1 RX actually
// clocking bytes at 9600 is SILICON's to prove (the same gap gap9/router carry for their PIO RX).
#include "pico/stdlib.h"

#define main decode_board_fw_main
#include "output_board.c"
#undef main

#include "b2b3_vectors.h"

int main(void) {
    stdio_init_all();
    rx_ring_init(&g_ring);
    sleep_ms(200);                            // let the Renode UART backend attach
    for (int v = 0; v < B3_N_VEC; v++) {
        // B2a: inject at the RING'S PRODUCER (the same place the ISR pushes). The ring and
        // everything above it are in the path; the ISR itself is NOT (Renode cannot drive a UART RX
        // line). See b2a_port.md 5.
        const uint8_t *f = &B3_F05[B3_F05_OFF[v]];
        for (int i = 0; i < B3_F05_LEN[v]; i++) {
            rx_ring_push(&g_ring, f[i]);                          // the ISR's job
            uint8_t buf[64];
            int n = 0;
            while (n < (int)sizeof buf && rx_ring_pop(&g_ring, &buf[n])) n++;
            if (n) on_rx_bytes(buf, n);                           // the main loop's job, verbatim
        }
    }
    // terminator: the 0x05 line's own end-of-pass sentinel (MODULE 0x05, KIND 0x02) fed through the
    // SAME path — decode_forward must REJECT it (R3), so it produces no result line. The runner's
    // end marker is therefore a real protocol frame AND a live rejection test; the harness keys on
    // the DONE line that follows.
    {
        static const uint8_t eop[7] = {0xA5, 0x5A, 0x05, 0x00, 0x01, 0x02, 0x55};
        for (int i = 0; i < 7; i++) {
            rx_ring_push(&g_ring, eop[i]);
            uint8_t buf[64];
            int n = 0;
            while (n < (int)sizeof buf && rx_ring_pop(&g_ring, &buf[n])) n++;
            if (n) on_rx_bytes(buf, n);
        }
    }
    putchar_raw('D'); putchar_raw('O'); putchar_raw('N'); putchar_raw('E'); putchar_raw('\n');
    fflush(stdout);
    while (true) tight_loop_contents();
    return 0;
}
