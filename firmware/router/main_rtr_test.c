// B2b.2 Renode/silicon runner: feed the 117 FROZEN (expected_0x06, expected_0x03) frame pairs
// through the SHIPPING router core — fa_feed -> decode_half -> router_ingest -> rtr_build_forward —
// and let the real uart_write_blocking emit the 0x05s, which Renode captures byte-for-byte.
//
// ⚠️ WHAT THIS RUNNER CANNOT COVER, and why (b2b_2_port.md 7): Renode's RP2040 platform models only
// uart0 and offers no way to drive GP2/GP3 with 8N1 UART waveforms, so the router's TWO PIO
// RECEIVERS and its uart1/GP4 TX are unmodellable. The bytes are therefore injected at fa_feed (the
// first line of the core, exactly where the PIO FIFOs would hand them over) and the TX is
// re-pointed to uart0. Everything from the frame assembler onward is the literal shipping path;
// the PIO receivers + the uart1 TX are SILICON's to prove (A4.2-style).
//
// The frames are fed INTERLEAVED across the two logical input lines, as the two receivers would.
#include "pico/stdlib.h"
#include "hardware/uart.h"

#define main router_hw_fw_main
#include "router_hw.c"
#undef main

#include "rtr_vectors.h"

int main(void) {
    status_led_init();
    status_led_set_identity(MODULE_ROUTER);
    uart_init(FWD_UART, LINE_BAUD);          // the 0x05 result line (re-pointed; see router_hw.c)
    gpio_set_function(PIN_FWD, GPIO_FUNC_UART);
    uart_set_format(FWD_UART, 8, 1, UART_PARITY_NONE);
    for (int i = 0; i < NIN; i++) { fa_reset(&fas[i]); rx_ring_init(&g_rings[i]); }
    router_init(&R);

    sleep_ms(200);                            // let the Renode UART backend attach
    static uint8_t pl_r[ROUTER_MAX_PL];
    static half_t  h_r;
    static uint8_t out_r[ROUTER_OUT_0x05_MAX];
    for (int v = 0; v < RTR_N_VEC; v++) {
        const uint8_t *f6 = &RTR_F06[RTR_F06_OFF[v]], *f3 = &RTR_F03[RTR_F03_OFF[v]];
        int n6 = RTR_F06_LEN[v], n3 = RTR_F03_LEN[v];
        int i = 0, j = 0;
        while (i < n6 || j < n3) {            // INTERLEAVED, as the two receivers would deliver
            for (int line = 0; line < 2; line++) {
                uint8_t b;
                if (line == 0) { if (i >= n6) continue; b = f6[i++]; }
                else           { if (j >= n3) continue; b = f3[j++]; }
                // B2a: inject at the RING'S PRODUCER (the same place the ISR pushes), then run the
                // main loop's drain verbatim. The ring is in the path; the ISR is NOT (Renode
                // cannot drive a PIO RX line). See b2a_port.md 5.
                rx_ring_push(&g_rings[line], b);                  // the ISR's job
                uint8_t rb;
                while (rx_ring_pop(&g_rings[line], &rb)) {        // the main loop's job, verbatim
                    uint8_t mid; uint16_t len;
                    if (fa_feed(&fas[line], rb, &mid, pl_r, &len)) {
                        if (decode_half(mid, pl_r, (int)len, &h_r)) {
                            int fl = router_ingest(&R, &h_r, out_r);
                            if (fl > 0) uart_write_blocking(FWD_UART, out_r, fl);   // the REAL emit
                        }
                    }
                }
            }
        }
    }
    // end-of-pass sentinel — the protocol's own KIND 0x02 on the 0x05 line, so the runner's
    // terminator is a real frame rather than instrumentation (the B2b.1 pattern).
    {
        static uint8_t eop[8];
        int po = 0;
        eop[po++] = ROUTER_START0; eop[po++] = ROUTER_START1; eop[po++] = ROUTER_MOD_FORWARD;
        eop[po++] = 0; eop[po++] = 1; eop[po++] = ROUTER_KIND_EOP;
        uint8_t t[4] = {ROUTER_MOD_FORWARD, 0, 1, ROUTER_KIND_EOP};
        eop[po++] = rtr_crc8(t, 4);
        uart_write_blocking(FWD_UART, eop, po);
    }
    while (true) tight_loop_contents();
    return 0;
}
