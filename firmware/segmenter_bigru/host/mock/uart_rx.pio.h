/* B2b.1 host harness — MOCK uart_rx.pio.h (normally GENERATED from uart_rx.pio by
 * pico_generate_pio_header). Stubs only: main()'s RX drain is not the host path. */
#ifndef MOCK_UART_RX_PIO_H
#define MOCK_UART_RX_PIO_H
#include <stdint.h>
#include "hardware/pio.h"
static const int uart_rx_program = 0;
static inline void uart_rx_program_init(PIO p, uint sm, uint off, uint pin, uint baud) {
    (void)p; (void)sm; (void)off; (void)pin; (void)baud;
}
static inline int uart_rx_getc_nonblock(PIO p, uint sm, uint8_t *out) {
    (void)p; (void)sm; (void)out; return 0;      /* host: no FIFO; the harness calls rx_feed_byte */
}
#endif
