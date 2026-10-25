/* B2b.1 host harness — MOCK hardware/uart.h. uart_write_blocking CAPTURES the emitted bytes: that
 * capture IS the frame the comparator checks against the frozen B2.0 golden. */
#ifndef MOCK_HARDWARE_UART_H
#define MOCK_HARDWARE_UART_H
#include <stdint.h>
#include <stdbool.h>

typedef struct { int id; } *uart_inst_t;
#define uart0 ((uart_inst_t)0)
#define uart1 ((uart_inst_t)1)
#define UART_PARITY_NONE 0

/* the capture sink (defined by the harness) */
extern void host_uart_capture(const uint8_t *src, int len);

static inline unsigned uart_init(uart_inst_t u, unsigned baud) { (void)u; return baud; }
static inline void uart_set_format(uart_inst_t u, int d, int s, int p) { (void)u; (void)d; (void)s; (void)p; }
static inline void uart_set_fifo_enabled(uart_inst_t u, bool e) { (void)u; (void)e; }
static inline void uart_write_blocking(uart_inst_t u, const uint8_t *src, size_t len) {
    (void)u; host_uart_capture(src, (int)len);
}
#endif
