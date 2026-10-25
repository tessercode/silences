/* B2b.3 host harness — MOCK hardware/uart.h. The decode board's RX is a hardware UART; the harness
 * feeds bytes straight into the SHIPPING rx_feed_byte(), so the FIFO is not under test. */
#ifndef MOCK_HARDWARE_UART_H
#define MOCK_HARDWARE_UART_H
#include <stdint.h>
#include <stdbool.h>
typedef struct { int id; } *uart_inst_t;
#define uart0 ((uart_inst_t)0)
#define uart1 ((uart_inst_t)1)
#define UART_PARITY_NONE 0
static inline unsigned uart_init(uart_inst_t u, unsigned baud) { (void)u; return baud; }
static inline void uart_set_format(uart_inst_t u, int d, int s, int p) { (void)u;(void)d;(void)s;(void)p; }
static inline void uart_set_fifo_enabled(uart_inst_t u, bool e) { (void)u; (void)e; }
static inline bool uart_is_readable(uart_inst_t u) { (void)u; return false; }
static inline int  uart_getc(uart_inst_t u) { (void)u; return 0; }
static inline void uart_set_irq_enables(uart_inst_t u, bool rx, bool tx) { (void)u;(void)rx;(void)tx; }
#endif
