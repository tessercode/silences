/* B2b.3 host harness — MOCK pico/stdlib.h. Satisfies the SHIPPING output_board.c's pico surface so
 * the literal shipping file compiles in-TU natively. Only the pico layer is mocked; the code under
 * test (fa_feed, decode_forward, beam_decode, emit_result) is the real thing. */
#ifndef MOCK_PICO_STDLIB_H
#define MOCK_PICO_STDLIB_H
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include "hardware/uart.h"

#define GPIO_FUNC_UART 2
#define GPIO_IN  0
#define GPIO_OUT 1

/* the capture sink (defined by the harness): putchar_raw IS the USB-CDC transport on this board */
extern void host_putchar_raw(int c);
#define putchar_raw(c) host_putchar_raw(c)

static inline void stdio_init_all(void) {}
static inline void sleep_ms(uint32_t ms) { (void)ms; }
static inline void tight_loop_contents(void) {}
static inline void gpio_set_function(unsigned p, int f) { (void)p; (void)f; }
static inline int  getchar_timeout_us(uint32_t us) { (void)us; return -1; }   /* no host commands */
#endif
