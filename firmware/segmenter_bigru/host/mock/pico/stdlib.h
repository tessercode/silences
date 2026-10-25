/* B2b.1 host harness — MOCK pico/stdlib.h. Satisfies the SHIPPING gap9_hw.c's pico surface so the
 * literal shipping file compiles in-TU natively. Only the pico layer is mocked; the code under
 * test (fa_feed, process_marks_frame, emit_result_0x03, emit_frame) is the real thing. */
#ifndef MOCK_PICO_STDLIB_H
#define MOCK_PICO_STDLIB_H
#include <stdint.h>
#include <stdbool.h>
#include "hardware/uart.h"
#include "hardware/pio.h"

#define GPIO_FUNC_UART 2
#define GPIO_IN  0
#define GPIO_OUT 1
static inline void stdio_init_all(void) {}
static inline void sleep_ms(uint32_t ms) { (void)ms; }
static inline void busy_wait_us(uint32_t us) { (void)us; }
static inline void tight_loop_contents(void) {}
static inline void gpio_init(unsigned p) { (void)p; }
static inline void gpio_set_dir(unsigned p, bool out) { (void)p; (void)out; }
static inline void gpio_put(unsigned p, bool v) { (void)p; (void)v; }
static inline bool gpio_get(unsigned p) { (void)p; return false; }   /* GO never asserted on host */
static inline void gpio_pull_down(unsigned p) { (void)p; }
static inline void gpio_pull_up(unsigned p) { (void)p; }
static inline void gpio_set_function(unsigned p, int f) { (void)p; (void)f; }
#endif
