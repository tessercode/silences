/* host-mock pico/stdlib.h — B1.1 host self-check. Declares exactly the pico-sdk surface
 * frontend_stage3.c touches; definitions live in host_harness.c (same TU as the included firmware). */
#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

typedef unsigned int uint;

/* gpio (no-ops on host) */
#define GPIO_IN 0
#define GPIO_FUNC_UART 2
void gpio_init(uint pin);
void gpio_set_dir(uint pin, int dir);
void gpio_pull_down(uint pin);
void gpio_set_function(uint pin, int fn);

void stdio_init_all(void);
void sleep_ms(uint32_t ms);
void tight_loop_contents(void);

/* mock clock: returns sim time and advances it (host_harness.c) */
uint32_t time_us_32(void);

void mock_hard_assert(int ok, const char *expr);
#define hard_assert(x) mock_hard_assert(!!(x), #x)
