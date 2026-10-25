/* host-mock pico/stdlib.h — E.1 host gate. Declares exactly the pico-sdk surface encoder_gpio.c
 * touches; definitions live in enc_host_check.c (same TU as the included firmware), mirroring
 * fe/host/mock/pico/stdlib.h. Nothing here is compiled into the target build. */
#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

typedef unsigned int uint;

/* gpio — no-ops on host. The GPIO EMISSION is not what the host gate checks: it checks the rendered
 * tick/channel array, which is emit_gpio's input. (Emission itself is silicon's — E.2.) */
#define GPIO_OUT 1
void gpio_init(uint pin);
void gpio_set_dir(uint pin, int dir);
void gpio_put(uint pin, int value);

void stdio_init_all(void);
void sleep_ms(uint32_t ms);
void tight_loop_contents(void);

/* busy_wait_us: on host this only ACCUMULATES simulated time, so the harness can assert the emit
 * schedule is monotonic without waiting ~75 s of real time per sentence. */
void busy_wait_us(uint64_t us);

#define PICO_ERROR_TIMEOUT (-1)
int getchar_timeout_us(uint32_t us);
