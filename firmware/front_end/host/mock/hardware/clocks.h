/* host-mock hardware/clocks.h — reports the pico-sdk default sysclk so the B1.1 boot assertion
 * (clock_get_hz(clk_sys) == 125 MHz) is exercised on host too. */
#pragma once
#include <stdint.h>
enum { clk_sys = 5 };
uint32_t clock_get_hz(int clk);
