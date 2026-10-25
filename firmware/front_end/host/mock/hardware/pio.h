/* host-mock hardware/pio.h — the two capture state machines are simulated from a per-channel
 * edge timeline (host_harness.c): a "FIFO word" becomes available at the wall-clock time of the
 * edge that ends a level run, with value = run length in counts. FIFO depth is NOT modeled (the
 * firmware polls far faster than events in sim); that is a B0a scheduling concern, not framing. */
#pragma once
#include <stdint.h>
#include <stdbool.h>
#include "pico/stdlib.h"   /* uint */

typedef struct mock_pio_inst *PIO;
extern struct mock_pio_inst mock_pio0;
#define pio0 (&mock_pio0)

bool pio_sm_is_rx_fifo_empty(PIO p, uint sm);
uint32_t pio_sm_get(PIO p, uint sm);
void pio_set_sm_mask_enabled(PIO p, uint32_t mask, bool en);
void pio_sm_clear_fifos(PIO p, uint sm);
void pio_restart_sm_mask(PIO p, uint32_t mask);
void pio_sm_exec(PIO p, uint sm, uint32_t instr);
uint pio_encode_jmp(uint addr);
void pio_enable_sm_mask_in_sync(PIO p, uint32_t mask);
uint pio_add_program(PIO p, const void *prog);
uint pio_claim_unused_sm(PIO p, bool required);
