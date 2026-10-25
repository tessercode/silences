/*
 * mlp_led.c — identity-LED shim for the spiking_gap9_mlp drop-in (ADDS the LED call; no compute code).
 *
 * The only clf_hw translation unit that includes status_led.h, so the shared enum constant
 * MODULE_GAP9_MLP cannot collide with clf_hw.c's `#define MODULE_GAP9 0x03` (the frame module id) —
 * the two never appear in one TU. Distinct from gap9's magenta and the LIF drop-in's orange so the
 * MLP drop-in is visually identifiable on the bench. White identity, once at boot on GP23 via pio1.
 */
#include "status_led.h"

void board_identity_led(void) {
    status_led_init();
    status_led_set_identity(MODULE_GAP9_MLP);
}
