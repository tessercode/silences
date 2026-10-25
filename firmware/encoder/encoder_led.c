/*
 * encoder_led.c — identity-LED shim (ADDS the LED call; contains no compute/render code).
 *
 * This is the ONLY encoder translation unit that includes status_led.h. Keeping it separate isolates
 * the shared enum constant MODULE_ENCODER from encoder_frame.h's `#define MODULE_ENCODER 0x04u` (the
 * frame module id) — the two same-named symbols never appear in one TU, so no existing encoder file
 * (frame builder, render, main compute) is changed. Green identity, driven once at boot on GP23/pio1.
 */
#include "status_led.h"

void board_identity_led(void) {
    status_led_init();
    status_led_set_identity(MODULE_ENCODER);
}
