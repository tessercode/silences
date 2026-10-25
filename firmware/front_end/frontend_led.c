/*
 * frontend_led.c — identity-LED shim (ADDS the LED call; contains no compute code).
 *
 * The only front-end translation unit that includes status_led.h. Isolating it keeps the shared
 * MODULE_* enum out of frontend_stage3.c / frontend.c, so those validated files are untouched except
 * for the single boot-time call. Blue identity, driven once at boot on GP23 via pio1 (the capture
 * PIO uses pio0 SM_dot/SM_dash — no overlap).
 */
#include "status_led.h"

void board_identity_led(void) {
    status_led_init();
    status_led_set_identity(MODULE_FRONTEND);
}
