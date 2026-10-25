/*
 * gap9_led.c — identity-LED shim (ADDS the LED call; contains no segmenter/compute code).
 *
 * The only gap9 translation unit that includes status_led.h. Keeping it separate isolates the shared
 * enum constant MODULE_GAP9 from gap9_hw.c's `#define MODULE_GAP9 0x03` (the frame module id) — the two
 * same-named symbols never appear in one TU, so no existing gap9 file (the UNCHANGED gap9.c/gap9_int8.c
 * segmenter, the wrapper, the GP5-bracketed compute) is modified. Magenta identity, once at boot on
 * GP23 via pio1.
 */
#include "status_led.h"

void board_identity_led(void) {
    status_led_init();
    status_led_set_identity(MODULE_GAP9);
}
