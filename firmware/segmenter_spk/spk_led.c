/*
 * spk_led.c — identity-LED shim for the spiking_gap9_spk drop-in (ADDS the LED call; no compute code).
 * Same arrangement as spiking_gap9_lif/lif_led.c: the only clf_hw translation unit that includes
 * status_led.h. Violet identity, once at boot on GP23 via pio1, distinct from the current-fed LIF's orange
 * and the MLP's white so the three segmenter images can be told apart on the bench.
 */
#include "status_led.h"

void board_identity_led(void) {
    status_led_init();
    status_led_set_identity(MODULE_GAP9_SPK);
}
