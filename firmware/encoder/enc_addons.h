/*
 * enc_addons.h — OUTPUT-ONLY creature comforts for the completion-spike encoder.
 *
 *   Feature 1  (ENC_AUDIO)      : Morse sidetone on a PCM5102 I2S DAC.
 *   Feature 2  (ENC_CHAR_UART)  : one ASCII byte per source character, 9600 8N1, on a monitor UART.
 *
 * Both are pure OBSERVERS of the emission the encoder already computed. They read g_tick[]/g_ch[]
 * (timing + channel) and the cleaned text (walked through the PUBLIC Morse table for char identity),
 * and drive FREE peripherals (I2S on pio0 + DMA + core1; a monitor UART) so the validated GP6/GP7
 * completion-spike path and its busy_wait timing are untouched. They draw NOTHING from the render RNG.
 *
 * The whole feature set is compiled ONLY when ENC_AUDIO and/or ENC_CHAR_UART is defined (CMake option
 * ENC_ADDONS). With neither defined, encoder_gpio.c does not reference this header at all and the
 * shipping image is bit-identical to the E.2-validated encoder_gpio.elf.
 *
 * Integration is exactly two guarded call sites in encoder_gpio.c and NOTHING else:
 *   - enc_addons_init()  once at boot (after GPIO init): brings up the peripherals + launches core1.
 *   - enc_addons_start() in process_line, AFTER enc_render_text() and BEFORE emit_gpio() — it builds
 *     the observer schedule and hands it to core1, then returns. emit_gpio() itself is byte-for-byte
 *     unchanged; core0's spike loop runs alone.
 */
#ifndef ENC_ADDONS_H
#define ENC_ADDONS_H
#include <stdint.h>

/* Boot-time bring-up (core0): init the monitor UART and/or the I2S clock, and launch the core1
 * playback worker. Safe to call once. */
void enc_addons_init(void);

/* Hand a just-rendered transmission to the core1 observer. Reads text (for char identity/boundaries)
 * and tick/ch (for timing) READ-ONLY, builds a time-ordered schedule, signals core1, and returns
 * immediately. `wpm`/`r` give the NOMINAL on-length (dot = 1*T_u, dash = r*T_u) used to place each
 * tone's onset before its jittered completion time tick[i]. */
void enc_addons_start(const char *text, const uint32_t *tick, const uint8_t *ch,
                      int n, double wpm, double r);

#endif /* ENC_ADDONS_H */
