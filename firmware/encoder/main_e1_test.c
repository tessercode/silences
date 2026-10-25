/*
 * main_e1_test.c — E.1 Renode/silicon runner for the sentence-keying encoder.
 *
 * Includes the SHIPPING encoder_gpio.c in-TU (main renamed, never called — the A2.1/A4.1/B2b.1
 * pattern), drives every baked INPUT (fist, seed, text) through the real path
 *      enc_clean_text -> count_spikes guard -> encoder_render_sentence_jit
 * and dumps the computed tick/channel array over uart0 for the host comparator.
 *
 * ⚠️ WHAT RENODE CAN AND CANNOT COVER (stated plainly; see e1_port.md §5).
 * Renode's RP2040 platform models only uart0 and CANNOT observe GPIO edges, so THE EMITTED WAVEFORM
 * IS NOT OBSERVABLE IN EMULATION. What this runner proves on real ARM is everything UPSTREAM of the
 * pin: the RNG stream, the fist arithmetic (soft-float doubles on M0+), enc_clean_text, the over-cap
 * guard, and the tick sequence AS COMPUTED — i.e. emit_gpio's INPUT. The GPIO emission itself
 * (busy_wait_us scheduling, pin timing, the CAL pulse on the wire) is E.2's alone.
 *
 * ⚠️ TRANSPORT. The seam has bitten this project four times (putchar / \r\n / fputs / host-side
 * cp1252), so every byte here goes out via putchar_raw: no stdio newline translation, no pico_stdio
 * wrapping question, no locale. The payload is plain ASCII decimal and the comparator re-parses it.
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "pico/stdlib.h"

#define main encoder_gpio_fw_main
#include "encoder_gpio.c"
#undef main

#include "e1_vectors.h"

/* Raw, unwrapped byte out. putchar_raw bypasses pico_stdio's CR/LF translation entirely. */
static void wr(const char *s) {
    while (*s) putchar_raw(*s++);
}

static void wr_u32(uint32_t v) {
    char b[12];
    int i = 0;
    if (!v) { putchar_raw('0'); return; }
    while (v) { b[i++] = (char)('0' + (v % 10u)); v /= 10u; }
    while (i) putchar_raw(b[--i]);
}

int main(void) {
    stdio_init_all();
    sleep_ms(2000);                       /* let the host attach before the stream starts */
    wr("E1BEGIN\n");

    for (int k = 0; k < E1_N_VEC; k++) {
        const e1_vec_t *v = &E1_VEC[k];
        /* drive the SHIPPING globals exactly as the console would: :fist N, :jit 1, :seed X */
        g_fist = v->fist_index;
        g_jit  = 1;
        g_seed = v->seed;

        static char cleaned[ENC_MAX_TEXT_LEN + 1];
        int mapped;
        int tlen = enc_clean_text(v->text, (int)strlen(v->text), cleaned, ENC_MAX_TEXT_LEN, &mapped);
        int nsp = 0;
        int n = (tlen > 0) ? enc_render_text(cleaned, &nsp) : -2;

        wr("V ");  wr_u32(v->vid);
        if (n < 0) {                       /* over-cap REJECTED: report it, emit no data */
            wr(" REJECT "); wr_u32((uint32_t)nsp); wr("\n");
            continue;
        }
        wr(" "); wr_u32((uint32_t)n); wr("\n");
        for (int i = 0; i < n; i++) {
            wr_u32(g_tick[i]);
            putchar_raw(' ');
            putchar_raw((char)('0' + g_ch[i]));
            putchar_raw('\n');
        }
    }
    wr("E1END\n");
    while (true) tight_loop_contents();
}
