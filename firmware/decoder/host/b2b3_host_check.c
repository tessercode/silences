/*
 * b2b3_host_check.c — B2b.3 HOST SELF-CHECK (THE GATE). Three frozen goldens meet here.
 *
 * IN : b2_frame_golden's expected_0x05_hex                      (B2.0, frozen)
 * L1 : a2_gap9_3c_golden's frozen ch[]/lp3[]                    (A2, frozen — the parse truth)
 * L2 : a4_beam_golden's decoded_text/seg/final_score_q16        (A4, frozen, silicon-validated)
 * Nothing new defines truth at either end; this step only has to not corrupt the bytes between them.
 * All three are SHA-verified and transcribed into b2b3_vectors.h by gen_b2b3_vectors.py, which also
 * CROSS-CHECKS that the baked L1 truth is byte-identical to what the 0x05 frames carry — so L1
 * cannot pass vacuously through a transcription slip.
 *
 * TWO LEVELS, because the second alone can mask a parse bug:
 *   L1 PARSE  — the parsed ch[]/lp3[] must equal the frozen A2 values. A wrong lp3 that still
 *               decodes plausibly would slip past L2; L1 localizes it to the parser.
 *   L2 DECODE — the emitted result line must equal A4's golden, including vector 218's beam death
 *               at n1 and its ensemble-rescue at n5.
 *
 * Compiles the SHIPPING files in-TU (the A2.1/A4.1 pattern) against mock pico headers; only the pico
 * layer is mocked. Bytes go in ONE AT A TIME through the real fa_feed.
 *
 * Build: gcc -O2 -fsanitize=address,undefined -fno-sanitize-recover=all -Ihost/mock -I. -I../common
 *            -DCHARLM_EMBED=1
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* the USB-CDC sink the mock putchar_raw feeds: this capture IS the emitted result line */
static char g_line[4096];
static int  g_line_n;
static int  g_lines;
void host_putchar_raw(int c) {
    if (g_line_n < (int)sizeof g_line - 1) g_line[g_line_n++] = (char)c;
    if (c == '\n') g_lines++;
}

/* The identity LED is a boot-time WS2812 write on pio1, not part of the byte path: stub its two
 * entry points rather than mock the whole PIO stack. (`status_led.h` sits next to output_board.c, so
 * a quoted include resolves to the REAL header regardless of -I order — the definitions are what
 * must be supplied, not the declarations.) */
#include "../status_led.h"
void status_led_init(void) {}
void status_led_set_identity(module_id_t m) { (void)m; }

/* the SHIPPING files, in-TU. Only output_board.c's main is renamed. */
#define main decode_board_fw_main
#include "../output_board.c"
#undef main
#include "../beam.c"
#include "../charlm_lookup.c"

#include "../b2b3_vectors.h"      /* the three frozen goldens, baked (gen_b2b3_vectors.py) */

/* B2a: every byte now goes through the SHIPPING RING — pushed as the ISR does, drained as the main
 * loop does (via on_rx_bytes, the seam, exactly as the firmware's loop calls it). Bursty by design.
 * THE EXPECTATIONS ARE UNCHANGED FROM B2b.3 — that is the transparency proof. */
static int g_burst = 1;
static void feed_via_ring(uint8_t b) {
    rx_ring_push(&g_ring, b);                          /* the ISR's job */
    if ((int)rx_ring_count(&g_ring) >= g_burst) {      /* the main loop's job, verbatim */
        uint8_t buf[64];
        int n = 0;
        while (n < (int)sizeof buf && rx_ring_pop(&g_ring, &buf[n])) n++;
        if (n) on_rx_bytes(buf, n);
    }
}
static void feed_flush(void) {
    uint8_t buf[64];
    int n;
    do {
        n = 0;
        while (n < (int)sizeof buf && rx_ring_pop(&g_ring, &buf[n])) n++;
        if (n) on_rx_bytes(buf, n);
    } while (n);
}

int main(void) {
    int bad1 = 0, bad2 = 0, max_nsp = 0, max_05 = 0, n_cap = 0, n_died = 0;
    static char want[8192];

    for (int k = 0; k < B3_N_VEC; k++) {
        const uint8_t *f = &B3_F05[B3_F05_OFF[k]];
        const int n05 = B3_F05_LEN[k];
        fa_reset(&g_fa); rx_ring_init(&g_ring);
        g_line_n = 0; g_lines = 0;
        memset(&g_fwd, 0, sizeof g_fwd);
        g_burst = 1 + (k % 400);                                /* B2a: burst sizes 1..400 */
        for (int i = 0; i < n05; i++) feed_via_ring(f[i]);      /* the SHIPPING path, byte by byte */
        feed_flush();

        if (g_lines != 1) {
            fprintf(stderr, "vec %ld: emitted %d result lines, want 1\n", (long)B3_VID[k], g_lines);
            return 1;
        }

        /* ---- L1 PARSE: parsed ch[]/lp3[] vs the FROZEN A2 values ---- */
        {
            const uint8_t *rch = &B3_REF_CH[B3_REF_CH_OFF[k]];
            const int32_t *rlp = &B3_REF_LP3[B3_REF_LP3_OFF[k]];
            int ok = ((int)g_fwd.n_spikes == B3_NSPIKES[k]) && ((int)g_fwd.n_gaps == B3_NGAPS[k]) &&
                     (memcmp(g_fwd.ch, rch, (size_t)B3_NSPIKES[k]) == 0) &&
                     (memcmp(g_fwd.lp3, rlp, (size_t)B3_NGAPS[k] * 3 * sizeof(int32_t)) == 0);
            if (!ok) {
                bad1++;
                if (bad1 <= 3) {
                    fprintf(stderr, "L1 MISMATCH vec %ld: parsed n_spikes=%u/%ld n_gaps=%u/%ld\n",
                            (long)B3_VID[k], g_fwd.n_spikes, (long)B3_NSPIKES[k],
                            g_fwd.n_gaps, (long)B3_NGAPS[k]);
                    for (int t = 0; t < B3_NGAPS[k] * 3; t++)
                        if (g_fwd.lp3[t] != rlp[t]) {
                            fprintf(stderr, "  first differing lp3[%d]: parsed %ld want %ld\n",
                                    t, (long)g_fwd.lp3[t], (long)rlp[t]);
                            break;
                        }
                }
            }
        }

        /* ---- L2 DECODE: the emitted line vs A4's frozen golden ---- */
        g_line[g_line_n] = 0;
        if (B3_DIED[k]) {
            snprintf(want, sizeof want, "DEC %ld %ld 1\n", (long)B3_VID[k], (long)B3_NGAPS[k]);
            n_died++;
        } else {
            snprintf(want, sizeof want, "DEC %ld %ld 0 %s %s %lld\n", (long)B3_VID[k],
                     (long)B3_NGAPS[k], B3_TEXT[k], B3_SEG[k], (long long)B3_SCORE[k]);
        }
        if (strcmp(g_line, want) != 0) {
            bad2++;
            if (bad2 <= 3)
                fprintf(stderr, "L2 MISMATCH vec %ld:\n  got  %s  want %s",
                        (long)B3_VID[k], g_line, want);
        }
        if (B3_NSPIKES[k] > max_nsp) max_nsp = B3_NSPIKES[k];
        if (n05 > max_05) max_05 = n05;
        if (B3_NSPIKES[k] == 256) n_cap++;
    }
    printf("L1 parse  (0x05 bytes -> ch[]/lp3[] vs the FROZEN A2 values): %d/%d EXACT%s\n",
           B3_N_VEC - bad1, B3_N_VEC, bad1 ? "  ** FAIL **" : "");
    printf("L2 decode (0x05 bytes -> text/seg/score vs the FROZEN A4 golden): %d/%d EXACT%s\n",
           B3_N_VEC - bad2, B3_N_VEC, bad2 ? "  ** FAIL **" : "");
    printf("    bounds: max n_spikes %d (%d at the 256 cap), max 0x05 on-wire %d B, %d beam death(s)\n",
           max_nsp, n_cap, max_05, n_died);
    if (bad1 || bad2) return 1;

    /* ---- R1: a corrupted 0x05 is CRC-rejected, and fa_feed resyncs to the next frame ---- */
    {
        const uint8_t *f = &B3_F05[B3_F05_OFF[0]];
        const int n05 = B3_F05_LEN[0];
        static uint8_t bad05[8192];
        memcpy(bad05, f, (size_t)n05);
        bad05[10] ^= 0x01;                          /* flip a payload bit */
        fa_reset(&g_fa); rx_ring_init(&g_ring); g_line_n = 0; g_lines = 0;
        for (int i = 0; i < n05; i++) feed_via_ring(bad05[i]);
        feed_flush();
        int after_corrupt = g_lines;
        g_line_n = 0; g_lines = 0;
        for (int i = 0; i < n05; i++) feed_via_ring(f[i]);
        feed_flush();
        g_line[g_line_n] = 0;
        snprintf(want, sizeof want, "DEC %ld %ld 0 %s %s %lld\n", (long)B3_VID[0],
                 (long)B3_NGAPS[0], B3_TEXT[0], B3_SEG[0], (long long)B3_SCORE[0]);
        int resync = (g_lines == 1 && strcmp(g_line, want) == 0);
        printf("R1 corrupted 0x05: %d result lines (want 0 = CRC-rejected); next clean frame -> %s\n",
               after_corrupt, resync ? "DECODED byte-exact (fa_feed resynced)" : "** FAIL **");
        if (after_corrupt != 0 || !resync) return 1;
    }

    /* ---- R2: an over-cap LENGTH drops without writing past fa.pl ---- */
    {
        fa_reset(&g_fa); rx_ring_init(&g_ring); g_line_n = 0; g_lines = 0;
        feed_via_ring(0xA5); feed_via_ring(0x5A); feed_via_ring(0x05);
        feed_via_ring(0xFF); feed_via_ring(0xFF);                /* LENGTH 65535 > ROUTER_MAX_PL */
        for (int i = 0; i < 200; i++) feed_via_ring(0x00);
        feed_flush();
        printf("R2 over-cap LENGTH 65535 + 200 bytes: %d result lines (want 0) -> %s\n",
               g_lines, g_lines == 0 ? "OK (ASan clean, no write past pl)" : "** FAIL **");
        if (g_lines != 0) return 1;
    }

    /* ---- R3: CRC-VALID but structurally wrong — decode_forward's own rejection paths ----
     * A CRC proves the bytes are the ones that were sent, not that they mean anything. */
    {
        struct { const char *name; int nsp_field; int len; } cases[] = {
            { "R3 KIND != 0x01 (the EOP sentinel)",   0,  1 },
            { "R4 n_spikes < 2",                      1,  6 },
            { "R6 len too short for its n_spikes",   10, 40 },
            { "R6 len says 4 gaps, n_spikes says 10", 10, 5 + 10 + 12 * 4 },
        };
        int rejected = 0, total = 0;
        for (unsigned c = 0; c < sizeof cases / sizeof cases[0]; c++) {
            static uint8_t pl[ROUTER_MAX_PL], frame[ROUTER_MAX_PL + 8], st[ROUTER_MAX_PL + 4];
            const int L = cases[c].len;
            memset(pl, 0, (size_t)L);
            pl[0] = (c == 0) ? ROUTER_KIND_EOP : ROUTER_KIND_RESULT;
            if (L >= 5) {
                pl[1] = 0; pl[2] = 7;                                   /* vid */
                pl[3] = (uint8_t)(cases[c].nsp_field >> 8);
                pl[4] = (uint8_t)cases[c].nsp_field;
            }
            int b = 0;
            frame[b++] = 0xA5; frame[b++] = 0x5A; frame[b++] = ROUTER_MOD_FORWARD;
            frame[b++] = (uint8_t)(L >> 8); frame[b++] = (uint8_t)L;
            memcpy(frame + b, pl, (size_t)L); b += L;
            st[0] = ROUTER_MOD_FORWARD; st[1] = (uint8_t)(L >> 8); st[2] = (uint8_t)L;
            memcpy(st + 3, pl, (size_t)L);
            frame[b++] = rtr_crc8(st, 3 + L);                           /* a VALID CRC */
            fa_reset(&g_fa); rx_ring_init(&g_ring); g_line_n = 0; g_lines = 0;
            for (int i = 0; i < b; i++) feed_via_ring(frame[i]);
            feed_flush();
            total++;
            if (g_lines == 0) rejected++;
            else fprintf(stderr, "  %s NOT rejected\n", cases[c].name);
        }
        /* R5 n_spikes > MAXSP: a 300-mark frame's payload (3,893 B) exceeds ROUTER_MAX_PL, so
         * fa_feed's own LENGTH guard drops it one layer EARLIER than decode_forward's R5. Call R5
         * directly so the guard is actually tested rather than shadowed by the outer one. */
        {
            static uint8_t pl5[16];
            static fwd_t f5;
            pl5[0] = ROUTER_KIND_RESULT; pl5[1] = 0; pl5[2] = 7;
            pl5[3] = 0x01; pl5[4] = 0x2C;                               /* n_spikes = 300 > 256 */
            int r = decode_forward(ROUTER_MOD_FORWARD, pl5, 5 + 300 + 12 * 299, &f5);
            total++;
            if (r == 0) rejected++;
            else fprintf(stderr, "  R5 n_spikes > MAXSP NOT rejected\n");
        }
        /* R1 wrong module: the FE's 0x06 arriving on this line must not be parsed as a 0x05 */
        {
            static uint8_t pl1[16];
            static fwd_t f1;
            pl1[0] = ROUTER_KIND_RESULT;
            int r = decode_forward(ROUTER_MOD_FE_FWD, pl1, 9, &f1);
            total++;
            if (r == 0) rejected++;
            else fprintf(stderr, "  R1 wrong module NOT rejected\n");
        }
        printf("R3 CRC-valid but structurally wrong (R1/R3/R4/R5/R6): %d/%d rejected -> %s\n",
               rejected, total, rejected == total ? "OK" : "** FAIL **");
        if (rejected != total) return 1;
    }

    printf("ALL HOST GATES PASS (B3_NSEEDS=%d)\n", (int)B3_NSEEDS);
    return 0;
}
