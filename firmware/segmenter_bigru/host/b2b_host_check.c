/*
 * b2b_host_check.c — B2b.1 HOST SELF-CHECK (THE GATE).
 *
 * Compiles the SHIPPING gap9_hw.c in-TU (the A2.1 host_main.c pattern) against mock pico headers.
 * Only the pico layer is mocked; everything under test is the literal shipping code — fa_feed (the
 * widened u16 assembler), the MODULE dispatch, process_marks_frame (the widened 0x06 parse),
 * gap9_run_marks (A2's silicon-validated compute), emit_result_0x03 and emit_frame (the widened,
 * argmax-free writer).
 *
 * END-TO-END BY CONSTRUCTION: the frozen B2.0 golden carries BOTH ends — expected_0x06_hex in,
 * expected_0x03_hex out. Bytes go in one at a time through the real assembler; the mock
 * uart_write_blocking captures what comes out; the comparator demands byte equality.
 *
 * A mismatch localizes: wrong lp3 => the parse; right lp3 wrong bytes => the writer.
 *
 * Build: gcc -O2 -fsanitize=address,undefined -fno-sanitize-recover=all
 *            -Ihost/mock -I. -DGAP9_3C_N_SEEDS={1,5}
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- the capture sink the mock uart_write_blocking calls (declared in mock hardware/uart.h) ---- */
static unsigned char g_cap[8192];
static int g_cap_n = 0;
void host_uart_capture(const unsigned char *src, int len) {
    if (g_cap_n + len > (int)sizeof g_cap) { fprintf(stderr, "capture overflow\n"); exit(2); }
    memcpy(g_cap + g_cap_n, src, (size_t)len);
    g_cap_n += len;
}

/* board_identity_led() lives in gap9_led.c (the ONLY TU that includes status_led.h — the shared
 * enum's MODULE_GAP9 would collide with gap9_hw.c's 0x03 frame id). The LED is a boot-time GPIO
 * write on pio1, not part of the byte path: stub it rather than mock the WS2812 PIO stack. */
void board_identity_led(void) {}

/* the SHIPPING files, in-TU. Only gap9_hw.c's main is renamed, so this harness owns main(). */
#define main gap9_hw_fw_main
#include "../gap9_hw.c"
#undef main
#include "../gap9_3c.c"

static int hex2bin(const char *h, unsigned char *out, int max) {
    int n = (int)strlen(h) / 2;
    if (n > max) return -1;
    for (int i = 0; i < n; i++) {
        unsigned v;
        if (sscanf(h + 2 * i, "%2x", &v) != 1) return -1;
        out[i] = (unsigned char)v;
    }
    return n;
}

/* B2a: every byte now goes through the SHIPPING RING — pushed as the ISR does, drained as the main
 * loop does. BURSTY BY DESIGN: bytes accumulate before the consumer runs, which is the whole point
 * of B2a (the ISR fills while the main loop is deaf). The burst size varies so both the trickle and
 * the backlog cases run. THE EXPECTATIONS BELOW ARE UNCHANGED FROM B2b.1 — that is the transparency
 * proof: B2a changes the byte SOURCE, not the bytes. */
static int g_burst = 1;
static void feed_via_ring(uint8_t b) {
    rx_ring_push(&g_ring, b);                      /* the ISR's job */
    if ((int)rx_ring_count(&g_ring) >= g_burst) {  /* the main loop's job */
        uint8_t x;
        while (rx_ring_pop(&g_ring, &x)) rx_feed_byte(x);
    }
}
static void feed_flush(void) { uint8_t x; while (rx_ring_pop(&g_ring, &x)) rx_feed_byte(x); }

/* Feed a whole frame's bytes through the SHIPPING RX path; return the captured emission. */
static int drive(const unsigned char *f06, int n06) {
    g_cap_n = 0;
    for (int i = 0; i < n06; i++) feed_via_ring(f06[i]);
    feed_flush();
    return g_cap_n;
}

static char line[262144];

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s golden_frames_nX.json\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "r");
    if (!f) { perror("golden"); return 2; }

    fa_reset(&g_fa);
    rx_ring_init(&g_ring);

    static unsigned char f06[8192], want03[8192];
    int n = 0, bad = 0, max_marks = 0, max_03 = 0, n_cap_marks = 0;
    int vid = -1, nsp = -1, ngaps = -1;

    /* The golden is one JSON object per vector across several lines; scan for the fields we need.
     * (Deliberately dumb: no JSON lib in the firmware tree; the fields are one-per-line.) */
    while (fgets(line, sizeof line, f)) {
        char *p;
        if ((p = strstr(line, "\"vector_id\":"))) vid = atoi(p + 12);
        else if ((p = strstr(line, "\"n_spikes\":"))) nsp = atoi(p + 11);
        else if ((p = strstr(line, "\"n_gaps\":"))) ngaps = atoi(p + 9);
        else if ((p = strstr(line, "\"expected_0x06_hex\":"))) {
            char *q = strchr(p + 20, '"') + 1, *e = strchr(q, '"');
            *e = 0;
            int k = hex2bin(q, f06, sizeof f06);
            if (k < 0) { fprintf(stderr, "vec %d: 0x06 hex too long\n", vid); return 2; }
            if (nsp > max_marks) max_marks = nsp;
            if (nsp == 256) n_cap_marks++;
            /* stash length via a static: the next line carries the 0x03 */
            static int n06s;
            n06s = k;
            /* read on to the 0x03 line */
            while (fgets(line, sizeof line, f)) {
                if ((p = strstr(line, "\"expected_0x03_hex\":"))) {
                    char *q3 = strchr(p + 20, '"') + 1, *e3 = strchr(q3, '"');
                    *e3 = 0;
                    int w = hex2bin(q3, want03, sizeof want03);
                    if (w < 0) { fprintf(stderr, "vec %d: 0x03 hex too long\n", vid); return 2; }
                    if (w > max_03) max_03 = w;
                    g_burst = 1 + (n % 400);          /* 1..400 bytes buffered before draining */
                    int got = drive(f06, n06s);
                    n++;
                    if (got != w || memcmp(g_cap, want03, (size_t)w) != 0) {
                        bad++;
                        if (bad <= 3) {
                            fprintf(stderr, "MISMATCH vec %d (n_spikes=%d n_gaps=%d): "
                                    "emitted %d B, want %d B\n", vid, nsp, ngaps, got, w);
                            int lim = got < w ? got : w;
                            for (int i = 0; i < lim; i++)
                                if (g_cap[i] != want03[i]) {
                                    fprintf(stderr, "  first differing byte %d: got %02x want %02x\n",
                                            i, g_cap[i], want03[i]);
                                    break;
                                }
                        }
                    }
                    break;
                }
            }
        }
    }
    fclose(f);
    printf("G1 end-to-end 0x06 bytes -> 0x03 bytes: %d/%d EXACT%s\n", n - bad, n,
           bad ? "  ** FAIL **" : "");
    printf("    bounds reached: max n_marks %d (%d vectors at the 256 cap), max 0x03 on-wire %d B\n",
           max_marks, n_cap_marks, max_03);
    if (bad || n != 117) { fprintf(stderr, "expected 117 vectors, got %d\n", n); return 1; }

    /* ---- G2: byte-stream robustness — a corrupted 0x06 is CRC-rejected AND fa_feed resyncs ---- */
    {
        /* rebuild vector 0's frame, corrupt one payload byte, then send a GOOD frame after it. */
        rewind(f = fopen(argv[1], "r"));
        unsigned char good06[8192], good03[8192];
        int n06 = 0, n03 = 0;
        while (fgets(line, sizeof line, f)) {
            char *p;
            if ((p = strstr(line, "\"expected_0x06_hex\":")) && !n06) {
                char *q = strchr(p + 20, '"') + 1, *e = strchr(q, '"'); *e = 0;
                n06 = hex2bin(q, good06, sizeof good06);
            } else if ((p = strstr(line, "\"expected_0x03_hex\":")) && !n03) {
                char *q = strchr(p + 20, '"') + 1, *e = strchr(q, '"'); *e = 0;
                n03 = hex2bin(q, good03, sizeof good03);
                break;
            }
        }
        fclose(f);
        fa_reset(&g_fa); rx_ring_init(&g_ring);
        unsigned char bad06[8192];
        memcpy(bad06, good06, (size_t)n06);
        bad06[10] ^= 0x01;                       /* flip a bit inside the payload */
        g_cap_n = 0;
        for (int i = 0; i < n06; i++) feed_via_ring(bad06[i]);
        feed_flush();
        int emitted_after_corrupt = g_cap_n;
        /* now a clean frame must still be assembled -> fa_feed resynced */
        g_cap_n = 0;
        for (int i = 0; i < n06; i++) feed_via_ring(good06[i]);
        feed_flush();
        int emitted_after_resync = g_cap_n;
        printf("G2 corrupted 0x06: emitted %d B (want 0 = CRC-rejected); "
               "next clean frame: emitted %d B (want %d = resynced) -> %s\n",
               emitted_after_corrupt, emitted_after_resync, n03,
               (emitted_after_corrupt == 0 && emitted_after_resync == n03 &&
                memcmp(g_cap, good03, (size_t)n03) == 0) ? "OK" : "** FAIL **");
        if (emitted_after_corrupt != 0 || emitted_after_resync != n03 ||
            memcmp(g_cap, good03, (size_t)n03) != 0) return 1;
    }

    /* ---- G3: over-cap input hits the drop, not a smash ---- */
    {
        /* A 257-mark 0x06: payload 9 + 5*257 = 1294 B, LENGTH 0x050E, n_marks 0x0101 = 257.
         * fa_feed accepts it (1294 <= ROUTER_MAX_PL); process_marks_frame must DROP it at the
         * `ns > MAX_SPIKES` guard. ASan proves nothing was smashed. */
        static unsigned char big[2048];
        int n_marks = 257, plen = 9 + 5 * n_marks;
        unsigned char pl[2048];
        int o = 0;
        pl[o++] = 0x01;
        pl[o++] = 0x00; pl[o++] = 0x63;                       /* seq = 99 */
        pl[o++] = 0x00; pl[o++] = 0x00; pl[o++] = 0x75; pl[o++] = 0x30;   /* tu = 30000 */
        pl[o++] = (unsigned char)(n_marks >> 8); pl[o++] = (unsigned char)n_marks;
        for (int i = 0; i < n_marks; i++) {
            int t = 1000 * (i + 1);
            pl[o++] = (unsigned char)(t >> 24); pl[o++] = (unsigned char)(t >> 16);
            pl[o++] = (unsigned char)(t >> 8);  pl[o++] = (unsigned char)t;
            pl[o++] = (unsigned char)(i & 1);
        }
        if (o != plen) { fprintf(stderr, "over-cap frame build error\n"); return 2; }
        int b = 0;
        big[b++] = 0xA5; big[b++] = 0x5A; big[b++] = 0x06;
        big[b++] = (unsigned char)(plen >> 8); big[b++] = (unsigned char)plen;
        memcpy(big + b, pl, (size_t)plen); b += plen;
        unsigned char hdr[3] = {0x06, (unsigned char)(plen >> 8), (unsigned char)plen};
        unsigned char st[2048];
        memcpy(st, hdr, 3); memcpy(st + 3, pl, (size_t)plen);
        big[b++] = crc8(st, 3 + plen);
        fa_reset(&g_fa);
        g_cap_n = 0;
        for (int i = 0; i < b; i++) feed_via_ring(big[i]);
        feed_flush();
        printf("G3 over-cap 257-mark 0x06 (CRC-valid, %d B payload): emitted %d B "
               "(want 0 = dropped at the ns > MAX_SPIKES guard) -> %s\n",
               plen, g_cap_n, g_cap_n == 0 ? "OK" : "** FAIL **");
        if (g_cap_n != 0) return 1;
    }

    /* ---- G4: an over-cap LENGTH is rejected by the assembler itself ---- */
    {
        fa_reset(&g_fa);
        g_cap_n = 0;
        feed_via_ring(0xA5); feed_via_ring(0x5A); feed_via_ring(0x06);
        feed_via_ring(0xFF); feed_via_ring(0xFF);            /* LENGTH 65535 > ROUTER_MAX_PL */
        for (int i = 0; i < 100; i++) feed_via_ring(0x00);   /* must not write past fa.pl */
        feed_flush();
        printf("G4 over-cap LENGTH 65535: emitted %d B (want 0), assembler resynced -> %s\n",
               g_cap_n, g_cap_n == 0 ? "OK" : "** FAIL **");
        if (g_cap_n != 0) return 1;
    }

    printf("ALL HOST GATES PASS (%s)\n", argv[1]);
    return 0;
}
