/*
 * b2b2_host_check.c — B2b.2 HOST SELF-CHECK (THE GATE).
 *
 * Compiles the SHIPPING firmware/common/router_core.h in-TU (the router_host.c pattern: the same
 * header the firmware runs, fed a replayed (input_idx, byte) interleaving). Nothing under test is
 * mocked — fa_feed, decode_half, router_ingest and rtr_build_forward are the literal shipping code.
 *
 * END-TO-END BY CONSTRUCTION: the frozen B2.0 golden carries all three frames per vector —
 * expected_0x06_hex + expected_0x03_hex IN (fed BYTE BY BYTE, INTERLEAVED across the two input
 * lines, exactly as the two PIO receivers hand them to the core), expected_0x05_hex OUT.
 *
 * Build: gcc -O2 -fsanitize=address,undefined -fno-sanitize-recover=all -I../../common
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "router_core.h"        /* the SHIPPING shared header, in-TU */
#include "rx_ring.h"            /* B2a: the SHIPPING shared ring, in-TU */

/* ------------------------------------------------------------------ the two-line replay harness */
#define IN_FE   0
#define IN_GAP9 1

static fa_t     g_fa[2];
static router_t g_R;
static uint8_t  g_out[ROUTER_OUT_0x05_MAX];
static half_t   g_h;

/* the emitted 0x05 capture (one per completed pair) */
static uint8_t g_cap[ROUTER_OUT_0x05_MAX];
static int     g_cap_n, g_cap_count;

/* Feed ONE byte on ONE input line through the SHIPPING path — this mirrors router_hw.c's drain body
 * exactly, including the single shared `pl` (the firmware reuses one buffer across both lines
 * because a frame completes before the next byte is consumed). */
static uint8_t g_pl[ROUTER_MAX_PL];

/* B2a: every byte now goes through the SHIPPING RING — pushed as the ISR does (one ring per line),
 * drained as the main loop does. Bursty by design; the burst size varies across the pass. THE
 * EXPECTATIONS ARE UNCHANGED FROM B2b.2 — that is the transparency proof. */
static rx_ring_t g_rings_h[2];
static int g_burst = 1;
static void consume(int line);
static void feed2(int line, uint8_t b) {
    rx_ring_push(&g_rings_h[line], b);                       /* the ISR's job */
    if ((int)rx_ring_count(&g_rings_h[line]) >= g_burst) consume(line);   /* the main loop's job */
}
static void feed_flush(void) { consume(IN_FE); consume(IN_GAP9); }

static void feed_raw(int line, uint8_t b) {
    uint8_t mid;
    uint16_t len;
    if (fa_feed(&g_fa[line], b, &mid, g_pl, &len)) {
        if (decode_half(mid, g_pl, (int)len, &g_h)) {
            int fl = router_ingest(&g_R, &g_h, g_out);
            if (fl > 0) { memcpy(g_cap, g_out, (size_t)fl); g_cap_n = fl; g_cap_count++; }
        }
    }
}

static void consume(int line) {
    uint8_t x;
    while (rx_ring_pop(&g_rings_h[line], &x)) feed_raw(line, x);
}

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

/* ---- the golden reader (fields are one-per-line in the emitted JSON; no JSON lib in firmware) ---- */
typedef struct {
    int vid, n_spikes, n_gaps;
    unsigned char f06[8192], f03[8192], f05[8192];
    int n06, n03, n05;
} vec_t;

static char line[262144];

static int load_golden(const char *path, vec_t *v, int max) {
    FILE *f = fopen(path, "r");
    if (!f) { perror("golden"); exit(2); }
    int n = 0, vid = -1, nsp = -1, ng = -1;
    while (fgets(line, sizeof line, f)) {
        char *p;
        if ((p = strstr(line, "\"vector_id\":"))) vid = atoi(p + 12);
        else if ((p = strstr(line, "\"n_spikes\":"))) nsp = atoi(p + 11);
        else if ((p = strstr(line, "\"n_gaps\":"))) ng = atoi(p + 9);
        else if ((p = strstr(line, "\"expected_0x06_hex\":"))) {
            if (n >= max) break;
            v[n].vid = vid; v[n].n_spikes = nsp; v[n].n_gaps = ng;
            char *q = strchr(p + 20, '"') + 1, *e = strchr(q, '"'); *e = 0;
            v[n].n06 = hex2bin(q, v[n].f06, sizeof v[n].f06);
        } else if ((p = strstr(line, "\"expected_0x03_hex\":"))) {
            char *q = strchr(p + 20, '"') + 1, *e = strchr(q, '"'); *e = 0;
            v[n].n03 = hex2bin(q, v[n].f03, sizeof v[n].f03);
        } else if ((p = strstr(line, "\"expected_0x05_hex\":"))) {
            char *q = strchr(p + 20, '"') + 1, *e = strchr(q, '"'); *e = 0;
            v[n].n05 = hex2bin(q, v[n].f05, sizeof v[n].f05);
            n++;
        }
    }
    fclose(f);
    return n;
}

static void reset_all(void) {
    fa_reset(&g_fa[IN_FE]); fa_reset(&g_fa[IN_GAP9]);
    rx_ring_init(&g_rings_h[IN_FE]); rx_ring_init(&g_rings_h[IN_GAP9]);
    router_init(&g_R);
    g_cap_n = g_cap_count = 0;
}

/* Drive one vector with the two frames INTERLEAVED byte-by-byte. `order` 0 = 0x06 leads (the live
 * skew: the FE emits before gap9 has computed), 1 = 0x03 leads. Both must pair identically. */
static void drive_interleaved(const vec_t *v, int order) {
    const unsigned char *a = order ? v->f03 : v->f06;
    const unsigned char *b = order ? v->f06 : v->f03;
    int na = order ? v->n03 : v->n06, nb = order ? v->n06 : v->n03;
    int la = order ? IN_GAP9 : IN_FE, lb = order ? IN_FE : IN_GAP9;
    int i = 0, j = 0;
    while (i < na || j < nb) {                 /* strict alternation where both remain */
        if (i < na) feed2(la, a[i++]);
        if (j < nb) feed2(lb, b[j++]);
    }
    feed_flush();                              /* B2a: drain whatever the rings still hold */
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s golden_frames_nX.json\n", argv[0]); return 2; }
    static vec_t V[128];
    int N = load_golden(argv[1], V, 128);
    if (N != 117) { fprintf(stderr, "expected 117 vectors, got %d\n", N); return 2; }

    int max_nsp = 0, max_05 = 0, n_cap = 0;

    /* ---- G1: 117/117, both arrival orders, interleaved byte-by-byte ---- */
    for (int order = 0; order < 2; order++) {
        int bad = 0;
        for (int k = 0; k < N; k++) {
            reset_all();
            g_burst = 1 + (k % 400);           /* B2a: 1..400 bytes buffered before draining */
            drive_interleaved(&V[k], order);
            if (g_cap_count != 1 || g_cap_n != V[k].n05 ||
                memcmp(g_cap, V[k].f05, (size_t)V[k].n05) != 0) {
                bad++;
                if (bad <= 3) {
                    fprintf(stderr, "MISMATCH vec %d (order=%s, n_spikes=%d): emitted %d B x%d, "
                            "want %d B\n", V[k].vid, order ? "0x03-first" : "0x06-first",
                            V[k].n_spikes, g_cap_n, g_cap_count, V[k].n05);
                    for (int i = 0; i < (g_cap_n < V[k].n05 ? g_cap_n : V[k].n05); i++)
                        if (g_cap[i] != V[k].f05[i]) {
                            fprintf(stderr, "  first differing byte %d: got %02x want %02x\n",
                                    i, g_cap[i], V[k].f05[i]);
                            break;
                        }
                }
            }
            if (order == 0) {
                if (V[k].n_spikes > max_nsp) max_nsp = V[k].n_spikes;
                if (V[k].n05 > max_05) max_05 = V[k].n05;
                if (V[k].n_spikes == 256) n_cap++;
            }
        }
        printf("G1 %-11s 0x06+0x03 bytes (interleaved) -> 0x05 bytes: %d/%d EXACT%s\n",
               order ? "0x03-first:" : "0x06-first:", N - bad, N, bad ? "  ** FAIL **" : "");
        if (bad) return 1;
    }
    printf("    bounds reached: max n_spikes %d (%d vectors at the 256 cap), max 0x05 on-wire %d B\n",
           max_nsp, n_cap, max_05);

    /* ---- G2: EVICTION — 9 orphaned 0x06s, then a complete pair must still succeed ---- */
    {
        reset_all();
        for (int k = 0; k < 9 && k < N; k++)                  /* 9 marks-halves, no partners */
            for (int i = 0; i < V[k].n06; i++) feed2(IN_FE, V[k].f06[i]);
            feed_flush();
        int ev_after_orphans = g_R.evictions;
        drive_interleaved(&V[10], 0);                         /* now a COMPLETE pair */
        int ok = (g_cap_count == 1 && g_cap_n == V[10].n05 &&
                  memcmp(g_cap, V[10].f05, (size_t)V[10].n05) == 0);
        printf("G2 eviction: 9 orphaned 0x06s -> evictions=%d (want >=1), dropped_full=%d (want 0); "
               "then a complete pair -> %s\n", ev_after_orphans, g_R.dropped_full,
               ok ? "PAIRED, 0x05 byte-exact" : "** FAIL **");
        if (!(ev_after_orphans >= 1 && g_R.dropped_full == 0 && ok)) return 1;
    }

    /* ---- G3: the MODULAR WRAP — vids straddling 65535 -> 0 ----
     * Fill all 8 slots with orphan vids 65530..65535 and 0..1 (i.e. across the wrap), then ingest a
     * new half at vid 2. The genuinely oldest is 65530. A plain `<` comparison would call 0 the
     * smallest and evict IT — passing G2 but corrupting the table here. */
    {
        reset_all();
        const uint16_t vids[8] = {65530, 65531, 65532, 65533, 65534, 65535, 0, 1};
        for (int i = 0; i < 8; i++) {
            memset(&g_h, 0, sizeof g_h);
            g_h.vector_id = vids[i]; g_h.kind = ROUTER_HALF_MARKS; g_h.n_spikes = 4;
            router_ingest(&g_R, &g_h, g_out);
        }
        if (rtr_occupancy(&g_R) != 8) { fprintf(stderr, "G3 setup: table not full\n"); return 1; }
        memset(&g_h, 0, sizeof g_h);
        g_h.vector_id = 2; g_h.kind = ROUTER_HALF_MARKS; g_h.n_spikes = 4;
        router_ingest(&g_R, &g_h, g_out);                    /* must evict 65530, the oldest */
        int found_65530 = 0, found_2 = 0, found_0 = 0, found_1 = 0;
        for (int i = 0; i < ROUTER_NSLOTS; i++) {
            if (!g_R.slots[i].used) continue;
            if (g_R.slots[i].vector_id == 65530) found_65530 = 1;
            if (g_R.slots[i].vector_id == 2) found_2 = 1;
            if (g_R.slots[i].vector_id == 0) found_0 = 1;
            if (g_R.slots[i].vector_id == 1) found_1 = 1;
        }
        printf("G3 modular wrap (vids 65530..65535,0,1 full; ingest vid 2): evicted 65530=%s, "
               "vid 2 present=%s, post-wrap vids 0/1 retained=%s -> %s\n",
               found_65530 ? "NO" : "yes", found_2 ? "yes" : "NO",
               (found_0 && found_1) ? "yes" : "NO",
               (!found_65530 && found_2 && found_0 && found_1) ? "OK (a plain `<` fails here)"
                                                               : "** FAIL **");
        if (found_65530 || !found_2 || !found_0 || !found_1) return 1;
    }

    /* ---- G4: MISMATCH — n_gaps != n_spikes-1 -> counted, slot freed, no 0x05 ---- */
    {
        reset_all();
        for (int i = 0; i < V[0].n06; i++) feed2(IN_FE, V[0].f06[i]);
        feed_flush();   /* vec 0's marks half */
        /* forge a 0x03 with the SAME vid but a deliberately wrong n_gaps */
        memset(&g_h, 0, sizeof g_h);
        g_h.vector_id = V[0].vid; g_h.kind = ROUTER_HALF_BND;
        g_h.n_gaps = (uint16_t)(V[0].n_gaps + 1);            /* != n_spikes - 1 */
        int fl = router_ingest(&g_R, &g_h, g_out);
        printf("G5 mismatch (n_gaps=%d vs n_spikes-1=%d): ingest=%d (want -1), mismatches=%d "
               "(want 1), occupancy=%d (want 0 = slot freed), emitted=%d B (want 0) -> %s\n",
               V[0].n_gaps + 1, V[0].n_spikes - 1, fl, g_R.mismatches, rtr_occupancy(&g_R),
               fl > 0 ? fl : 0,
               (fl == -1 && g_R.mismatches == 1 && rtr_occupancy(&g_R) == 0) ? "OK" : "** FAIL **");
        if (!(fl == -1 && g_R.mismatches == 1 && rtr_occupancy(&g_R) == 0)) return 1;
    }

    /* ---- G5: ROBUSTNESS — corrupted frame CRC-rejected + fa_feed resyncs; over-cap LENGTH drops -- */
    {
        reset_all();
        unsigned char bad06[8192];
        memcpy(bad06, V[0].f06, (size_t)V[0].n06);
        bad06[10] ^= 0x01;                                   /* flip a payload bit */
        for (int i = 0; i < V[0].n06; i++) feed2(IN_FE, bad06[i]);
        feed_flush();
        int occ_after_corrupt = rtr_occupancy(&g_R);         /* nothing decoded -> no slot */
        drive_interleaved(&V[0], 0);                         /* a clean pair must still work */
        int resync_ok = (g_cap_count == 1 && g_cap_n == V[0].n05 &&
                         memcmp(g_cap, V[0].f05, (size_t)V[0].n05) == 0);
        /* over-cap LENGTH on the gap9 line: must drop + resync, never write past fa.pl */
        feed2(IN_GAP9, 0xA5); feed2(IN_GAP9, 0x5A); feed2(IN_GAP9, 0x03);
        feed2(IN_GAP9, 0xFF); feed2(IN_GAP9, 0xFF);          /* LENGTH 65535 > ROUTER_MAX_PL */
        for (int i = 0; i < 200; i++) feed2(IN_GAP9, 0x00);
        feed_flush();
        printf("G6 robustness: corrupted 0x06 -> occupancy %d (want 0 = CRC-rejected); next clean "
               "pair -> %s; over-cap LENGTH 65535 -> dropped, no smash -> %s\n",
               occ_after_corrupt, resync_ok ? "PAIRED byte-exact (resynced)" : "** FAIL **",
               "OK (ASan clean)");
        if (occ_after_corrupt != 0 || !resync_ok) return 1;
    }

    printf("ALL HOST GATES PASS (%s)\n", argv[1]);
    return 0;
}
