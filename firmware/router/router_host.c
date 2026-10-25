/*
 * router_host.c — native-gcc host driver for the router CORE (Phase A). Exercises the SAME
 * router_core.h that the firmware runs, fed a REPLAYED byte stream reconstructed from the real
 * encoder 0x04 and gap9 0x03 frames (see gen in run_phase_a.py). No pico deps, no hardware.
 *
 * Input file = a sequence of (input_idx : u8, data_byte : u8) pairs — the exact byte-arrival
 * interleaving across the two producer lines (input 0 = encoder line, 1 = gap9 line). This mirrors
 * what the two PIO UART receivers will hand the core on-device, so the byte assembler + pairing are
 * tested identically. Emits each assembled 0x05 forward frame as one hex line on stdout; router stats
 * + any alignment mismatch go to stderr.
 */
#include <stdio.h>
#include <stdint.h>
#include "router_core.h"

#define NIN 2    /* input 0 = encoder (0x04) line, input 1 = gap9 (0x03) line */

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: router_host events.bin\n"); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("open"); return 2; }

    fa_t in[NIN];
    for (int i = 0; i < NIN; i++) fa_reset(&in[i]);
    router_t R; router_init(&R);

    uint8_t mid, pl[ROUTER_MAX_PL], len, out[320];
    half_t h = {0};
    unsigned char pair[2];
    while (fread(pair, 1, 2, f) == 2) {
        int ii = pair[0];
        if (ii < 0 || ii >= NIN) continue;
        if (fa_feed(&in[ii], pair[1], &mid, pl, &len)) {          /* complete CRC-valid frame */
            int k = decode_half(mid, pl, len, &h);
            if (k) {
                int fl = router_ingest(&R, &h, out);
                if (fl > 0) { for (int i = 0; i < fl; i++) printf("%02x", out[i]); printf("\n"); }
                else if (fl < 0)
                    fprintf(stderr, "ALIGN MISMATCH vid=%u n_spikes=%u n_gaps=%u\n",
                            h.vector_id, h.n_spikes, h.n_gaps);
            }
        }
    }
    fclose(f);
    fprintf(stderr, "router stats: paired=%d mismatches=%d dropped_full=%d table_hwm=%d\n",
            R.paired, R.mismatches, R.dropped_full, R.hwm);
    return 0;
}
