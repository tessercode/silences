/*
 * charlm_lookup.c — A4.1 char-LM lookup implementation (decode board), Q16.16 lineage.
 *
 * THE SINGLE TRANSLATION UNIT that includes the A4.0b golden header (A1 golden 6 inclusion rule
 * carries over: `static const` in a header gives every including TU a private ~198 KB copy; this
 * file is the one and only include site, so the table exists exactly once, in .rodata/XIP flash).
 * The header is embedded VERBATIM (sha256 d920d9ca... == manifest_q16.json's; copied by
 * gen_a4_port_artifacts.py, which refuses on mismatch).
 *
 * vs A1.1: table symbol CHARLM_LOGP_Q -> CHARLM_LOGP_Q16, return type int8_t -> int32_t. The
 * index arithmetic and the history rule are byte-for-byte the A1.1 logic (silicon-validated
 * exhaustively at A1.2); only the artifact changed — hence the exhaustive gate re-runs (Task 1).
 */
#include "charlm_lookup.h"
#include "charlm_mix_q16.h"   /* the ONLY include of the golden table header */

/* assert-at-build (the project's pattern): tie this port to the golden's pinned shape. */
_Static_assert(CHARLM_Q16_V == 37, "alphabet drifted from the A4.0b golden");
_Static_assert(CHARLM_Q16_ENTRIES == 50653, "table entry count drifted from the A4.0b golden");
_Static_assert(sizeof(CHARLM_LOGP_Q16) == 50653 * 4, "embedded table size != 37^3 * 4 bytes");

int charlm_char_index(uint8_t c) {
    if (c == ' ') return 0;
    if (c >= '0' && c <= '9') return 1 + (c - '0');
    if (c >= 'A' && c <= 'Z') return 11 + (c - 'A');
    return 0;                          /* golden rule: non-alphabet -> space (charlm.py:86-87) */
}

int32_t charlm_lookup_idx(int c1, int c2, int c3) {
    return CHARLM_LOGP_Q16[(CHARLM_Q16_V * c1 + c2) * CHARLM_Q16_V + c3];
}

int32_t charlm_letter_logprob_q16(uint8_t next, const uint8_t *history, int hist_len) {
    /* context = last 2 chars, space-padded on the left (BOS == space): charlm.py:85 — UNCHANGED */
    int c1 = 0, c2 = 0;
    if (hist_len >= 2) {
        c1 = charlm_char_index(history[hist_len - 2]);
        c2 = charlm_char_index(history[hist_len - 1]);
    } else if (hist_len == 1) {
        c2 = charlm_char_index(history[0]);   /* c1 stays space */
    }                                          /* hist_len == 0: (space, space) = index-0 row */
    return charlm_lookup_idx(c1, c2, charlm_char_index(next));
}
