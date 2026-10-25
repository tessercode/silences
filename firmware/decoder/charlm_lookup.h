/*
 * charlm_lookup.h — A4.1 char-LM table lookup API (decode board), Q16.16 lineage.
 *
 * A4.1 CONTRACT (supersedes the A1.1 int8 contract; a1_golden_reference.md 9): this module
 * returns the Q16.16 ADDEND directly — the int32 mixture log-prob in nats * 65536 from the A4.0b
 * golden table (charlm_mix_q16.h). There is NOTHING to dequantize and NOTHING to scale: the table
 * entry IS the value. beta (0x00028000) is still NOT added here — it is the beam's per-letter-
 * close constant (beam.c), keeping this lookup a pure table map so the exhaustive gate stays a
 * strict value-equality check.
 *
 * UNCHANGED from A1.1 (do not touch): the index arithmetic ((37*c1 + c2)*37 + c3) and the
 * space-pad/BOS history rule (charlm.py:85-87). Only the table symbol and return type changed.
 *
 * Pure C (stdint only, no pico-sdk includes) so the host self-check compiles the SHIPPING file
 * with zero mocks.
 */
#ifndef CHARLM_LOOKUP_H
#define CHARLM_LOOKUP_H
#include <stdint.h>

/* Alphabet (pinned): index 0 = ' ', 1..10 = '0'..'9', 11..36 = 'A'..'Z'.
 * Any other byte maps to 0 (space) — the golden fallback (charlm.py:86-87).
 * uint8_t parameters kept from the A1.2 type fix (x86-signed vs ARM-unsigned char). */
int charlm_char_index(uint8_t c);

/* Raw table lookup by alphabet indices: CHARLM_LOGP_Q16[(37*c1 + c2)*37 + c3].
 * Returns the Q16.16 log-prob (nats * 65536). Callers pass 0..36. */
int32_t charlm_lookup_idx(int c1, int c2, int c3);

/* The research-rule wrapper (charlm.py:85-87, replicated EXACTLY): context = the last 2 bytes of
 * `history` (length hist_len), LEFT-PADDED WITH SPACE when shorter (BOS == space); `next` and both
 * context bytes map through charlm_char_index (non-alphabet -> space). Returns the Q16.16 addend
 * (beta NOT included). */
int32_t charlm_letter_logprob_q16(uint8_t next, const uint8_t *history, int hist_len);

#endif /* CHARLM_LOOKUP_H */
