/*
 * beam.h — A4.1 integer sentence beam (decode board). Port of the FROZEN A4.0b reference
 * a4_beam_golden/beam_int.py (itself a statement-for-statement mirror of
 * analysis/sentence_beam.py:135-176). Pure C, stdint only — the host self-check compiles the
 * SHIPPING file in-TU with zero mocks (A1.1/A2.1 pattern).
 *
 * Scoring is Q16.16 nats in int64 (i32 would suffice — measured max |score| 9,192,283 = 233.6x
 * inside i32 — i64 is the golden's stated free insurance). LM addend = the A4.0b Q16.16 table
 * value + BEAM_BETA_Q at every letter close INCLUDING the final; the oracle/uniform mode scores
 * 0 (no beta), mirroring beam_int.lm_uniform_q16.
 *
 * Tie-break is the PINNED rule (a4_golden_reference.md 2, NORMATIVE):
 *   RULE 1  stable descending sort of the <=60 candidates, first `width` survive; equal scores
 *           keep generation order (parent beam rank, then extend -> close-letter -> close-word).
 *   RULE 2  final winner = FIRST maximum in final-list order (strict > when replacing).
 */
#ifndef A4_BEAM_H
#define A4_BEAM_H
#include <stdint.h>

#define BEAM_WIDTH     20
#define BEAM_EXPAND    (3 * BEAM_WIDTH)          /* <=3 children per hypothesis, never alphabet */
#define BEAM_TEXT_MAX  512                       /* golden bound; max reached in goldens = 119 */
#define BEAM_MAX_GAPS  255                       /* N_MARKS_MAX 256 */
#define BEAM_SEG_BYTES ((BEAM_MAX_GAPS + 3) / 4) /* 2 bits per gap decision = 64 B */
#define BEAM_BETA_Q    0x00028000                /* 2.5 in Q16.16, exact */

#define BEAM_LM_DEPLOY  0                        /* Q16.16 table + beta (beam_int.lm_deploy_q16) */
#define BEAM_LM_UNIFORM 1                        /* 0, no beta (beam_int.lm_uniform_q16) */

typedef struct {
    char    text[BEAM_TEXT_MAX];  /* decoded text, trailing space stripped (beam_int :175) */
    int32_t text_len;
    uint8_t seg[BEAM_SEG_BYTES];  /* per-gap decision, 2 bits each: 0 extend / 1 CL / 2 CW */
    int32_t n_gaps;
    int64_t score_q16;            /* final winner's score (incl. final LM+beta) */
    int32_t died;                 /* 1 = beam death or empty final -> no other field valid */
} beam_result_t;

/* Decode one transmission. ch[] = per-mark channel bits (0 = '.', 1 = '-'), n_marks >= 1;
 * lp3[] = 3 x int32 Q16.16 log-posteriors per gap, gap-major (n_marks-1 gaps: the A2/A3 device
 * output / 0x05 payload order). width <= BEAM_WIDTH (117-vector gate uses BEAM_WIDTH; smaller
 * widths exist for the tie-break unit vectors). Returns 0, or -1 with r->died = 1. */
int beam_decode(const uint8_t *ch, int n_marks, const int32_t *lp3, int width, int lm_mode,
                beam_result_t *r);

/* seg accessor (test/emit convenience): decision for gap i in {0,1,2}. */
static inline int beam_seg_get(const beam_result_t *r, int i) {
    return (r->seg[i >> 2] >> ((i & 3) << 1)) & 3;
}

#endif /* A4_BEAM_H */
