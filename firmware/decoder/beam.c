/*
 * beam.c — A4.1 integer sentence beam. Statement-for-statement port of the FROZEN
 * a4_beam_golden/beam_int.py (line refs "bi:" below; beam_int itself carries the
 * sentence_beam.py:135-176 refs). Every divergence surface from the A4.0 inventory is
 * reproduced, none "improved":
 *   - additive scoring only; ONE LM evaluation per (hypothesis, gap), consumed by BOTH closes
 *   - <=3 children per hypothesis (extend / close-letter / close-word); alphabet never enumerated
 *   - NO dedup of identical (emitted, current) states — duplicates coexist and consume width
 *   - stable descending sort, first `width` survive (PINNED RULE 1); strict-> final max (RULE 2)
 *   - beam death -> died; final complete-pattern filter; .strip() on the winning text
 *   - beta added at EVERY letter close INCLUDING the final; uniform mode adds 0 (no beta)
 *
 * The hypothesis store is .bss (g_beam/g_next below, ~47 KB) — NEVER stack (B1.1 lesson).
 * Naive per-child struct copy is deliberate: match the reference, do not optimize.
 */
#include "beam.h"
#include "charlm_lookup.h"
#include "morse_tables.h"
#include <assert.h>

_Static_assert(MORSE_MAX_LEN == 5, "beam's 2-B pattern encoding relies on max pattern length 5");
_Static_assert(sizeof(MORSE_CHAR) == 64, "MORSE_CHAR must cover codes 0..63");
_Static_assert(BEAM_SEG_BYTES == 64, "seg packing drifted");

typedef struct {
    int64_t  score;                 /* Q16.16 nats (bi: python int; device i64, golden's choice) */
    int32_t  text_len;
    uint8_t  cur_bits, cur_len;     /* open pattern, mark i at bit i ('.'=0,'-'=1); code=(1<<len)|bits */
    uint8_t  seg[BEAM_SEG_BYTES];   /* 2-bit decisions; positions >= current gap are 0 by carry */
    char     text[BEAM_TEXT_MAX];   /* emitted letters + spaces (bi: `emitted`) */
} hyp_t;

/* .bss, double-buffered: 20 + 60 hypotheses (~592 B each ~= 47 KB). */
static hyp_t g_beam[BEAM_WIDTH];
static hyp_t g_next[BEAM_EXPAND];

static inline int morse_code(const hyp_t *h) { return (1 << h->cur_len) | h->cur_bits; }

static inline int64_t lm_addend(int lm_mode, uint8_t ch, const hyp_t *h) {
    if (lm_mode == BEAM_LM_UNIFORM)                 /* bi: lm_uniform_q16 — 0, no beta */
        return 0;
    return (int64_t)charlm_letter_logprob_q16(ch, (const uint8_t *)h->text, h->text_len)
           + BEAM_BETA_Q;                            /* bi: lm_deploy_q16 — table + beta */
}

static inline void seg_set(hyp_t *h, int gap, int d) {
    h->seg[gap >> 2] |= (uint8_t)(d << ((gap & 3) << 1));   /* position is 0 before (carried) */
}

/* PINNED RULE 1 (a4_golden_reference.md 2): stable descending insertion sort of candidate
 * indices. Precedent: sort_i64 (firmware/fe/frontend.c:21-27) — insertion, strict compare =>
 * stable; this is its record/index variant. Shifting only past STRICTLY-smaller scores means
 * equal-score candidates keep insertion (= generation) order. */
static void sort_idx_desc_stable(uint8_t *idx, int n, const hyp_t *pool) {
    for (int i = 1; i < n; i++) {
        uint8_t key = idx[i];
        int64_t ks = pool[key].score;
        int j = i - 1;
        while (j >= 0 && pool[idx[j]].score < ks) {  /* strict <: equals are NOT passed */
            idx[j + 1] = idx[j];
            j--;
        }
        idx[j + 1] = key;
    }
}

#ifdef BEAM_TESTING
void beam_sort_test_hook(uint8_t *idx, int n, const int64_t *scores) {
    static hyp_t pool[BEAM_EXPAND];                  /* .bss, test-only */
    for (int i = 0; i < n; i++) pool[i].score = scores[i];
    sort_idx_desc_stable(idx, n, pool);
}
#endif

int beam_decode(const uint8_t *ch, int n_marks, const int32_t *lp3, int width, int lm_mode,
                beam_result_t *r) {
    r->died = 1;
    r->text_len = 0;
    r->n_gaps = n_marks - 1;
    if (n_marks <= 0 || width <= 0 || width > BEAM_WIDTH)    /* bi:94-96 (N == 0 -> None) */
        return -1;

    /* bi:97 — beam = [("", spike_channels[0], 0, ())] */
    hyp_t *root = &g_beam[0];
    root->score = 0;
    root->text_len = 0;
    root->cur_bits = (uint8_t)(ch[0] & 1);
    root->cur_len = 1;
    for (int k = 0; k < BEAM_SEG_BYTES; k++) root->seg[k] = 0;
    int n_beam = 1;

    for (int i = 0; i < n_marks - 1; i++) {                  /* bi:98 */
        const int64_t lp_in = lp3[3 * i], lp_il = lp3[3 * i + 1], lp_iw = lp3[3 * i + 2]; /* bi:99 */
        const uint8_t c_next = (uint8_t)(ch[i + 1] & 1);     /* bi:100 */
        int n_next = 0;
        for (int b = 0; b < n_beam; b++) {                   /* bi:101 — beam order preserved */
            const hyp_t *h = &g_beam[b];
            /* EXTEND (bi:102-103): only while a longer prefix can exist; validity via the mask */
            if (h->cur_len < MORSE_MAX_LEN) {
                const uint8_t nb = (uint8_t)(h->cur_bits | (c_next << h->cur_len));
                const int ncode = (1 << (h->cur_len + 1)) | nb;
                if ((MORSE_VALID >> ncode) & 1u) {
                    hyp_t *c = &g_next[n_next++];
                    *c = *h;
                    c->cur_bits = nb;
                    c->cur_len++;
                    c->score += lp_in;
                    seg_set(c, i, 0);
                }
            }
            /* CLOSE (bi:104-108): complete pattern only; ONE LM eval feeds BOTH branches */
            const uint8_t letter = MORSE_CHAR[morse_code(h)];
            if (letter) {
                const int64_t lm = lm_addend(lm_mode, letter, h);   /* bi:106 */
                hyp_t *cl = &g_next[n_next++];                       /* bi:107 close-letter */
                *cl = *h;
                assert(cl->text_len + 1 <= BEAM_TEXT_MAX);
                cl->text[cl->text_len++] = (char)letter;
                cl->cur_bits = c_next;
                cl->cur_len = 1;
                cl->score += lp_il + lm;
                seg_set(cl, i, 1);
                hyp_t *cw = &g_next[n_next++];                       /* bi:108 close-word */
                *cw = *h;
                assert(cw->text_len + 2 <= BEAM_TEXT_MAX);
                cw->text[cw->text_len++] = (char)letter;
                cw->text[cw->text_len++] = ' ';
                cw->cur_bits = c_next;
                cw->cur_len = 1;
                cw->score += lp_iw + lm;
                seg_set(cw, i, 2);
            }
        }
        /* bi:109-113 — RULE 1 prune: stable sort, first `width` survive */
        uint8_t idx[BEAM_EXPAND];
        for (int k = 0; k < n_next; k++) idx[k] = (uint8_t)k;
        sort_idx_desc_stable(idx, n_next, g_next);
        n_beam = n_next < width ? n_next : width;
        for (int k = 0; k < n_beam; k++) g_beam[k] = g_next[idx[k]];
        if (n_beam == 0)                                     /* bi:114-115 beam death */
            return -1;
    }

    /* bi:116-121 — final: complete-pattern filter, final LM+beta, RULE 2 strict-> first max */
    int best = -1;
    int64_t best_score = 0;
    uint8_t best_letter = 0;
    for (int b = 0; b < n_beam; b++) {
        const uint8_t letter = MORSE_CHAR[morse_code(&g_beam[b])];
        if (!letter)                                         /* incomplete: silently dropped */
            continue;
        const int64_t s = g_beam[b].score + lm_addend(lm_mode, letter, &g_beam[b]);
        if (best < 0 || s > best_score) {                    /* STRICT >: first max wins ties */
            best = b;
            best_score = s;
            best_letter = letter;
        }
    }
    if (best < 0)                                            /* bi:119-120 empty final */
        return -1;

    const hyp_t *w = &g_beam[best];
    int len = w->text_len;
    for (int k = 0; k < len; k++) r->text[k] = w->text[k];
    r->text[len++] = (char)best_letter;                      /* bi:116 emitted + ch */
    /* bi:124 .strip(): trailing (close-word residue) and leading (unreachable, kept faithful) */
    while (len > 0 && r->text[len - 1] == ' ') len--;
    int lead = 0;
    while (lead < len && r->text[lead] == ' ') lead++;
    if (lead) {
        for (int k = lead; k < len; k++) r->text[k - lead] = r->text[k];
        len -= lead;
    }
    r->text_len = len;
    if (len < BEAM_TEXT_MAX) r->text[len] = '\0';
    for (int k = 0; k < BEAM_SEG_BYTES; k++) r->seg[k] = w->seg[k];
    r->score_q16 = best_score;
    r->died = 0;
    return 0;
}
