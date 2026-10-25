/*
 * router_core.h — portable interconnect-router CORE (increment 1). Pure integer, no pico/host deps,
 * so the SAME code runs in the host test (router_host.c) and the RP2040 firmware (router_hw.c).
 *
 * Responsibilities (transport-agnostic — bytes in, forward frame out):
 *   1. per-input RESULT_FRAME byte assembler (sync 0xA5 0x5A -> module -> len -> payload -> CRC-8),
 *   2. decode the two producer payloads into a "half" (encoder 0x04 -> marks; gap9 0x03 -> boundaries),
 *   3. pair the two halves by vector_id (buffering the earlier-arriving marks across the ~76ms skew),
 *   4. validate the alignment invariant (n_gaps == n_spikes-1) and build the 0x05 forward frame.
 *
 * N-INPUT EXTENSIBLE: inputs are an array of frame assemblers (fa_t); increment 1 uses two (encoder,
 * gap9). A third input (front-end) is one more array entry + a decode_half case — no path is hardcoded.
 * Producer frames are consumed UNCHANGED; nothing here modifies producer firmware.
 */
#ifndef ROUTER_CORE_H
#define ROUTER_CORE_H
#include <stdint.h>

/* ---- sizes (transmission scale; N_MARKS_MAX = 256 marks / 255 gaps) ---- */
/* B2b.1 WIDENING: ROUTER_MAX_PL 255 -> 3321 = the largest payload any sentence-tree consumer can
 * receive (the widened 0x05: 5 + 256 + 12*255; b2_frame_golden.md 1.3). ONE shared constant for all
 * four consumers, per decision. On gap9's line the largest is the 0x06 (1289), but the constant is
 * shared, so fa_t.pl is sized to the max.
 *
 * B2b.2: this header is now SHARED (firmware/common/), included by gap9's wrapper and the router —
 * and later the decode board. B2b.1's marker here warned that the pairing half was still B3's and
 * still carried the `3 +`-for-a-4-byte-header overflow; **B2b.2 RETIRES that marker**: the pairing
 * half below is widened to transmission scale and the `3 +` defect is FIXED (see
 * rtr_build_forward). Both gap9 and the router are now raised, so the chain is flashable. */
#define ROUTER_MAXSP    256      /* B2b.2: max marks per TRANSMISSION (was 64 per word) */
#define ROUTER_MAXGAP   (ROUTER_MAXSP - 1)   /* 255 — derived, never a literal */
#define ROUTER_NSLOTS   8        /* pairing table depth; with eviction it cannot fatally fill (B3) */
#define ROUTER_NCLASS   3        /* D4: 3 log-posteriors per gap, argmax-free */

/* Payload/buffer bounds, DERIVED from the constants above — never literals. Six buffers across two
 * files have now broken because they were sized to the shipped bound rather than to the cap
 * (b2_frame_golden.md 2.2/2.3, b2b_1_port.md 5). Values at the cap: 3321 / 3324 / 3327. */
#define ROUTER_PL_0x05_MAX  (5 + ROUTER_MAXSP + 4 * ROUTER_NCLASS * ROUTER_MAXGAP)   /* 3321 */
#define ROUTER_TMP_0x05_MAX (3 + ROUTER_PL_0x05_MAX)   /* 3324 — `3 +`: mid + BOTH length bytes */
#define ROUTER_OUT_0x05_MAX (6 + ROUTER_PL_0x05_MAX)   /* 3327 — on-wire (marker+mid+len2+crc) */
#define ROUTER_MAX_PL   ROUTER_PL_0x05_MAX             /* max RESULT_FRAME payload (was 255) */

/* ---- module ids (see common/RESULT_FRAME.md) ---- */
#define ROUTER_MOD_GAP9     0x03u
#define ROUTER_MOD_ENCODER  0x04u   /* LEGACY: encoder no longer emits a frame (completion-spike GPIO now) */
#define ROUTER_MOD_FORWARD  0x05u   /* THIS router's output frame (next free id) */
#define ROUTER_MOD_FE_FWD   0x06u   /* front-end marks+T̂u forward -> the LIVE marks source (was encoder-0x04) */
#define ROUTER_KIND_RESULT  0x01u
#define ROUTER_KIND_EOP     0x02u
#define ROUTER_START0       0xA5u
#define ROUTER_START1       0x5Au

/* ======================= CRC-8/SMBus (identical to every module) ======================= */
static inline uint8_t rtr_crc8(const uint8_t *d, int n) {
    uint8_t c = 0;
    for (int i = 0; i < n; i++) { c ^= d[i];
        for (int b = 0; b < 8; b++) c = (c & 0x80) ? (uint8_t)((c << 1) ^ 0x07) : (uint8_t)(c << 1); }
    return c;
}
static inline uint16_t rtr_u16(const uint8_t *p) { return (uint16_t)((p[0] << 8) | p[1]); }
static inline int32_t  rtr_i32(const uint8_t *p) {
    return (int32_t)(((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3]);
}
static inline int rtr_put_u16(uint8_t *p, uint16_t v) { p[0] = (uint8_t)(v >> 8); p[1] = (uint8_t)v; return 2; }
static inline int rtr_put_i32(uint8_t *p, int32_t v) { uint32_t u = (uint32_t)v;
    p[0] = (uint8_t)(u >> 24); p[1] = (uint8_t)(u >> 16); p[2] = (uint8_t)(u >> 8); p[3] = (uint8_t)u; return 4; }

/* ======================= per-input frame assembler (byte state machine) ======================= */
/* B2b.1 WIDENING: LENGTH u8 -> u16 BE. The state machine gains ONE state (the second length byte),
 * `len`/`idx` become u16, and the CRC staging spans THREE header bytes (mid + len_hi + len_lo). */
typedef struct {
    int      st;                      /* 0:A5 1:5A 2:module 3:len_hi 4:len_lo 5:payload 6:crc */
    uint8_t  mid;
    uint16_t len, idx;                /* WIDENED (were u8): a payload is up to 3321 B */
    uint8_t  pl[ROUTER_MAX_PL];
} fa_t;

static inline void fa_reset(fa_t *f) { f->st = 0; }

/* Feed one byte. Returns 1 when a CRC-valid frame completes (out_mid/out_pl/out_len filled); else 0.
 * Any structural/CRC error silently resyncs (drops the frame, hunts the next 0xA5 0x5A). */
static inline int fa_feed(fa_t *f, uint8_t b, uint8_t *out_mid, uint8_t *out_pl, uint16_t *out_len) {
    switch (f->st) {
    case 0: f->st = (b == ROUTER_START0) ? 1 : 0; return 0;
    case 1: f->st = (b == ROUTER_START1) ? 2 : (b == ROUTER_START0 ? 1 : 0); return 0;
    case 2: f->mid = b; f->st = 3; return 0;
    case 3: f->len = (uint16_t)((uint16_t)b << 8); f->st = 4; return 0;    /* WIDENED: LENGTH hi */
    case 4:
        f->len = (uint16_t)(f->len | b);                                   /* WIDENED: LENGTH lo */
        f->idx = 0;
        if (f->len > ROUTER_MAX_PL) { f->st = 0; return 0; }  /* over-cap LENGTH: drop + resync */
        f->st = (f->len == 0) ? 6 : 5;
        return 0;
    case 5:
        f->pl[f->idx++] = b;
        if (f->idx >= f->len) f->st = 6;
        return 0;
    case 6: {
        /* B1.1's lesson: this staging buffer is STATIC, not a stack local. At the widened size it
         * is 3324 B — a 2048 B default stack would be smashed by the declaration alone. It is
         * `3 +`, NOT `2 +`: the widened CRC scope carries TWO length bytes (a one-length-byte CRC
         * mismatches every frame — the B1.1 trap). */
        static uint8_t tmp[3 + ROUTER_MAX_PL];
        tmp[0] = f->mid;
        tmp[1] = (uint8_t)(f->len >> 8);
        tmp[2] = (uint8_t)f->len;
        for (uint16_t i = 0; i < f->len; i++) tmp[3 + i] = f->pl[i];
        uint8_t c = rtr_crc8(tmp, 3 + (int)f->len);  /* CRC-8/SMBus over module_id, LENGTH(2), payload */
        f->st = 0;
        if (c == b) {
            *out_mid = f->mid; *out_len = f->len;
            for (uint16_t i = 0; i < f->len; i++) out_pl[i] = f->pl[i];
            return 1;
        }
        return 0;                     /* bad CRC -> dropped, resync */
    }
    default: f->st = 0; return 0;
    }
}

/* ======================= decode a validated frame into a "half" ======================= */
enum { ROUTER_HALF_NONE = 0, ROUTER_HALF_MARKS = 1, ROUTER_HALF_BND = 2 };

/* B2b.2: counts u8 -> u16 (256 does not fit a u8) and q16 carries ROUTER_NCLASS values per gap
 * (D4). 332 B -> 3,336 B. ⚠️ half_t is far too big to be a stack local — see router_hw.c. */
typedef struct {
    uint16_t vector_id;
    uint8_t  kind;                    /* ROUTER_HALF_MARKS | ROUTER_HALF_BND */
    uint16_t n_spikes; uint8_t ch[ROUTER_MAXSP];                        /* WIDENED (was u8/64) */
    uint16_t n_gaps;   int32_t q16[ROUTER_NCLASS * ROUTER_MAXGAP];      /* WIDENED (was u8/1x63) */
} half_t;

/* Returns half kind (MARKS/BND) if a usable KIND-0x01 record was decoded; ROUTER_HALF_NONE for the
 * EOP sentinel or any unrecognised module/kind. */
static inline int decode_half(uint8_t module_id, const uint8_t *pl, int len, half_t *h) {
    if (len < 1 || pl[0] != ROUTER_KIND_RESULT) return ROUTER_HALF_NONE;   /* EOP / non-result */
    if (module_id == ROUTER_MOD_FE_FWD) {
        /* front-end 0x06 = the LIVE marks source. Layout (the B1-FROZEN writer,
         * fe/frontend_stage3.c:234-241; b2_frame_golden.md 3):
         *   [0x01][seq u16][tu_shortest_isi i32][n_marks u16][(end_tick i32, channel u8) x n_marks]
         * o:    1 ---------- 3 ------------------ 7 --------- 9 -> mark i channel at pl[9 + i*5 + 4]
         * Extract vector_id=seq, n_spikes=n_marks, ch[]; SKIP tu (pl[3..6]) and the per-mark end_tick
         * i32 (the router needs only channels + count — A4 confirmed the beam consumes no timing).
         *
         * ⚠️ B2b.2 WIDENING: n_marks u8 -> u16, so the header is 9 B, not 8, and every downstream
         * offset moves. This is NOT cosmetic: the shipped `uint8_t n = pl[7]` read the u16's HIGH
         * byte against B1's emitter — every sub-256-mark transmission yields n = 0 and the mark
         * records start one byte late (B2b.1 found the identical bug in gap9's parse). */
        if (len < 9) return ROUTER_HALF_NONE;                        /* KIND+seq+tu+n_marks header */
        int o = 1;
        h->vector_id = rtr_u16(pl + o); o += 2;   /* seq -> pairing id (pl[1..2]) */
        o += 4;                                   /* tu_shortest_isi:i32 -> SKIP (pl[3..6]) */
        uint16_t n = rtr_u16(pl + o); o += 2;     /* WIDENED: n_marks u16 BE (pl[7..8]); o now = 9 */
        if (n > ROUTER_MAXSP || len < 9 + (int)n * 5) return ROUTER_HALF_NONE;
        for (int i = 0; i < n; i++) h->ch[i] = pl[o + i * 5 + 4];   /* channel at pl[9 + i*5 + 4] */
        h->n_spikes = n; h->kind = ROUTER_HALF_MARKS;
        return ROUTER_HALF_MARKS;
    }
    /* B2b.2: the LEGACY ROUTER_MOD_ENCODER (0x04) branch is REMOVED. The encoder has emitted
     * completion-spike GPIO rather than a UART frame since the FE took over marks; no producer sends
     * 0x04 on any path, the sentence chain has no 0x04 at all, and the branch's own u8 layout would
     * have needed widening to stay compilable against the new half_t. Dead code carrying a latent
     * width bug is worse than no code. (ROUTER_MOD_ENCODER stays defined: it is a wire id.) */
    if (module_id == ROUTER_MOD_GAP9) {
        /* [0x01][vid u16][n_gaps u16][3 x i32 BE per gap]  — the B2b.1 writer / B2.0 golden.
         * B2b.2 WIDENING: n_gaps u8 -> u16; the decision_bits skip is DELETED (D4 argmax-free);
         * ROUTER_NCLASS values per gap instead of 1. payload = 5 + 12*ng. */
        if (len < 5) return ROUTER_HALF_NONE;                        /* KIND+vid+n_gaps header */
        int o = 1;
        h->vector_id = rtr_u16(pl + o); o += 2;
        uint16_t ng = rtr_u16(pl + o); o += 2;    /* WIDENED: n_gaps u16 BE */
                                                  /* decision_bits skip -> DELETED (D4) */
        if (ng > ROUTER_MAXGAP || len < 5 + (int)ng * 4 * ROUTER_NCLASS) return ROUTER_HALF_NONE;
        for (int t = 0; t < ng * ROUTER_NCLASS; t++) h->q16[t] = rtr_i32(pl + o + 4 * t);
        h->n_gaps = ng; h->kind = ROUTER_HALF_BND;
        return ROUTER_HALF_BND;
    }
    return ROUTER_HALF_NONE;
}

/* ======================= pair-by-vector_id table + forward build ======================= */
/* B2b.2: counts u8 -> u16; q16 carries ROUTER_NCLASS per gap (D4). 332 B -> 3,336 B per slot;
 * x ROUTER_NSLOTS = 26,688 B of .bss. */
typedef struct {
    int      used, have_marks, have_bnd;
    uint16_t vector_id;
    uint16_t n_spikes; uint8_t ch[ROUTER_MAXSP];                        /* WIDENED */
    uint16_t n_gaps;   int32_t q16[ROUTER_NCLASS * ROUTER_MAXGAP];      /* WIDENED */
} pair_slot_t;

typedef struct {
    pair_slot_t slots[ROUTER_NSLOTS];
    /* stats. `dropped_full` is retained ONLY as a provable-zero invariant: with B3's eviction the
     * table cannot fatally fill, so nothing increments it any more (see router_ingest). */
    int paired, mismatches, dropped_full, hwm, evictions;
} router_t;

static inline void router_init(router_t *R) {
    for (int i = 0; i < ROUTER_NSLOTS; i++) R->slots[i].used = 0;
    R->paired = R->mismatches = R->dropped_full = R->hwm = R->evictions = 0;
}

static inline int rtr_occupancy(const router_t *R) {
    int n = 0; for (int i = 0; i < ROUTER_NSLOTS; i++) n += R->slots[i].used; return n;
}

/* Build the 0x05 forward frame into out. Returns total frame length.
 *
 * Mirrors b2_frame_golden/frames_b2.py:build_0x05_payload (the FROZEN golden defines these bytes):
 *   [KIND 0x01][vector_id u16 BE][n_spikes u16 BE][ch u8 x n][3 x i32 BE per gap]
 *   payload = 5 + n + 12*(n-1);  on-wire = payload + 6
 *
 * ⚠️ B2b.2 FIXES THE `3 +` DEFECT. The shipped declaration was
 *     uint8_t pl[3 + ROUTER_MAXSP + 4*ROUTER_MAXGAP]   = 319
 * while the header it writes is KIND(1) + vid(2) + n_spikes(1) = FOUR bytes, so o_max was
 *     4 + 64 + 4*63 = 320  -> pl overran by 1, tmp[2+sizeof pl] by 1, and the caller's out[320]
 * by 5 (b2_frame_golden.md 2.2). All three are now DERIVED from the constants (ROUTER_PL/TMP/OUT_
 * 0x05_MAX), so the header size is expressed once and cannot drift again.
 * ⚠️ tmp is `3 +`, not `2 +`: the widened CRC scope carries BOTH length bytes. */
_Static_assert(ROUTER_PL_0x05_MAX == 3321 && ROUTER_TMP_0x05_MAX == 3324
               && ROUTER_OUT_0x05_MAX == 3327, "B2.0 golden buffer sizes");

static inline int rtr_build_forward(uint16_t vid, const uint8_t *ch, int n_spikes,
                                    const int32_t *q16, int n_gaps, uint8_t *out) {
    /* STATIC, not stack locals: at the cap these are 3,321 + 3,324 B and the router's default stack
     * is 2,048 B (B1.1's lesson; b2b_1_port.md 5 caught the same shape in gap9's writer). */
    static uint8_t pl[ROUTER_PL_0x05_MAX];
    static uint8_t tmp[ROUTER_TMP_0x05_MAX];
    int o = 0;
    pl[o++] = ROUTER_KIND_RESULT;                                  /* :172 */
    o += rtr_put_u16(pl + o, vid);                                 /* :173 */
    o += rtr_put_u16(pl + o, (uint16_t)n_spikes);                  /* :174 WIDENED (was u8) */
    for (int i = 0; i < n_spikes; i++) pl[o++] = ch[i];            /* :175 1 byte per channel */
    for (int t = 0; t < n_gaps * ROUTER_NCLASS; t++)               /* :176 WIDENED (was 1 x i32) */
        o += rtr_put_i32(pl + o, q16[t]);                          /* raw Q16.16, dequant downstream */
    int po = 0;
    out[po++] = ROUTER_START0; out[po++] = ROUTER_START1;
    out[po++] = ROUTER_MOD_FORWARD;
    out[po++] = (uint8_t)(o >> 8); out[po++] = (uint8_t)o;         /* WIDENED: u16 BE LENGTH */
    for (int i = 0; i < o; i++) out[po++] = pl[i];
    tmp[0] = ROUTER_MOD_FORWARD;
    tmp[1] = (uint8_t)(o >> 8); tmp[2] = (uint8_t)o;               /* CRC spans BOTH length bytes */
    for (int i = 0; i < o; i++) tmp[3 + i] = pl[i];
    out[po++] = rtr_crc8(tmp, 3 + o);
    return po;
}

/* ======================= 0x05 READER — the decode board's input (B2b.3) =======================
 *
 * WHY IT LIVES HERE, next to rtr_build_forward rather than in firmware/decode/: the 0x05 WRITER is
 * three functions up. Putting the reader beside it states the layout TWICE IN ONE FILE, adjacent,
 * in opposite directions — so a future widening that touches one and misses the other shows up in
 * the same diff hunk. That is not hypothetical: the `n_marks` u8->u16 widening broke the 0x06 reader
 * in BOTH of its consumers (gap9_hw.c and decode_half) precisely because the reader lived far from
 * the writer, and each read the u16's HIGH byte (b2b_1_port.md 2.3, b2b_2_port.md 3.1). This header
 * is already the one place that knows every frame's shape; the 0x05 was the only gap.
 * Cost to non-users: zero — `static inline` + a type declaration are dropped in gap9's and the
 * router's TUs, which instantiate no fwd_t (the B2b.1 dead-code precedent).
 *
 * MIRRORS: b2_frame_golden/frames_b2.py:parse_0x05_frame (the FROZEN golden's reference parser) and
 * the shape of output_stage/sources.py:154-175 (the only other 0x05 parser in the repo — word-scale
 * Python, u8 n_spikes, 1 x i32/gap; it validates NOTHING beyond the CRC, which this does not copy). */
typedef struct {
    uint16_t vector_id;                              /* the beam never reads it; the result line does */
    uint16_t n_spikes;
    uint16_t n_gaps;                                 /* == n_spikes - 1, by the layout */
    uint8_t  ch[ROUTER_MAXSP];                       /* -> beam_decode's `ch`   (0/1 per mark) */
    int32_t  lp3[ROUTER_NCLASS * ROUTER_MAXGAP];     /* -> beam_decode's `lp3`  (gap-major 3xi32) */
} fwd_t;

/* Parse a CRC-VALIDATED 0x05 payload into fwd_t. Returns 1 on success, 0 on ANY structural
 * rejection. A frame that passes CRC can still be structurally wrong (a CRC proves the bytes are the
 * ones that were sent, not that they mean anything), so every field is checked against the layout
 * rather than trusted:
 *   R1 module_id != 0x05            -> not ours
 *   R2 len < 5                      -> too short for KIND+vid+n_spikes
 *   R3 pl[0] != KIND_RESULT         -> the EOP sentinel (0x02) or an unknown KIND
 *   R4 n_spikes < 2                 -> no gap to decode
 *   R5 n_spikes > ROUTER_MAXSP      -> over the cap; would overrun ch[]/lp3[]
 *   R6 len != 5 + n + 12*(n-1)      -> LENGTH inconsistent with n_spikes: the CRC-valid-but-wrong
 *                                      case, and the ONLY guard standing between a malformed frame
 *                                      and a read past the payload.
 * Layout: [KIND 0x01][vector_id u16 BE][n_spikes u16 BE][ch u8 x n][3 x i32 BE per gap];
 * payload = 5 + n + 12*(n-1). */
static inline int decode_forward(uint8_t module_id, const uint8_t *pl, int len, fwd_t *f) {
    if (module_id != ROUTER_MOD_FORWARD) return 0;                          /* R1 */
    if (len < 5) return 0;                                                  /* R2 */
    if (pl[0] != ROUTER_KIND_RESULT) return 0;                              /* R3 */
    uint16_t n = rtr_u16(pl + 3);
    if (n < 2) return 0;                                                    /* R4 */
    if (n > ROUTER_MAXSP) return 0;                                         /* R5 */
    uint16_t ng = (uint16_t)(n - 1);
    if (len != 5 + (int)n + 4 * ROUTER_NCLASS * (int)ng) return 0;          /* R6 */
    f->vector_id = rtr_u16(pl + 1);
    f->n_spikes = n;
    f->n_gaps = ng;
    for (int i = 0; i < n; i++) f->ch[i] = pl[5 + i];
    const uint8_t *q = pl + 5 + n;
    for (int t = 0; t < (int)ng * ROUTER_NCLASS; t++) f->lp3[t] = rtr_i32(q + 4 * t);
    return 1;
}

/* Ingest one decoded half. Returns:
 *   >0  : a pair completed -> `out` holds a 0x05 forward frame of this length (transmit it).
 *    0  : buffered, waiting for the partner (or ignored half).
 *   -1  : partner found but ALIGNMENT MISMATCH (n_gaps != n_spikes-1) -> nothing forwarded. */
/* B3 — MODULAR "oldest" comparison. `vector_id` is the FE's monotonic u16 `seq`, so no timestamp is
 * needed — but it WRAPS at 65,536 (~78 h at the >=4.27 s/transmission cadence). age > 0 means `a` is
 * older than `b`; the int16_t cast makes the comparison modular, so it stays correct across the
 * wrap. A plain `a < b` would, at the wrap, judge seq 65,530 NEWER than seq 3 and evict the wrong
 * slot — it passes a naive eviction test and fails only after ~78 h of running. */
static inline int rtr_age(uint16_t a, uint16_t b) { return (int16_t)(b - a); }

static inline int router_ingest(router_t *R, const half_t *h, uint8_t *out) {
    if (h->kind != ROUTER_HALF_MARKS && h->kind != ROUTER_HALF_BND) return 0;
    /* find existing slot for this vector_id, else a free one, else the OLDEST (to evict) */
    int si = -1, free_i = -1, old_i = -1;
    for (int i = 0; i < ROUTER_NSLOTS; i++) {
        if (R->slots[i].used && R->slots[i].vector_id == h->vector_id) { si = i; break; }
        if (!R->slots[i].used) { if (free_i < 0) free_i = i; continue; }
        if (old_i < 0 || rtr_age(R->slots[i].vector_id, h->vector_id) >
                         rtr_age(R->slots[old_i].vector_id, h->vector_id))
            old_i = i;                                       /* most-positive age = oldest */
    }
    if (si < 0) {
        if (free_i >= 0) {
            si = free_i;
        } else {
            /* B3: EVICT-OLDEST-ON-FULL replaces the shipped silent `dropped_full++` drop. That drop
             * made the table a MONOTONIC RATCHET: `used` cleared at exactly two sites (pair-complete
             * and mismatch), so an orphaned half — one whose partner never arrives — held its slot
             * FOREVER and the router wedged, silently, after 8 orphans
             * (b2_router_decode_inventory.md 1.3). Eviction means the table CANNOT fatally fill.
             *
             * Why 8 slots is safe with eviction: a slot is freed the moment its pair completes, and
             * transmissions arrive >= 4.27 s apart (T_ITX 3.5 + D_min 0.770) while a pair completes
             * within one gap9 compute+emit (<= ~9.4 s at the A3 cap, and the FE's own 0x06 for the
             * NEXT transmission cannot precede it). Filling all 8 would need 8 transmissions in
             * flight at once, which that cadence makes unreachable — so eviction can only ever
             * displace a GENUINE orphan, never a live half awaiting its partner.
             * Reachable orphan sources: a CRC-dropped 0x03 (fa_feed resyncs silently), and gap9's
             * silent over-cap drop (`ns > MAX_SPIKES`, gap9_hw.c) — the latter now unreachable from
             * the FE, which emits KIND 0x7F and drops above 256 marks. */
            si = old_i;
            R->evictions++;
        }
        R->slots[si].used = 1; R->slots[si].vector_id = h->vector_id;
        R->slots[si].have_marks = R->slots[si].have_bnd = 0;
    }
    pair_slot_t *s = &R->slots[si];
    if (h->kind == ROUTER_HALF_MARKS) {
        s->n_spikes = h->n_spikes;
        for (int i = 0; i < h->n_spikes; i++) s->ch[i] = h->ch[i];
        s->have_marks = 1;
    } else {
        s->n_gaps = h->n_gaps;
        for (int t = 0; t < h->n_gaps * ROUTER_NCLASS; t++) s->q16[t] = h->q16[t];  /* WIDENED (D4) */
        s->have_bnd = 1;
    }
    { int occ = rtr_occupancy(R); if (occ > R->hwm) R->hwm = occ; }

    if (!(s->have_marks && s->have_bnd)) return 0;            /* wait for partner (skew buffer) */

    /* both halves present -> validate alignment (sources.py::validate_alignment) */
    if (s->n_gaps != s->n_spikes - 1) {
        R->mismatches++; s->used = 0; return -1;             /* refuse to forward a misaligned pair */
    }
    int fl = rtr_build_forward(s->vector_id, s->ch, s->n_spikes, s->q16, s->n_gaps, out);
    R->paired++; s->used = 0;                                 /* free the slot */
    return fl;
}
#endif /* ROUTER_CORE_H */
