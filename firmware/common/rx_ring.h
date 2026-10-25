/*
 * rx_ring.h — B2a: the shared SPSC byte ring for IRQ RX buffering (decision #10).
 *
 * WHY ITS OWN HEADER, not inside router_core.h: router_core.h is the place that knows every FRAME's
 * shape — the CRC, the assembler, the layouts, the pairing. A ring knows about none of that; it
 * moves bytes and has no idea what a frame is. They are also consumed differently: router_core.h is
 * included by three boards' PARSING paths, while the ring is touched by an ISR. Keeping them apart
 * means the ISR-facing code has one small, auditable surface (this file) rather than being buried in
 * a 300-line frame header. Both live in firmware/common/ and both are dependency-free (stdint only),
 * so the sharing property is identical.
 *
 * ⚠️ B2a IS INVISIBLE TO THE FRAME LAYER. It changes the byte SOURCE, not the bytes. Nothing here
 * knows what a 0x05 is; the goldens are untouched; the B2b.1/2/3 gates are the transparency proof.
 *
 * ============================== SPSC + MEMORY ORDERING (Cortex-M0+) ==========================
 * ONE producer: the ISR, which writes `head` ONLY (and buf[]).
 * ONE consumer: the main loop, which writes `tail` ONLY.
 * Neither ever writes the other's index. That is what makes this correct WITHOUT atomics — the M0+
 * has no LDREX/STREX and no barriers worth the name; what it does guarantee is that an ALIGNED
 * 32-BIT LOAD OR STORE IS ATOMIC (single AHB-Lite transfer, and the core cannot be interrupted
 * mid-word). So:
 *   - `head`/`tail` are `uint32_t` and, being the first two members of a naturally-aligned struct,
 *     are word-aligned. A TORN READ IS THEREFORE IMPOSSIBLE: the consumer never sees half of a head
 *     update. (This is the whole reason the counters are u32 and not, say, u16 packed together.)
 *   - both are `volatile`: the compiler must not cache them across the loop that polls them, and
 *     must not reorder the buf[] write past the head store within rx_ring_push (see below).
 *   - NO BARRIER IS NEEDED, and here is the argument rather than the assertion: a barrier would be
 *     for (a) multi-core visibility or (b) store buffering. Neither applies. The ISR and the main
 *     loop run on the SAME core — exception entry/exit is a context synchronization point by
 *     construction, so anything the ISR stored is visible to the code it interrupted. And the M0+
 *     has no store buffer and no speculation; stores complete in program order. The one ordering
 *     that matters *within* the ISR — "the byte must be in buf[] before head advertises it" — is a
 *     data dependency the compiler cannot reorder across, because `head` is volatile and a volatile
 *     store may not be moved before a preceding write that the abstract machine sequences first.
 *     (If this ever moves to an M0+ pair with both cores touching one ring, that argument dies and
 *     a __dmb() plus real atomics are required. It does not today: one ISR, one main loop, one core.)
 *
 * ============================== FREE-RUNNING COUNTERS ========================================
 * `head`/`tail` are FREE-RUNNING and never masked; only the INDEX is masked (`& RX_RING_MASK`).
 * Occupancy is the MODULAR difference `(uint32_t)(head - tail)`, which stays correct across the u32
 * wrap. A naive `head >= tail` breaks the moment head wraps past 2^32 and tail has not — the ring
 * would report empty while holding a full buffer, or vice versa.
 *   At 960 B/s a u32 wraps in ~52 days. That is long. It is ALSO exactly the shape of B3's
 *   `vector_id` wrap (~78 h) where a plain `<` would have silently corrupted the pairing table — a
 *   bug that passes every test that does not deliberately straddle the wrap. Same class, same
 *   remedy, and rx_ring_test.c carries the same NEGATIVE CONTROL to prove the test can fail.
 * Size MUST be a power of two so the mask is exact.
 *
 * ============================== OVERFLOW: DETECT, COUNT, DO NOT OVERWRITE ====================
 * On full, the ARRIVING byte is dropped and `dropped` increments. The ring's existing contents are
 * never overwritten.
 *
 * ⚠️ WHAT AN OVERFLOW ACTUALLY COSTS — MEASURED, not assumed (b2a_port.md 1.4). The tempting claim
 * is "it loses one transmission". It loses THREE. A truncated frame still carries its declared
 * LENGTH in the header, so fa_feed sits in its payload state and must SWALLOW that many more bytes
 * before it even reaches the CRC — eating the next frame whole and part of the one after, then
 * failing, resetting, and hunting the next 0xA5 0x5A. Measured: the lossy frame + 2, recovering at
 * k+3.
 * It is still CONTAINED, not a wedge: no state is corrupted, fa_feed resyncs on its own, and on the
 * router the orphaned half is reclaimed by B3's eviction. But `dropped` != 0 means ~3 transmissions
 * went missing, not one — and it is the ONLY evidence the sizing assumption (T_ITX = 3.5 s) was
 * violated. The ring is sized (b2a_sizing.py) so that on the paced chain this never happens.
 */
#ifndef RX_RING_H
#define RX_RING_H
#include <stdint.h>

/* Derived from b2a_sizing.py: the worst line (the router's 0x03 line) buckets 746 B while the board
 * emits a cap 0x05; the others are 595/630/602 B. 1024 is the smallest power of two above all four
 * (margin 1.37x on the worst). NOT "one frame per line" — a ring holds what ARRIVES WHILE DEAF. */
#define RX_RING_SIZE 1024u
#define RX_RING_MASK (RX_RING_SIZE - 1u)
_Static_assert((RX_RING_SIZE & RX_RING_MASK) == 0u, "RX_RING_SIZE must be a power of two");
_Static_assert(RX_RING_SIZE >= 746u, "RX_RING_SIZE below the derived worst-line bound (b2a_sizing.py)");

typedef struct {
    volatile uint32_t head;      /* ISR writes ONLY this. Word-aligned -> atomic on M0+. */
    volatile uint32_t tail;      /* main loop writes ONLY this. */
    volatile uint32_t dropped;   /* overflow counter; ISR-written, main-read (diagnostic) */
#ifdef RX_RING_TRACK_HWM
    /* B2.2 silicon: producer-side (ISR) peak occupancy, read over SWD after a run. GUARDED so the
     * deployable image is byte-identical when off; INVISIBLE to the frame layer (bytes/frames
     * unchanged -> the B2b host gates re-pass). Placed after the frozen counters so buf[] alignment
     * is unaffected for the off build. */
    volatile uint32_t hwm;
#endif
    uint8_t  buf[RX_RING_SIZE];
} rx_ring_t;

static inline void rx_ring_init(rx_ring_t *r) {
    r->head = 0u;
    r->tail = 0u;
    r->dropped = 0u;
#ifdef RX_RING_TRACK_HWM
    r->hwm = 0u;
#endif
}

/* Occupancy — the MODULAR difference. Correct across the u32 wrap; `head >= tail` is not. */
static inline uint32_t rx_ring_count(const rx_ring_t *r) {
    return (uint32_t)(r->head - r->tail);
}
static inline int rx_ring_empty(const rx_ring_t *r) { return rx_ring_count(r) == 0u; }
static inline int rx_ring_full(const rx_ring_t *r)  { return rx_ring_count(r) >= RX_RING_SIZE; }

/* PRODUCER — ISR ONLY. Returns 1 if the byte was stored, 0 if the ring was full (byte dropped,
 * `dropped` incremented, existing contents untouched). */
static inline int rx_ring_push(rx_ring_t *r, uint8_t b) {
    const uint32_t h = r->head;
    if ((uint32_t)(h - r->tail) >= RX_RING_SIZE) {   /* full: DROP, never overwrite */
        r->dropped++;
        return 0;
    }
    r->buf[h & RX_RING_MASK] = b;   /* the byte lands BEFORE head advertises it: `head` is volatile,
                                     * so this store may not be reordered after the store below. */
    r->head = h + 1u;               /* free-running; never masked */
#ifdef RX_RING_TRACK_HWM
    { const uint32_t c = (uint32_t)(h + 1u - r->tail); if (c > r->hwm) r->hwm = c; }  /* B2.2 peak */
#endif
    return 1;
}

/* CONSUMER — main loop ONLY. Returns 1 and the byte, or 0 if empty. */
static inline int rx_ring_pop(rx_ring_t *r, uint8_t *out) {
    const uint32_t t = r->tail;
    if (r->head == t) return 0;                      /* empty */
    *out = r->buf[t & RX_RING_MASK];
    r->tail = t + 1u;
    return 1;
}

/* High-water mark, for the fit report. Consumer-side; approximate under concurrency by design. */
static inline uint32_t rx_ring_hwm(const rx_ring_t *r, uint32_t prev) {
    const uint32_t c = rx_ring_count(r);
    return c > prev ? c : prev;
}

#endif /* RX_RING_H */
