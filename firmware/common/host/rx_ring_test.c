/*
 * rx_ring_test.c — B2a ring unit tests (corpus-independent).
 *
 * B3's lesson: a test that cannot fail is not a test. The wrap test therefore ships with a NEGATIVE
 * CONTROL — a `head >= tail` occupancy variant — which must FAIL the wrap test while PASSING every
 * other one. That is what made B3's modular vid rule credible, and it is the same class of bug.
 *
 * Build: gcc -O2 -fsanitize=address,undefined -fno-sanitize-recover=all -I.. -o rx_ring_test
 *        (add -DRX_RING_NEGATIVE_CONTROL for the broken-comparison build)
 */
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#include "rx_ring.h"

/* ---- the negative control: occupancy via `head >= tail` instead of the modular difference ----
 * Everything else identical. This is what a reasonable person writes if they do not think about the
 * u32 wrap, and it passes every test below EXCEPT the wrap one. */
#ifdef RX_RING_NEGATIVE_CONTROL
static uint32_t nc_count(const rx_ring_t *r) {
    return (r->head >= r->tail) ? (r->head - r->tail) : 0u;   /* "obviously" fine... */
}
static int nc_full(const rx_ring_t *r) { return nc_count(r) >= RX_RING_SIZE; }
static int nc_push(rx_ring_t *r, uint8_t b) {
    if (nc_full(r)) { r->dropped++; return 0; }
    r->buf[r->head & RX_RING_MASK] = b;
    r->head = r->head + 1u;
    return 1;
}
static int nc_pop(rx_ring_t *r, uint8_t *o) {
    if (nc_count(r) == 0u) return 0;
    *o = r->buf[r->tail & RX_RING_MASK];
    r->tail = r->tail + 1u;
    return 1;
}
#define PUSH  nc_push
#define POP   nc_pop
#define COUNT nc_count
#define VARIANT "NEGATIVE CONTROL (head >= tail)"
#else
#define PUSH  rx_ring_push
#define POP   rx_ring_pop
#define COUNT rx_ring_count
#define VARIANT "SHIPPING (modular difference)"
#endif

static int fails;
static void check(int ok, const char *what) {
    if (!ok) { printf("  ** FAIL ** %s\n", what); fails++; }
}

static rx_ring_t R;

/* ---- T1: fill / drain / order ---- */
static void t_order(void) {
    rx_ring_init(&R);
    for (int i = 0; i < 300; i++) check(PUSH(&R, (uint8_t)(i & 0xFF)), "T1 push");
    check(COUNT(&R) == 300, "T1 occupancy after 300 pushes");
    for (int i = 0; i < 300; i++) {
        uint8_t b;
        check(POP(&R, &b) == 1, "T1 pop");
        check(b == (uint8_t)(i & 0xFF), "T1 FIFO order");
    }
    check(POP(&R, (uint8_t[]){0}) == 0, "T1 empty after drain");
    printf("T1 fill/drain/order (300 bytes, FIFO): %s\n", fails ? "FAIL" : "OK");
}

/* ---- T2: occupancy across the power-of-two mask (wrap the INDEX, not the counter) ---- */
static void t_mask(void) {
    int f0 = fails;
    rx_ring_init(&R);
    /* push/pop past the mask boundary many times; the index wraps, the counters do not */
    for (int cycle = 0; cycle < 5; cycle++) {
        for (int i = 0; i < (int)RX_RING_SIZE - 1; i++) PUSH(&R, (uint8_t)i);
        check(COUNT(&R) == RX_RING_SIZE - 1, "T2 occupancy at size-1");
        for (int i = 0; i < (int)RX_RING_SIZE - 1; i++) {
            uint8_t b;
            POP(&R, &b);
            check(b == (uint8_t)i, "T2 order across the mask");
        }
        check(COUNT(&R) == 0, "T2 empty");
    }
    check(R.head > RX_RING_SIZE, "T2 the index wrapped (head kept counting)");
    printf("T2 occupancy across the mask (5 cycles x %u): %s\n", RX_RING_SIZE,
           fails > f0 ? "FAIL" : "OK");
}

/* ---- T3: THE COUNTER WRAP — head/tail initialized near 2^32 ----
 * This is the one the negative control must fail. */
static void t_wrap(void) {
    int f0 = fails;
    rx_ring_init(&R);
    /* place the counters 5 bytes short of the u32 wrap, ring empty */
    R.head = R.tail = 0xFFFFFFFBu;
    check(COUNT(&R) == 0, "T3 empty at head=tail near 2^32");

    /* push 10 bytes: head wraps THROUGH zero while tail stays behind it */
    for (int i = 0; i < 10; i++) check(PUSH(&R, (uint8_t)(0xA0 + i)), "T3 push across the wrap");
    check(R.head == 5u, "T3 head wrapped to 5");
    check(R.tail == 0xFFFFFFFBu, "T3 tail still pre-wrap");
    /* THE assertion: occupancy is 10, not 0 and not 4-billion-something. `head >= tail` says
     * head(5) < tail(0xFFFFFFFB) -> reports 0 -> the ring claims EMPTY while holding 10 bytes. */
    check(COUNT(&R) == 10u, "T3 MODULAR occupancy across the u32 wrap");

    for (int i = 0; i < 10; i++) {
        uint8_t b;
        check(POP(&R, &b) == 1, "T3 pop across the wrap");
        check(b == (uint8_t)(0xA0 + i), "T3 data intact across the wrap");
    }
    check(COUNT(&R) == 0, "T3 empty after the wrap drain");
    printf("T3 counter wrap at 2^32 (%s): %s\n", VARIANT, fails > f0 ? "FAIL" : "OK");
}

/* ---- T4: overflow — drop the ARRIVING byte, count it, contents INTACT ---- */
static void t_overflow(void) {
    int f0 = fails;
    rx_ring_init(&R);
    for (uint32_t i = 0; i < RX_RING_SIZE; i++) check(PUSH(&R, (uint8_t)(i & 0xFF)), "T4 fill");
    check(COUNT(&R) == RX_RING_SIZE, "T4 full");
    check(rx_ring_full(&R), "T4 rx_ring_full");

    check(PUSH(&R, 0xEE) == 0, "T4 push on full is REJECTED");
    check(R.dropped == 1u, "T4 dropped counter incremented");
    check(COUNT(&R) == RX_RING_SIZE, "T4 occupancy unchanged after the drop");
    for (int i = 0; i < 20; i++) PUSH(&R, 0xEE);
    check(R.dropped == 21u, "T4 dropped counts every rejected byte");

    /* the ring's EXISTING contents must be intact — no overwrite */
    for (uint32_t i = 0; i < RX_RING_SIZE; i++) {
        uint8_t b;
        POP(&R, &b);
        check(b == (uint8_t)(i & 0xFF), "T4 contents INTACT (no overwrite)");
    }
    printf("T4 overflow: drop + count + no overwrite: %s\n", fails > f0 ? "FAIL" : "OK");
}

/* ---- T5: ISR/main interleave — the producer fires at arbitrary points inside the consumer ----
 * A single-threaded push-then-pop test cannot see this surface at all. */
static void t_interleave(void) {
    int f0 = fails;
    for (int phase = 0; phase < 64; phase++) {
        rx_ring_init(&R);
        uint32_t produced = 0, consumed = 0;
        uint8_t next_expect = 0;
        /* stream 4000 bytes through a 1024 ring, "interrupting" the consumer every `phase+1` pops */
        while (consumed < 4000) {
            /* --- simulated ISR: produce a burst at an arbitrary point --- */
            for (int k = 0; k <= (phase & 7) && produced < 4000; k++)
                if (PUSH(&R, (uint8_t)(produced & 0xFF))) produced++;
            /* --- consumer --- */
            for (int k = 0; k <= phase && consumed < 4000; k++) {
                uint8_t b;
                if (!POP(&R, &b)) break;
                check(b == next_expect, "T5 no byte lost/reordered under interleave");
                next_expect++;
                consumed++;
            }
        }
        check(R.dropped == 0, "T5 no drops (the consumer keeps up)");
        check(COUNT(&R) == 0, "T5 drained");
    }
    printf("T5 ISR/main interleave (64 phases x 4000 bytes): %s\n", fails > f0 ? "FAIL" : "OK");
}

int main(void) {
    printf("=== rx_ring unit tests — %s ===\n", VARIANT);
    t_order();
    t_mask();
    t_wrap();
    t_overflow();
    t_interleave();
    printf(fails ? "\n%d CHECK(S) FAILED\n" : "\nALL RING UNIT TESTS PASS\n", fails);
    return fails ? 1 : 0;
}
