// Shared framed-output emitter — identical text on host (stdout) and firmware (UART), so one
// comparator checks both. Integer-only printing (no %f), keeping the FPU-free claim intact.
// A2.1 3-class form of phase2b/firmware/emit.h (sha dd0325d2...): the per-gap line carries the
// THREE Q16.16 log-posteriors (D4 argmax-free) instead of lpb+argmax.
#ifndef GAP9_3C_EMIT_H
#define GAP9_3C_EMIT_H
#include <stdint.h>
#include <stdio.h>
#include "gap9_3c.h"
#include "vectors_3c.h"
#if PICO_ON_DEVICE
#include "pico/time.h"   /* A2.2: time_us_32 latency bracket (device bench only; host unchanged) */
#endif

static inline void gap9_emit_all(void) {
    static int32_t lp3[3 * MAX_GAPS];
    printf("GAP9_3C_BEGIN\n");
    for (int vi = 0; vi < N_VEC3; vi++) {
#if PICO_ON_DEVICE
        /* A2.2 latency: bracket gap9_run ONLY (emit excluded) — the GAP9_HARDWARE.md:69-76
         * measurement basis. The T line is ignored by compare_output.py (unknown-token skip). */
        uint32_t t0_us = time_us_32();
        int ng = gap9_run(vi, lp3);
        uint32_t dt_us = time_us_32() - t0_us;
#else
        int ng = gap9_run(vi, lp3);
#endif
        printf("VEC %ld %d\n", (long)VEC3_ID[vi], ng);
        for (int t = 0; t < ng; t++) {
            printf("G %d %ld %ld %ld\n", t,
                   (long)lp3[t * 3], (long)lp3[t * 3 + 1], (long)lp3[t * 3 + 2]);
        }
        printf("EVEC %ld\n", (long)VEC3_ID[vi]);
#if PICO_ON_DEVICE
        printf("T %ld %d %lu\n", (long)VEC3_ID[vi], ng, (unsigned long)dt_us);
#endif
    }
    printf("DONE %d\n", (int)N_VEC3);
}

#endif
