// Front-end fixed-point module — Phase 2b C port of frontend_ref.py (bit-exact).
#ifndef FRONTEND_H
#define FRONTEND_H
#include <stdint.h>

#define FE_MAXN 256       // B1.1: max marks per TRANSMISSION (sentence scale; was 64 per word).
                          // == N_MARKS_MAX in frontend_stage3.c (static-asserted there); sizes the
                          // mark arrays AND frontend.c's estimator scratch (isis[]/d[]) identically.
#define FE_MAX_T 1600

// Raw per-mark timing handed to an estimator (the coarsest honest interface).
typedef struct {
    const int64_t *end_ticks;   // mark end times (microsecond ticks)
    const int64_t *dur_ticks;   // mark durations (ticks)
    const int8_t  *ch;          // 0=dot, 1=dash
    int n;
} timing_t;

// THE SWAP POINT: T_u-hat estimation behind a function-pointer table. A third estimator drops in
// by adding a function and extending the table; callers only use estimate_tu(id, timing).
typedef int64_t (*estimate_tu_fn)(const timing_t *);
int64_t estimate_tu(int id, const timing_t *t);   // id 0 = shortest_isi, 1 = antimode
#define FE_EST_SHORTEST_ISI 0
#define FE_EST_ANTIMODE     1
#define FE_N_ESTIMATORS     2
#define FE_SENTINEL ((int64_t)-1)                 // "no estimate" (Python None)

// Consumers take tu_ticks as an INPUT (insulated from which estimator produced it):
int  fe_frame_letter(const int64_t *end_g, const int8_t *ch_g, int n, int64_t tu_ticks,
                     int *out_idx, int *out_ch);  // returns count; idx/ch in [0,FE_MAX_T)
int64_t fe_tick_convert(int64_t sq_seconds_q16);  // Q16.16 seconds -> integer microsecond ticks

#endif
