// Front-end fixed-point module — Phase 2b C port. BIT-EXACT mirror of frontend_ref.py.
//
// Integer-only. The non-trivial part vs SG-64/gap9: these estimators contain a SORT, a percentile
// interpolation, and divisions. The shortest-ISI estimator uses exact-rational arithmetic
// (percentile = (10*s[k] + r*(s[k+1]-s[k]))/10, scaled by 9/20). The antimode carries its 2-means
// centroids in Q16.16 so the iteration trajectory tracks float's fractional means. All divisions are
// round-to-nearest (round_div). int64 throughout.
#include "frontend.h"

#define FE_KFRAC 16
#define FE_KONE  (1LL << FE_KFRAC)
#define TICK_HZ  1000000LL
#define T_U_REF_MS 60LL

// round(a/b) for integers, round-to-nearest (half away from zero); b > 0. Mirrors _round_div.
static int64_t round_div(int64_t a, int64_t b) {
    if (a >= 0) return (a + b / 2) / b;
    return -(((-a) + b / 2) / b);
}

static void sort_i64(int64_t *a, int n) {        // insertion sort (small arrays)
    for (int i = 1; i < n; i++) {
        int64_t key = a[i]; int j = i - 1;
        while (j >= 0 && a[j] > key) { a[j + 1] = a[j]; j--; }
        a[j + 1] = key;
    }
}

// ---- IMPL A: shortest-ISI percentile (on END TICKS) ----
static int64_t est_shortest_isi(const timing_t *t) {
    if (t->n < 2) return FE_SENTINEL;
    int64_t isis[FE_MAXN]; int m = 0;
    for (int i = 0; i < t->n - 1; i++) {
        int64_t d = t->end_ticks[i + 1] - t->end_ticks[i];
        if (d > 0) isis[m++] = d;
    }
    if (m == 0) return FE_SENTINEL;
    sort_i64(isis, m);
    int pos = m - 1, k = pos / 10, r = pos - 10 * k;
    int64_t pct_num, pct_den;
    if (r == 0 || k + 1 >= m) { pct_num = isis[k]; pct_den = 1; }
    else { pct_num = 10 * isis[k] + (int64_t)r * (isis[k + 1] - isis[k]); pct_den = 10; }
    return round_div(pct_num * 9, pct_den * 20);  // *0.90/2 = *9/20 (exact)
}

// ---- IMPL B: antimode 2-means (on DURATIONS), Q16.16 centroids ----
static int64_t est_antimode(const timing_t *t) {
    if (t->n < 2) return FE_SENTINEL;
    int64_t d[FE_MAXN];
    for (int i = 0; i < t->n; i++) d[i] = t->dur_ticks[i];
    sort_i64(d, t->n);
    if (d[t->n - 1] - d[0] < 1) return FE_SENTINEL;
    int64_t c0 = d[0] * FE_KONE, c1 = d[t->n - 1] * FE_KONE;
    for (int it = 0; it < 20; it++) {
        int64_t mid = (c0 + c1) >> 1;
        int64_t s0 = 0, s1 = 0; int n0 = 0, n1 = 0;
        for (int i = 0; i < t->n; i++) {
            if ((d[i] * FE_KONE) <= mid) { s0 += d[i]; n0++; }
            else { s1 += d[i]; n1++; }
        }
        if (n0 == 0 || n1 == 0) break;
        int64_t nc0 = round_div(s0 * FE_KONE, n0);
        int64_t nc1 = round_div(s1 * FE_KONE, n1);
        if (nc0 == c0 && nc1 == c1) break;
        c0 = nc0; c1 = nc1;
    }
    return round_div(c0 * 87, 100LL * FE_KONE);   // *K_BIAS = 0.87
}

// ---- the swap point ----
static const estimate_tu_fn ESTIMATORS[FE_N_ESTIMATORS] = { est_shortest_isi, est_antimode };

int64_t estimate_tu(int id, const timing_t *t) {
    if (id < 0 || id >= FE_N_ESTIMATORS) return FE_SENTINEL;
    return ESTIMATORS[id](t);
}

// ---- consumers (insulated from the estimator) ----
int fe_frame_letter(const int64_t *end_g, const int8_t *ch_g, int n, int64_t tu,
                    int *out_idx, int *out_ch) {
    if (n == 0 || tu <= 0) return 0;
    int64_t first_dur = (ch_g[0] == 1 ? 3 : 1) * tu;
    int64_t t0 = end_g[0] - first_dur; if (t0 < 0) t0 = 0;
    int cnt = 0;
    for (int i = 0; i < n; i++) {
        int64_t idx = round_div((end_g[i] - t0) * T_U_REF_MS, tu);
        if (idx >= 0 && idx < FE_MAX_T) { out_idx[cnt] = (int)idx; out_ch[cnt] = ch_g[i]; cnt++; }
    }
    return cnt;
}

int64_t fe_tick_convert(int64_t sq_seconds_q16) {
    return round_div(sq_seconds_q16 * TICK_HZ, FE_KONE);
}
