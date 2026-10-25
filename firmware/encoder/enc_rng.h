/*
 * enc_rng.h — shared, seedable, CROSS-TOOLCHAIN-BIT-EXACT PRNG + Gaussian transform for the encoder
 * jitter increment (increment 2). Header-only, compiled byte-identically by BOTH the RP2040 firmware
 * (arm-none-eabi-gcc, cortex-m0plus soft-float) AND the native host proxy / reference (x86-64 gcc,
 * CPython port in enc_rng_ref.py). Same seed -> same uint32 stream -> same normals -> same render.
 *
 * WHY A HAND-ROLLED log/cos (the whole trick):
 *   Increment 1 was bit-exact device-vs-host because it used ONLY IEEE basic ops (+,-,*, lrint), which
 *   +,-,*,/ are correctly-rounded and IDENTICAL on any IEEE-754 double platform (arm soft-float libgcc
 *   == x86-64 SSE2 == CPython float), and lrint/frexp/sqrt are EXACT / correctly-rounded (IEEE mandates
 *   it; glibc, newlib and CPython all comply). But libm's TRANSCENDENTALS (log, cos, sin, exp) are NOT
 *   correctly-rounded in general — glibc-x86 and newlib-arm can differ in the last ULP — so a Box-Muller
 *   that called libm log()/cos() would DIVERGE device-vs-host on near-boundary inputs, breaking the
 *   recovered bit-exact-given-seed invariant. So log and cos are implemented HERE in pure double +,-,*,/
 *   (deterministic on every IEEE platform); only sqrt/frexp/lrint are taken from libm (safe, exact).
 *
 * BUILD REQUIREMENT: compile with -ffp-contract=off on the host so no FMA fuses a*b+c into one rounding
 * (the M0+ has no FMA, so the device is contraction-free by ISA; -ffp-contract=off makes the x86 host
 * match). No -mfma / -march=native on the host build. See build_host.sh / verify_jitter.py gcc flags.
 *
 * PRNG: xoshiro128** (Blackman & Vigna) — 128-bit state (4x uint32), all-integer core, tiny, fast,
 * FPU-free, and trivially portable (uint32 wraparound == Python `& 0xFFFFFFFF`). Seeded via SplitMix64.
 */
#ifndef ENC_RNG_H
#define ENC_RNG_H
#include <stdint.h>
#include <math.h>      /* frexp, ldexp, lrint — EXACT ops (bit extraction / power-of-2 scale / integer
                        * round), identical on every toolchain. NOTE: libm sqrt/log/cos are NOT used —
                        * sqrt is self-implemented (enc_sqrt) because the RP2040 SDK's software double
                        * sqrt (dsqrt_shim) is an approximation that need not match x86's correctly-
                        * rounded sqrt; doing it ourselves keeps device==host by construction. */

/* Shared math constants (identical decimal->double on gcc and CPython). 2*PI computed as 2.0*PI so the
 * device and both host ports use the exact same double. */
#define ENC_PI      3.141592653589793
#define ENC_TWO_PI  6.283185307179586      /* == 2.0*ENC_PI (nearest double) */
#define ENC_HALF_PI 1.5707963267948966
#define ENC_LN2     0.6931471805599453     /* log(2), nearest double */

/* ---------------------------------------------------------------------------------------------------
 * xoshiro128** integer core
 * ------------------------------------------------------------------------------------------------- */
typedef struct { uint32_t s[4]; } enc_rng_t;

static inline uint32_t enc_rotl32(uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }

static inline uint32_t enc_rng_next(enc_rng_t *st) {
    uint32_t *s = st->s;
    const uint32_t result = enc_rotl32(s[1] * 5u, 7) * 9u;    /* **-scrambler */
    const uint32_t t = s[1] << 9;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = enc_rotl32(s[3], 11);
    return result;
}

/* Seed the 128-bit state from a single 64-bit seed via SplitMix64 (standard xoshiro init): run
 * SplitMix64 four times, take the LOW 32 bits of each output as s[0..3]. SplitMix64 output is
 * well-mixed for any seed, so the state is never all-zero for practical seeds (incl. 0). */
static inline void enc_rng_seed(enc_rng_t *st, uint64_t seed) {
    uint64_t z = seed;
    for (int i = 0; i < 4; ++i) {
        z += 0x9E3779B97F4A7C15ULL;
        uint64_t x = z;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        x = x ^ (x >> 31);
        st->s[i] = (uint32_t)(x & 0xFFFFFFFFu);
    }
}

/* uint32 -> double in the OPEN interval (0,1): (x + 0.5) / 2^32. Never 0 (avoids log(0) in Box-Muller)
 * nor 1. Every op is exact (uint32<2^53, 0.5 exact, 2^-32 exact power-of-two scale) -> deterministic. */
static inline double enc_rng_unit(enc_rng_t *st) {
    return ((double)enc_rng_next(st) + 0.5) * (1.0 / 4294967296.0);   /* 4294967296 = 2^32 */
}

/* ---------------------------------------------------------------------------------------------------
 * Deterministic natural log for x in (0,1] (pure IEEE +,-,*,/ + frexp; NO libm log)
 *   x = m * 2^e (frexp, exact), shift m into [sqrt(1/2), sqrt(2)); f = (m-1)/(m+1), |f| <= 0.1716;
 *   log(m) = 2*(f + f^3/3 + f^5/5 + ... + f^17/17)   (atanh series; ~1e-16 truncation at |f|<=0.172)
 *   log(x) = e*ln2 + log(m)
 * ------------------------------------------------------------------------------------------------- */
static inline double enc_log(double x) {
    int e;
    double m = frexp(x, &e);                         /* x = m*2^e, m in [0.5,1) ; frexp exact/portable */
    if (m < 0.7071067811865476) { m *= 2.0; e -= 1; } /* -> m in [0.7071,1.4142) */
    double f  = (m - 1.0) / (m + 1.0);               /* |f| <= 0.1716 */
    double f2 = f * f;
    double term = f;                                 /* f^1 */
    double sum  = f;
    term *= f2; sum += term /  3.0;                  /* f^3/3 */
    term *= f2; sum += term /  5.0;
    term *= f2; sum += term /  7.0;
    term *= f2; sum += term /  9.0;
    term *= f2; sum += term / 11.0;
    term *= f2; sum += term / 13.0;
    term *= f2; sum += term / 15.0;
    term *= f2; sum += term / 17.0;                  /* f^17/17 */
    return (double)e * ENC_LN2 + 2.0 * sum;
}

/* Deterministic square root (pure IEEE +,*,/ + exact frexp/ldexp; NO libm sqrt). x>=0. sqrt(x) =
 * sqrt(m)*2^(e/2) with x=m*2^e (even e), sqrt(m) by Newton from y0=1 (6 iterations -> converged double
 * precision for m in [0.5,2)). Not guaranteed correctly-rounded, but BIT-IDENTICAL on arm and x86
 * (every step is a correctly-rounded basic op), which is what bit-exactness needs. */
static inline double enc_sqrt(double x) {
    if (x <= 0.0) return 0.0;
    int e;
    double m = frexp(x, &e);                         /* m in [0.5,1), x = m*2^e */
    if (e & 1) { m *= 2.0; e -= 1; }                 /* even exponent -> m in [0.5,2) */
    double y = 1.0;
    y = 0.5 * (y + m / y);
    y = 0.5 * (y + m / y);
    y = 0.5 * (y + m / y);
    y = 0.5 * (y + m / y);
    y = 0.5 * (y + m / y);
    y = 0.5 * (y + m / y);
    return ldexp(y, e / 2);                          /* * 2^(e/2), exact */
}

/* Deterministic cosine (pure IEEE +,-,*; NO libm cos). Range-reduce theta in (0,2pi) to [0,pi/2] via
 * cos symmetry, then a Taylor series to a^18/18! (|a|<=pi/2, a^2<=2.47 -> ~3e-15 truncation). */
static inline double enc_cos(double theta) {
    double a = theta;
    while (a >= ENC_TWO_PI) a -= ENC_TWO_PI;         /* fold into [0,2pi) (input already is; guard only) */
    while (a < 0.0)         a += ENC_TWO_PI;
    if (a > ENC_PI) a = ENC_TWO_PI - a;              /* cos(2pi-a)=cos(a) -> a in [0,pi] */
    double sign = 1.0;
    if (a > ENC_HALF_PI) { a = ENC_PI - a; sign = -1.0; }  /* cos(a)=-cos(pi-a) -> a in [0,pi/2] */
    double a2 = a * a;
    /* Horner in a2: 1 - a2/2! + a2^2/4! - ... - a2^9/18! */
    double c = 1.0 - a2*(1.0/2.0
                 - a2*(1.0/24.0
                 - a2*(1.0/720.0
                 - a2*(1.0/40320.0
                 - a2*(1.0/3628800.0
                 - a2*(1.0/479001600.0
                 - a2*(1.0/87178291200.0
                 - a2*(1.0/20922789888000.0
                 - a2*(1.0/6402373705728000.0)))))))));
    return sign * c;
}

/* Standard Normal N(0,1) via Box-Muller: consumes EXACTLY two uniforms per call and returns z0 (the
 * z1 companion is discarded, NOT cached — so the per-word draw sequence is a pure function of (seed,
 * draw-index), trivially mirrorable on host). scale by sigma / shift by mu at the call site. */
static inline double enc_rng_normal(enc_rng_t *st) {
    double u1 = enc_rng_unit(st);                    /* (0,1) */
    double u2 = enc_rng_unit(st);                    /* (0,1) */
    double r  = enc_sqrt(-2.0 * enc_log(u1));        /* self-implemented sqrt (device==host by construction) */
    return r * enc_cos(ENC_TWO_PI * u2);
}

#endif /* ENC_RNG_H */
