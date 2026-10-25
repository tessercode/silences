// Stand-in for Vigna's f2x.c, which reference/xoshiro128starstar.c includes only for its arbitrary-jump
// functions (jump_ce, jump_n). The known-answer driver never jumps, so these only need to compile; they
// abort if called. The vendored generator itself is used unmodified.
#define POLY_WORDS 2                         /* POLY_DEG 128 bits in 64-bit words */
static void f2x_jumppoly_ce(uint64_t c, uint32_t e, uint64_t *poly) { (void)c; (void)e; (void)poly; abort(); }
static void f2x_jumppoly_n(const uint64_t *jump, int n, uint64_t *poly) { (void)jump; (void)n; (void)poly; abort(); }
