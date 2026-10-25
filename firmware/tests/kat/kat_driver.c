// Known-answer driver: the published SplitMix64 and xoshiro128** 1.1 (reference/, unmodified) seeding
// and stepping the generator the encoder uses. For each seed on the command line it prints the four
// state words, the first N_STREAM outputs, and the sum and xor of the first N_FOLD outputs.
//   cc -O2 -I. -o kat_driver kat_driver.c && ./kat_driver 0 1 7 ...      (run from this directory)
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define next splitmix64_next
#include "reference/splitmix64.c"     // unmodified (sha256 in provenance.json)
#undef next
#define next xoshiro_next
#include "reference/xoshiro128starstar.c"   // unmodified; its #include "f2x.c" finds the stand-in here
#undef next

#define N_STREAM 16
#define N_FOLD 10000

static void seed(uint64_t sd) {
    x = sd;                                   // splitmix64's state
    for (int i = 0; i < 4; i++) s[i] = (uint32_t)splitmix64_next();
}

int main(int argc, char **argv) {
    for (int a = 1; a < argc; a++) {
        uint64_t sd = strtoull(argv[a], NULL, 10);
        seed(sd);
        printf("%llu state %u %u %u %u stream", (unsigned long long)sd, s[0], s[1], s[2], s[3]);
        for (int i = 0; i < N_STREAM; i++) printf(" %u", xoshiro_next());
        seed(sd);
        uint32_t sum = 0, xr = 0;
        for (int i = 0; i < N_FOLD; i++) { uint32_t v = xoshiro_next(); sum += v; xr ^= v; }
        printf(" fold %u %u\n", sum, xr);
    }
    return 0;
}
