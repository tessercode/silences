// SHARED host-gcc gate (used by both modules). Links against a module's clf_forward.c (which pulls in
// clf_core.h's CLF_CORE_IMPL + that module's weights); runs clf_run over the baked vectors and prints
// "vid ng lp0 lp1 ..." per line. clf_compare.py diffs it bit-exact vs the golden JSON.
//   gcc -I<module_dir> -I<common> clf_host_main.c <module_dir>/clf_forward.c -o host
#include <stdio.h>
#include "clf_core.h"
#include "clf_vectors.h"

static int32_t lp3[CLF_MAX_GAPS * 3];

int main(void) {
    for (int v = 0; v < CLF_NVEC; v++) {
        int ng = clf_run(v, lp3);
        printf("%d %d", CLF_VEC_VID[v], ng);
        for (int i = 0; i < ng * 3; i++) printf(" %d", (int)lp3[i]);
        printf("\n");
    }
    return 0;
}
