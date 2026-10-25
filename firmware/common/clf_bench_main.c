// Renode/host bench: run clf_run over the baked vectors, emit "vid ng lp0 lp1 lp2 ..." per line on
// uart0 (Renode models only uart0), then a DONE sentinel. compare_clf.py diffs vs the golden JSON.
#include <stdio.h>
#include "pico/stdlib.h"
#include "clf_core.h"
#include "clf_vectors.h"

static int32_t lp3[CLF_MAX_GAPS * 3];

int main(void) {
    stdio_init_all();
    sleep_ms(200);                       // let the Renode UART backend attach before the first bytes
    for (int v = 0; v < CLF_NVEC; v++) {
        int ng = clf_run(v, lp3);
        printf("%d %d", CLF_VEC_VID[v], ng);
        for (int i = 0; i < ng * 3; i++) printf(" %d", (int)lp3[i]);
        printf("\n");
    }
    printf("DONE\n");
    while (true) tight_loop_contents();
    return 0;
}
