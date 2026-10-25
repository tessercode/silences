// Host-only driver: builds the gap9_3c module with native gcc (identical int32/int64 semantics)
// and emits the same framed output as the firmware, so the logic can be checked bit-exactly
// against the A2.0b golden before involving the cross-compiler / Renode.
// (phase2b/firmware/host_main.c pattern, sha 1ff14d5b..., + harness-side table self-checks —
// the module under test, gap9_3c.c, is UNCHANGED by this harness.)
#include <stdio.h>
#include <string.h>
#include "emit.h"
#include "luts.h"          /* originals, for the pop0/beta identity self-check (harness TU only) */
#include "pop_luts_3c.h"

int main(void) {
    /* beta-vector semantic check: pop0's corrected table KEPT the deployed [0,16]/(1/256) grid,
     * so it must be BIT-IDENTICAL to the read-only original — pins the beta provenance of the
     * corrected tables to the deployed ones (a2_golden_reference.md par 6.2). */
    if (memcmp(POP3C_LUTS, POP_LUTS, (POP_N + 1) * sizeof(int32_t)) != 0) {
        fprintf(stderr, "SELF-CHECK FAIL: pop0 table != deployed POP_LUTS[0]\n");
        return 2;
    }
    for (int j = 0; j < 6; j++) {
        if (POP3C_LUTS[j * (POP3C_N + 1)] != 65536) {
            fprintf(stderr, "SELF-CHECK FAIL: pop j(0) != Q_ONE\n");
            return 2;
        }
    }
    gap9_emit_all();
    return 0;
}
