// spiking_gap9_lif — the LIF SPIKING gap segmenter's forward pass. This is the ONLY
// architecture-specific compute in this module; everything else (Q16.16 primitives, windowed feature
// extraction, softmax->lp3 head, the per-gap driver, and the 0x06/0x03 transport) is SHARED in
// firmware/common/ (clf_core.h + clf_hw.c). A fixed-beta feedforward LIF net: integrate weighted input
// on the membrane, spike at threshold, reset by subtraction, over T timesteps (sg64.c pattern).
// BIT-EXACT mirror of spiking_gap9/snn_model.SNNModel._forward_numpy.
#include "clf_weights.h"      // Q16.16 weights + CLF_D*/BETA/THETA/T/MU/INSD (LIF model; CLF_IS_LIF=1)
#include "luts.h"             // shared Q16.16 LUT tables            (firmware/common/)
#include "exp_lut.h"          // shared exp LUT                      (firmware/common/)
#include "clf_vectors.h"      // shared baked eval vectors           (firmware/common/)
#define CLF_CORE_IMPL
#include "clf_core.h"         // SHARED compute core (instantiated here) (firmware/common/)

// LIF-specific state: int64 membranes, int32 binary spikes. (The MLP module has its own buffers.)
static int64_t mem0[CLF_D1], mem1[CLF_D2], memout[3], accout[3];
static int32_t spk0[CLF_D1], spk1[CLF_D2], I0[CLF_D1];

void clf_forward(void) {                                   // reads shared feat[], writes shared logits[]
    matvec_q16(CLF_W0, CLF_D1, CLF_D0, feat, mv);          // layer0 input current (constant over T)
    for (int k = 0; k < CLF_D1; k++) { I0[k] = mv[k] + CLF_B0[k]; mem0[k] = 0; }
    for (int o = 0; o < CLF_D2; o++) mem1[o] = 0;
    for (int c = 0; c < 3; c++) { memout[c] = 0; accout[c] = 0; }
    for (int t = 0; t < CLF_T; t++) {
        for (int k = 0; k < CLF_D1; k++) {                 // hidden layer 1: spike + subtract-reset
            int64_t m = fxmul_round(CLF_BETA_Q, mem0[k]) + I0[k];
            int s = (m >= CLF_THETA_Q);
            mem0[k] = s ? (m - CLF_THETA_Q) : m; spk0[k] = s;
        }
        for (int o = 0; o < CLF_D2; o++) {                 // hidden layer 2: spike-input matvec (no >>16)
            int64_t I = CLF_B1[o];
            const int32_t *Wr = CLF_W1 + (int)o * CLF_D1;
            for (int i = 0; i < CLF_D1; i++) if (spk0[i]) I += Wr[i];
            int64_t m = fxmul_round(CLF_BETA_Q, mem1[o]) + I;
            int s = (m >= CLF_THETA_Q);
            mem1[o] = s ? (m - CLF_THETA_Q) : m; spk1[o] = s;
        }
        for (int c = 0; c < 3; c++) {                      // readout: leaky membrane integrator
            int64_t I = CLF_B2[c];
            const int32_t *Wr = CLF_W2 + (int)c * CLF_D2;
            for (int i = 0; i < CLF_D2; i++) if (spk1[i]) I += Wr[i];
            memout[c] = fxmul_round(CLF_BETA_Q, memout[c]) + I;
            accout[c] += memout[c];
        }
    }
    for (int c = 0; c < 3; c++) logits[c] = (int32_t)floordiv(accout[c], CLF_T);
}
