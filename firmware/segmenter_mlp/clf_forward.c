// spiking_gap9_mlp — the tanh MLP gap segmenter's forward pass. This is the ONLY architecture-specific
// compute in this module; everything else (Q16.16 primitives, windowed feature extraction, softmax->lp3
// head, the per-gap driver, and the 0x06/0x03 transport) is SHARED in firmware/common/ (clf_core.h +
// clf_hw.c). A 2-hidden-layer feedforward net: matvec + tanh. BIT-EXACT mirror of
// spiking_gap9/ff_model.FFModel._logits.
#include "clf_weights.h"      // Q16.16 weights + CLF_D* (MLP model; CLF_IS_LIF=0)
#include "luts.h"             // shared Q16.16 LUT tables            (firmware/common/)
#include "exp_lut.h"          // shared exp LUT                      (firmware/common/)
#include "clf_vectors.h"      // shared baked eval vectors           (firmware/common/)
#define CLF_CORE_IMPL
#include "clf_core.h"         // SHARED compute core (instantiated here) (firmware/common/)

// MLP-specific hidden activations (Q16.16). (The LIF module has its own membrane/spike buffers.)
static int32_t h0[CLF_D1], h1[CLF_D2];

void clf_forward(void) {                                   // reads shared feat[], writes shared logits[]
    matvec_q16(CLF_W0, CLF_D1, CLF_D0, feat, mv);
    for (int k = 0; k < CLF_D1; k++) h0[k] = tanh_q(mv[k] + CLF_B0[k]);
    matvec_q16(CLF_W1, CLF_D2, CLF_D1, h0, mv);
    for (int k = 0; k < CLF_D2; k++) h1[k] = tanh_q(mv[k] + CLF_B1[k]);
    matvec_q16(CLF_W2, 3, CLF_D2, h1, mv);
    for (int c = 0; c < 3; c++) logits[c] = mv[c] + CLF_B2[c];
}
