// spiking_gap9_spk — the SPIKE-FED LIF gap segmenter's forward pass (Exp 213, the spiking network of
// record). The ONLY architecture-specific compute in this module; the Q16.16 primitives, windowed feature
// extraction, softmax->lp3 head, per-gap driver and the 0x06/0x03 transport are SHARED in firmware/common/
// (clf_core.h + clf_hw.c), so this module consumes byte-identical features to the current-fed LIF and the
// MLP. Two things differ from spiking_gap9_lif/clf_forward.c:
//   1. the ENCODER: each standardized window value feat[j] (Q16.16) drives an ON channel with the clipped
//      value zc = clip(feat, 0, SCALE) and an OFF channel with clip(-feat, 0, SCALE); on step t (1-based) the
//      channel has emitted cum_t = floor(t * zc / SCALE) spikes in all, so it spikes at t iff cum_t > cum_{t-1}.
//      Exact integer floor division (both operands non-negative), mirroring fixedpoint_clf.FixedClf.encode_q.
//   2. layer 1 is a SPIKE-INPUT matvec (sum of the Q16.16 weights of the channels that spiked this step, no
//      >>16), accumulated over the active-channel list, like layers 2 and 3.
// BIT-EXACT mirror of spiking_gap9/fixedpoint_clf.FixedClf.forward_logits(kind="spk").
#include "clf_weights.h"      // Q16.16 weights + CLF_D*/BETA/THETA/T/MU/INSD/NIN/SCALE_Q (CLF_IS_SPK=1)
#include "luts.h"             // shared Q16.16 LUT tables            (firmware/common/)
#include "exp_lut.h"          // shared exp LUT                      (firmware/common/)
#include "clf_vectors.h"      // shared baked eval vectors           (firmware/common/)
#define CLF_CORE_IMPL
#include "clf_core.h"         // SHARED compute core (instantiated here) (firmware/common/)

_Static_assert(CLF_IS_SPK == 1, "spiking_gap9_spk needs the spike-fed weights header");
_Static_assert(CLF_NIN == 2 * CLF_NFEAT, "one ON and one OFF channel per window value");
_Static_assert(CLF_D0 == CLF_NIN, "layer 1 reads the encoded channels");

// module state: encoder (clipped values, cumulative spike counts, this step's active channels), int64
// membranes, int32 binary hidden spikes
static int32_t zc[CLF_NIN];
static int32_t cum_prev[CLF_NIN];
static int16_t act[CLF_NIN];
static int64_t mem0[CLF_D1], mem1[CLF_D2], memout[3], accout[3];
static int32_t spk0[CLF_D1], spk1[CLF_D2];

void clf_forward(void) {                                   // reads shared feat[], writes shared logits[]
    for (int j = 0; j < CLF_NFEAT; j++) {                  // encoder: ON = clip(z, 0, S), OFF = clip(-z, 0, S)
        int32_t z = feat[j];
        zc[j]             = (z <= 0) ? 0 : (z > CLF_SCALE_Q ? CLF_SCALE_Q : z);
        zc[CLF_NFEAT + j] = (z >= 0) ? 0 : (-z > CLF_SCALE_Q ? CLF_SCALE_Q : -z);
        cum_prev[j] = 0; cum_prev[CLF_NFEAT + j] = 0;
    }
    for (int k = 0; k < CLF_D1; k++) mem0[k] = 0;
    for (int o = 0; o < CLF_D2; o++) mem1[o] = 0;
    for (int c = 0; c < 3; c++) { memout[c] = 0; accout[c] = 0; }
    for (int t = 1; t <= CLF_T; t++) {
        int na = 0;                                        // input spikes of step t: cum_t - cum_{t-1} in {0,1}
        for (int i = 0; i < CLF_NIN; i++) {
            int32_t c = (int32_t)(((int64_t)t * zc[i]) / CLF_SCALE_Q);
            if (c != cum_prev[i]) act[na++] = (int16_t)i;
            cum_prev[i] = c;
        }
        for (int k = 0; k < CLF_D1; k++) {                 // hidden layer 1: spike-input matvec + LIF
            int64_t I = CLF_B0[k];
            const int32_t *Wr = CLF_W0 + (int)k * CLF_NIN;
            for (int a = 0; a < na; a++) I += Wr[act[a]];
            int64_t m = fxmul_round(CLF_BETA_Q, mem0[k]) + I;
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
