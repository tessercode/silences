"""The five-board chain in software (Section 9), stage for stage:

    encoder  --waveform-->  front end  --0x06-->  segmenter  --0x03-->  router  --0x05-->  decoder

compose() runs one keyed sentence through every stage with the boards' own arithmetic and returns
every intermediate: the waveform, the 0x06 frame, the segmenter's posteriors, the 0x03 and 0x05
frames and the decoded text. Its outputs are the chain goldens the hardware runs are compared with.
"""
import numpy as np

from . import encoder_ref, frames, front_end
from .decoder.beam import decode_tx
from .operators import OPERATORS

# The encoder console's operator numbers (":fist N"), as the chain goldens record them.
CONSOLE_FIST = {0: "slow_normalr", 1: "fast_normalr", 2: "high_r_bug", 3: "cootie_high_jitter", 5: "cootie_corrected"}
REPRESENTATIVE_FIST = 5


def compose(text, fist, seed, segmenter, seq=0):
    """segmenter: a function (ticks, ch, tu_us) -> lp3 rows (the board's fixed-point model)."""
    op = OPERATORS[CONSOLE_FIST[fist]]
    ticks, chans, _, _, words = encoder_ref.render(text, op["wpm"], op["r"], op["sigma_elem"], op["sigma_gap"],
                                                   op["omega"], seed)
    wave = encoder_ref.waveform(ticks, chans)
    rise, fall, ch = (np.array([w[k] for w in wave], dtype=np.int64) for k in range(3))
    captures, _, n_mid, n_orph = front_end.split_transmissions(rise, fall)
    f06, rec = front_end.frame_for_capture(seq, rise, fall, ch, captures[0][0], captures[0][1])
    out = dict(text=text, words=words, fist=fist, fist_name=op["name"], seed=seed, n_marks=len(wave),
               waveform_us=wave, n_captures=len(captures), n_midmark=n_mid, n_orphan=n_orph,
               overflow=rec is None, hex_0x06=f06.hex())
    if rec is None:                     # too long: the chain stops at the front end
        return out
    m_ticks, m_ch = [m[0] for m in rec["marks"]], [m[1] for m in rec["marks"]]
    lp3 = segmenter(m_ticks, m_ch, rec["tu_us"])
    text_out, seg, score = decode_tx(m_ch, lp3)
    out.update(tu_us=rec["tu_us"], marks=rec["marks"], n_gaps=len(lp3), lp3_q16=lp3,
               hex_0x03=frames.segmenter_frame(seq, lp3).hex(), hex_0x05=frames.router_frame(seq, m_ch, lp3).hex(),
               decoded_text=text_out, seg=seg, final_score_q16=score, beam_died=text_out is None)
    return out
