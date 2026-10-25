"""Playing the Silences: gap segmentation for word-level decoding of TEMPO Morse.

The package is organised as the paper is: synthetic operators render text into TEMPO spike
trains (corpus), a front end estimates the time unit and builds per-gap features (front_end,
features), a segmenter classifies every gap as intra-letter, inter-letter or inter-word
(segmenters), and a beam search with a character language model turns those posteriors into text
(decoder). evaluation holds the protocol, the decoder-weight fits and the statistics; training
retrains every model; golden writes the vectors the RP2040 firmware is checked against.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # the repository root
DATA = ROOT / "data"
MODELS = ROOT / "models"
RECORDS = ROOT / "records"
PROTOCOL = ROOT / "protocol"
FIRMWARE = ROOT / "firmware"
OUTPUTS = ROOT / "outputs"                         # regenerated intermediates, never committed
