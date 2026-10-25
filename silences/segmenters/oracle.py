"""The oracle segmenter: confident log-posteriors from the true gap labels. Its decoded error is the
residual the channel errors leave, the ceiling no segmenter can pass (Sections 3 and 8)."""

CONF = 20.0                  # the confidence at which the oracle's decoded error stops changing
CONF_PRE_CALIBRATION = 8.0   # the value the random-letter record at T = 1 used, before that was measured
TRUE_LP = -1096              # log-posterior given to the true class, in Q16.16


def producer(conf=CONF):
    def _p(tx):
        rows = []
        for lab in tx["labels"]:
            r = [-int(conf * 65536)] * 3
            r[int(lab)] = TRUE_LP
            rows.append(r)
        return rows
    return _p
