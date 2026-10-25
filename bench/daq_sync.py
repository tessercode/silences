#!/usr/bin/env python3
"""
Reusable clock-locked DAQ generate+capture primitive (Stage 2 RTSI7 method).

generate_capture(do_data, rate) drives `do_data` (uint32 per sample; bit0 -> port0/line0 = GP2,
bit1 -> port0/line1 = GP3) on the 6535 and synchronously captures port1/line0:1 (bit0 = GP4 result
line, bit1 = GP5 spare) on the SAME sample clock: the DO OnboardClock is exported to RTSI7 and DI is
clocked from it, with DI armed before DO so DI sample 0 == DO edge 0. Returns the captured uint32
array (same length N). Used for both the decoder loopback self-test and the front-end stimulus/read.
"""
import numpy as np
import nidaqmx
from nidaqmx.constants import LineGrouping, AcquisitionType, Edge
from nidaqmx.stream_writers import DigitalSingleChannelWriter
from nidaqmx.stream_readers import DigitalSingleChannelReader


def generate_capture(do_data, rate, device="Dev1", shared_term=None, timeout=None):
    do_data = np.asarray(do_data, dtype=np.uint32)
    N = int(do_data.shape[0])
    shared_term = shared_term or f"/{device}/RTSI7"
    if timeout is None:
        timeout = max(10.0, N / rate + 5.0)
    di_data = np.zeros(N, dtype=np.uint32)
    with nidaqmx.Task("do_gen") as do, nidaqmx.Task("di_cap") as di:
        do.do_channels.add_do_chan(f"{device}/port0/line0:1", line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
        do.timing.cfg_samp_clk_timing(rate, sample_mode=AcquisitionType.FINITE, samps_per_chan=N)
        do.export_signals.samp_clk_output_term = shared_term
        DigitalSingleChannelWriter(do.out_stream, auto_start=False).write_many_sample_port_uint32(do_data)

        di.di_channels.add_di_chan(f"{device}/port1/line0:1", line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
        di.timing.cfg_samp_clk_timing(rate, source=shared_term, active_edge=Edge.RISING,
                                      sample_mode=AcquisitionType.FINITE, samps_per_chan=N)
        di.start()
        do.start()
        DigitalSingleChannelReader(di.in_stream).read_many_sample_port_uint32(
            di_data, number_of_samples_per_channel=N, timeout=timeout)
        do.wait_until_done(timeout=timeout)
    return di_data
