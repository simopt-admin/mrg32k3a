#!/usr/bin/env python

import os
import sys
import unittest

import numpy as np

import mrg32k3a.mrg32k3a as mrg

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

seed = (12345, 12345, 12345, 12345, 12345, 12345)


class TestMRG32k3a(unittest.TestCase):
    def test_get_current_state(self):
        rng = mrg.MRG32k3a()
        self.assertEqual(rng.get_current_state(), seed)

    def test_first_state(self):
        rng = mrg.MRG32k3a()
        self.assertEqual(rng._current_state, seed)

    def test_second_state(self):
        rng = mrg.MRG32k3a()
        rng.random()
        st1_mult = mrg.A1p0 @ seed[0:3]
        st1 = st1_mult % mrg.mrgm1
        st2_mult = mrg.A2p0 @ seed[3:6]
        st2 = st2_mult % mrg.mrgm2
        state = st1.tolist() + st2.tolist()
        self.assertSequenceEqual(rng._current_state, state)

    def test_third_state(self):
        rng = mrg.MRG32k3a()
        rng.random()
        rng.random()
        A1sq = mrg.A1p0 @ mrg.A1p0
        A2sq = mrg.A2p0 @ mrg.A2p0
        st1_mult = A1sq @ seed[0:3]
        st1 = st1_mult % mrg.mrgm1
        st2_mult = A2sq @ seed[3:6]
        st2 = st2_mult % mrg.mrgm2
        state = st1.tolist() + st2.tolist()
        self.assertSequenceEqual(rng._current_state, state)

    def test_hundreth_state(self):
        # Cycle through 99 states so we're at the 100th
        rng = mrg.MRG32k3a()
        for _ in range(99):
            rng.random()

        # Check the 100th state
        expected_state = [
            2937268593,
            1819035667,
            3047838441,
            3193417629,
            1641899773,
            1738356667,
        ]
        self.assertSequenceEqual(rng._current_state, expected_state)

    def test_advance_stream(self):
        rng = mrg.MRG32k3a(s_ss_sss_index=[0, 1, 1])
        rng.advance_stream()
        rng2 = mrg.MRG32k3a(s_ss_sss_index=[1, 0, 0])
        self.assertEqual(rng._current_state, rng2._current_state)
        self.assertEqual(rng.stream_start, rng._current_state)
        self.assertEqual(rng.substream_start, rng._current_state)
        self.assertEqual(rng.subsubstream_start, rng._current_state)
        self.assertEqual(rng.s_ss_sss_index, [1, 0, 0])

    def test_advance_substream(self):
        rng = mrg.MRG32k3a(s_ss_sss_index=[0, 0, 1])
        rng.advance_substream()
        rng2 = mrg.MRG32k3a(s_ss_sss_index=[0, 1, 0])
        self.assertEqual(rng._current_state, rng2._current_state)
        self.assertEqual(rng.stream_start, seed)
        self.assertEqual(rng.substream_start, rng._current_state)
        self.assertEqual(rng.subsubstream_start, rng._current_state)
        self.assertEqual(rng.s_ss_sss_index, [0, 1, 0])

    def test_advance_subsubstream(self):
        rng = mrg.MRG32k3a()
        rng.advance_subsubstream()
        rng2 = mrg.MRG32k3a(s_ss_sss_index=[0, 0, 1])
        self.assertEqual(rng._current_state, rng2._current_state)
        self.assertEqual(rng.stream_start, seed)
        self.assertEqual(rng.substream_start, seed)
        self.assertEqual(rng.subsubstream_start, rng._current_state)
        self.assertEqual(rng.s_ss_sss_index, [0, 0, 1])

    def test_reset_stream(self):
        rng = mrg.MRG32k3a(s_ss_sss_index=[1, 1, 1])
        rng.random()
        rng.reset_stream()
        rng2 = mrg.MRG32k3a(s_ss_sss_index=[1, 0, 0])
        self.assertEqual(rng._current_state, rng2._current_state)
        self.assertEqual(rng.stream_start, rng._current_state)
        self.assertEqual(rng.substream_start, rng._current_state)
        self.assertEqual(rng.subsubstream_start, rng._current_state)
        self.assertEqual(rng.s_ss_sss_index, [1, 0, 0])

    def test_reset_substream(self):
        rng = mrg.MRG32k3a(s_ss_sss_index=[1, 1, 1])
        rng.random()
        rng.reset_substream()
        rng2 = mrg.MRG32k3a(s_ss_sss_index=[1, 1, 0])
        self.assertEqual(rng._current_state, rng2._current_state)
        rng3 = mrg.MRG32k3a(s_ss_sss_index=[1, 0, 0])
        self.assertEqual(rng.stream_start, rng3._current_state)
        self.assertEqual(rng.substream_start, rng._current_state)
        self.assertEqual(rng.subsubstream_start, rng._current_state)
        self.assertEqual(rng.s_ss_sss_index, [1, 1, 0])

    def test_reset_subsubstream(self):
        rng = mrg.MRG32k3a(s_ss_sss_index=[1, 1, 1])
        rng.random()
        rng.reset_subsubstream()
        rng2 = mrg.MRG32k3a(s_ss_sss_index=[1, 1, 1])
        self.assertEqual(rng._current_state, rng2._current_state)
        rng3 = mrg.MRG32k3a(s_ss_sss_index=[1, 0, 0])
        rng4 = mrg.MRG32k3a(s_ss_sss_index=[1, 1, 0])
        self.assertEqual(rng.stream_start, rng3._current_state)
        self.assertEqual(rng.substream_start, rng4._current_state)
        self.assertEqual(rng.subsubstream_start, rng._current_state)
        self.assertEqual(rng.s_ss_sss_index, [1, 1, 1])

    def test_init_fixed_s_ss_sss(self):
        rng = mrg.MRG32k3a(s_ss_sss_index=[1, 1, 1])
        rng2 = mrg.MRG32k3a()
        rng2.start_fixed_s_ss_sss([1, 1, 1])
        self.assertEqual(rng._current_state, rng2._current_state)
        self.assertEqual(rng.stream_start, rng2.stream_start)
        self.assertEqual(rng.substream_start, rng2.substream_start)
        self.assertEqual(rng.subsubstream_start, rng2.subsubstream_start)
        self.assertEqual(rng.s_ss_sss_index, rng2.s_ss_sss_index)

    def test_jump_fixed_s_ss_sss(self):
        rng = mrg.MRG32k3a()
        rng.start_fixed_s_ss_sss([1, 1, 1])
        rng2 = mrg.MRG32k3a()
        rng2.advance_stream()
        rng2.advance_substream()
        rng2.advance_subsubstream()
        self.assertEqual(rng._current_state, rng2._current_state)
        self.assertEqual(rng.stream_start, rng2.stream_start)
        self.assertEqual(rng.substream_start, rng2.substream_start)
        self.assertEqual(rng.subsubstream_start, rng2.subsubstream_start)
        self.assertEqual(rng.s_ss_sss_index, rng2.s_ss_sss_index)

    def test_bsm_ab(self):
        result_low = mrg.bsm(0.1)
        self.assertAlmostEqual(result_low, np.float64(-1.2815515632770351))
        result_high = mrg.bsm(0.9)
        self.assertAlmostEqual(result_high, np.float64(1.2815515632770351))
        self.assertAlmostEqual(result_low, -result_high)

        result_low = mrg.bsm(0.2)
        self.assertAlmostEqual(result_low, np.float64(-0.8416212348979941))
        result_high = mrg.bsm(0.8)
        self.assertAlmostEqual(result_high, np.float64(0.8416212348979941))
        self.assertAlmostEqual(result_low, -result_high)

        result_low = mrg.bsm(0.3)
        self.assertAlmostEqual(result_low, np.float64(-0.5244005119066527))
        result_high = mrg.bsm(0.7)
        self.assertAlmostEqual(result_high, np.float64(0.5244005119066527))
        self.assertAlmostEqual(result_low, -result_high)

        result_low = mrg.bsm(0.4)
        self.assertAlmostEqual(result_low, np.float64(-0.2533471033214436))
        result_high = mrg.bsm(0.6)
        self.assertAlmostEqual(result_high, np.float64(0.2533471033214436))
        self.assertAlmostEqual(result_low, -result_high)

        result = mrg.bsm(0.5)
        self.assertEqual(result, np.float64(0.0))

    def test_bsm_c(self):
        result = mrg.bsm(0.92)
        self.assertAlmostEqual(result, np.float64(1.4050715603096322))

        result = mrg.bsm(0.94)
        self.assertAlmostEqual(result, np.float64(1.5547735946074814))

        result = mrg.bsm(0.96)
        self.assertAlmostEqual(result, np.float64(1.7506860713064076))

        result = mrg.bsm(0.98)
        self.assertAlmostEqual(result, np.float64(2.0537489105686255))


if __name__ == "__main__":
    unittest.main()
