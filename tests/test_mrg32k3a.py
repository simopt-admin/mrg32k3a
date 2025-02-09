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

        # Check powers
        power_mod_1 = mrg.power_mod(mrg.A1p0, 99, mrg.mrgm1)
        power_mod_1_expected = [
            [799720564, 1904270070, 1363680284],
            [2278512092, 270284082, 1904270070],
            [2328946625, 990418809, 270284082],
        ]
        self.assertEqual(power_mod_1.tolist(), power_mod_1_expected)
        power_mod_2 = mrg.power_mod(mrg.A2p0, 99, mrg.mrgm2)
        power_mod_2_expected = [
            [3836971470, 1481396816, 1475521836],
            [722527348, 3836971470, 2878594268],
            [685711492, 722527348, 2167021383],
        ]
        self.assertEqual(
            power_mod_2.tolist(),
            power_mod_2_expected,
        )

        # Check multiplies
        multi_1 = power_mod_1 @ seed[0:3]
        multi_1_expected = [50215397482710, 54973102782180, 44314223275020]
        self.assertEqual(multi_1.tolist(), multi_1_expected)
        multi_2 = power_mod_2 @ seed[3:6]
        multi_2_expected = [83870573556090, 91823259146670, 44136587452935]
        self.assertEqual(multi_2.tolist(), multi_2_expected)

        # Check sequences
        st1 = multi_1 % mrg.mrgm1
        st1_expected = [2937268593, 1819035667, 3047838441]
        self.assertEqual(st1.tolist(), st1_expected)
        st2 = multi_2 % mrg.mrgm2
        st2_expected = [3193417629, 1641899773, 1738356667]
        self.assertEqual(st2.tolist(), st2_expected)
        sequence = st1.tolist() + st2.tolist()
        self.assertSequenceEqual(rng._current_state, sequence)

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


if __name__ == "__main__":
    unittest.main()
