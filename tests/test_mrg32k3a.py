#!/usr/bin/env python

import os
import sys
import unittest

import numpy as np

import mrg32k3a.mrg32k3a as mrg

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

seed = (12345, 12345, 12345, 12345, 12345, 12345)


class TestRandom(unittest.TestCase):
    def __get_nth_random(self, n) -> float:
        rng = mrg.MRG32k3a()
        for _ in range(n - 1):
            rng.random()
        return rng.random()

    def __assert_almost_equal(self, a, b):
        a = round(a, 15)
        b = round(b, 15)
        self.assertEqual(a, b)

    def test_random_1(self):
        rand_val = self.__get_nth_random(1)
        self.__assert_almost_equal(rand_val, 0.127011122046577)

    def test_random_2(self):
        rand_val = self.__get_nth_random(2)
        self.assertEqual(rand_val, 0.3185275653967945)

    def test_random_3(self):
        rand_val = self.__get_nth_random(3)
        self.assertEqual(rand_val, 0.3091860155832701)

    def test_random_10(self):
        rand_val = self.__get_nth_random(10)
        self.assertEqual(rand_val, 0.7558522371615435)

    def test_random_100(self):
        rand_val = self.__get_nth_random(100)
        self.assertEqual(rand_val, 0.7592386016439714)

    def test_random_1000(self):
        rand_val = self.__get_nth_random(1000)
        self.assertEqual(rand_val, 0.9860784868021322)

    def test_random_10000(self):
        rand_val = self.__get_nth_random(10000)
        self.assertEqual(rand_val, 0.2044975435211065)

    def test_random_100000(self):
        rand_val = self.__get_nth_random(100000)
        self.assertEqual(rand_val, 0.6962891099574359)

    def test_random_1000000(self):
        rand_val = self.__get_nth_random(1000000)
        self.assertEqual(rand_val, 0.37578835621568796)

    def test_uniform_distribution(self):
        rng = mrg.MRG32k3a()
        rand_vals = [rng.random() for _ in range(1000000)]
        # Calculate mean and std of random values
        mean = np.mean(rand_vals)
        std = np.std(rand_vals)
        # Make sure mean and std approximate a uniform distribution
        self.assertTrue(np.isclose(mean, 0.5, rtol=1e-3))
        self.assertTrue(np.isclose(std, 1 / np.sqrt(12), rtol=1e-3))


class TestStates(unittest.TestCase):
    A1p0 = np.array([[0, 1, 0], [0, 0, 1], [mrg.mrga13n, mrg.mrga12, 0]])
    A2p0 = np.array([[0, 1, 0], [0, 0, 1], [mrg.mrga23n, 0, mrg.mrga21]])

    def test_get_current_state(self):
        rng = mrg.MRG32k3a()
        self.assertEqual(rng.get_current_state(), seed)

    def test_first_state(self):
        rng = mrg.MRG32k3a()
        self.assertEqual(rng._current_state, seed)

    def test_second_state(self):
        rng = mrg.MRG32k3a()
        rng.random()
        st1_mult = self.A1p0 @ seed[0:3]
        st1 = st1_mult % mrg.mrgm1
        st2_mult = self.A2p0 @ seed[3:6]
        st2 = st2_mult % mrg.mrgm2
        state = st1.tolist() + st2.tolist()
        self.assertSequenceEqual(rng._current_state, state)

    def test_third_state(self):
        rng = mrg.MRG32k3a()
        rng.random()
        rng.random()
        A1sq = self.A1p0 @ self.A1p0
        A2sq = self.A2p0 @ self.A2p0
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


class TestStreams(unittest.TestCase):
    def test_advance_stream_1(self):
        rng = mrg.MRG32k3a(s_ss_sss_index=[0, 1, 1])
        rng.advance_stream()
        rng2 = mrg.MRG32k3a(s_ss_sss_index=[1, 0, 0])
        self.assertEqual(rng._current_state, rng2._current_state)
        self.assertEqual(rng.stream_start, rng._current_state)
        self.assertEqual(rng.substream_start, rng._current_state)
        self.assertEqual(rng.subsubstream_start, rng._current_state)
        self.assertEqual(rng.s_ss_sss_index, [1, 0, 0])

    def test_advance_stream_10(self):
        rng = mrg.MRG32k3a(s_ss_sss_index=[0, 1, 1])
        for _ in range(10):
            rng.advance_stream()
        rng2 = mrg.MRG32k3a(s_ss_sss_index=[10, 0, 0])
        self.assertEqual(rng._current_state, rng2._current_state)
        self.assertEqual(rng.stream_start, rng._current_state)
        self.assertEqual(rng.substream_start, rng._current_state)
        self.assertEqual(rng.subsubstream_start, rng._current_state)
        self.assertEqual(rng.s_ss_sss_index, [10, 0, 0])

    def test_advance_substream_1(self):
        rng = mrg.MRG32k3a(s_ss_sss_index=[0, 0, 1])
        rng.advance_substream()
        rng2 = mrg.MRG32k3a(s_ss_sss_index=[0, 1, 0])
        self.assertEqual(rng._current_state, rng2._current_state)
        self.assertEqual(rng.stream_start, seed)
        self.assertEqual(rng.substream_start, rng._current_state)
        self.assertEqual(rng.subsubstream_start, rng._current_state)
        self.assertEqual(rng.s_ss_sss_index, [0, 1, 0])

    def test_advance_substream_10(self):
        rng = mrg.MRG32k3a(s_ss_sss_index=[0, 0, 1])
        for _ in range(10):
            rng.advance_substream()
        rng2 = mrg.MRG32k3a(s_ss_sss_index=[0, 10, 0])
        self.assertEqual(rng._current_state, rng2._current_state)
        self.assertEqual(rng.stream_start, seed)
        self.assertEqual(rng.substream_start, rng._current_state)
        self.assertEqual(rng.subsubstream_start, rng._current_state)
        self.assertEqual(rng.s_ss_sss_index, [0, 10, 0])

    def test_advance_subsubstream_1(self):
        rng = mrg.MRG32k3a()
        rng.advance_subsubstream()
        rng2 = mrg.MRG32k3a(s_ss_sss_index=[0, 0, 1])
        self.assertEqual(rng._current_state, rng2._current_state)
        self.assertEqual(rng.stream_start, seed)
        self.assertEqual(rng.substream_start, seed)
        self.assertEqual(rng.subsubstream_start, rng._current_state)
        self.assertEqual(rng.s_ss_sss_index, [0, 0, 1])

    def test_advance_subsubstream_10(self):
        rng = mrg.MRG32k3a()
        for _ in range(10):
            rng.advance_subsubstream()
        rng2 = mrg.MRG32k3a(s_ss_sss_index=[0, 0, 10])
        self.assertEqual(rng._current_state, rng2._current_state)
        self.assertEqual(rng.stream_start, seed)
        self.assertEqual(rng.substream_start, seed)
        self.assertEqual(rng.subsubstream_start, rng._current_state)
        self.assertEqual(rng.s_ss_sss_index, [0, 0, 10])

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


class TestBSM(unittest.TestCase):
    def test_bsm_ab_diff_00(self):
        result = mrg.bsm(0.5)
        self.assertEqual(result, 0.0)

    def test_bsm_ab_diff_01(self):
        result_low = mrg.bsm(0.4)
        result_high = mrg.bsm(0.6)
        expected = 0.2533471033214436
        self.assertAlmostEqual(result_low, -expected)
        self.assertAlmostEqual(result_high, expected)
        self.assertAlmostEqual(result_low, -result_high)

    def test_bsm_ab_diff_02(self):
        result_low = mrg.bsm(0.3)
        result_high = mrg.bsm(0.7)
        expected = 0.5244005119066527
        self.assertAlmostEqual(result_low, -expected)
        self.assertAlmostEqual(result_high, expected)
        self.assertAlmostEqual(result_low, -result_high)

    def test_bsm_ab_diff_03(self):
        result_low = mrg.bsm(0.2)
        result_high = mrg.bsm(0.8)
        expected = 0.8416212348979941
        self.assertAlmostEqual(result_low, -expected)
        self.assertAlmostEqual(result_high, expected)
        self.assertAlmostEqual(result_low, -result_high)

    def test_bsm_ab_diff_04(self):
        result_low = mrg.bsm(0.1)
        result_high = mrg.bsm(0.9)
        expected = 1.2815515632770351
        self.assertAlmostEqual(result_low, -expected)
        self.assertAlmostEqual(result_high, expected)
        self.assertAlmostEqual(result_low, -result_high)

    def test_bsm_c(self):
        self.assertAlmostEqual(mrg.bsm(0.92), 1.4050715603096322)
        self.assertAlmostEqual(mrg.bsm(0.94), 1.5547735946074814)
        self.assertAlmostEqual(mrg.bsm(0.96), 1.7506860713064076)
        self.assertAlmostEqual(mrg.bsm(0.98), 2.0537489105686255)


if __name__ == "__main__":
    unittest.main()
