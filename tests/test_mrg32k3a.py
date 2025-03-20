#!/usr/bin/env python

import os
import sys
import unittest

import numpy as np

import mrg32k3a.mrg32k3a as mrg

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


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
    seed = (12345, 12345, 12345, 12345, 12345, 12345)

    A1p0 = np.array([[0, 1, 0], [0, 0, 1], [mrg.mrga13n, mrg.mrga12, 0]])
    A2p0 = np.array([[0, 1, 0], [0, 0, 1], [mrg.mrga23n, 0, mrg.mrga21]])

    def test_get_current_state(self):
        rng = mrg.MRG32k3a()
        self.assertEqual(rng.get_current_state(), self.seed)

    def test_first_state(self):
        rng = mrg.MRG32k3a()
        self.assertEqual(rng._current_state, self.seed)

    def test_second_state(self):
        rng = mrg.MRG32k3a()
        rng.random()
        st1_mult = self.A1p0 @ self.seed[0:3]
        st1 = st1_mult % mrg.mrgm1
        st2_mult = self.A2p0 @ self.seed[3:6]
        st2 = st2_mult % mrg.mrgm2
        state = np.hstack((st1, st2)).tolist()
        self.assertSequenceEqual(rng._current_state, state)

    def test_third_state(self):
        rng = mrg.MRG32k3a()
        rng.random()
        rng.random()
        A1sq = self.A1p0 @ self.A1p0
        A2sq = self.A2p0 @ self.A2p0
        st1_mult = A1sq @ self.seed[0:3]
        st1 = st1_mult % mrg.mrgm1
        st2_mult = A2sq @ self.seed[3:6]
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
    seed = (12345, 12345, 12345, 12345, 12345, 12345)

    next_rand = {
        (0, 0, 0): 0.12701112204657714,
        (0, 0, 1): 0.1981528990938801,
        (0, 0, 10): 0.14034946942531729,
        (0, 1, 0): 0.07661060219048645,
        (0, 1, 1): 0.6410743580999473,
        (0, 10, 0): 0.1041233145300414,
        (1, 0, 0): 0.35183402690605203,
        (1, 1, 0): 0.9128064636289478,
        (1, 1, 1): 0.036327319116350815,
        (10, 0, 0): 0.8579536770597018,
    }

    def __check_next_rand(self, rng: mrg.MRG32k3a, index: list[int]):
        random = rng.random()
        self.assertEqual(random, self.next_rand[tuple(index)])

    def test_advance_stream_1(self):
        stream_idx = [0, 1, 1]
        rng = mrg.MRG32k3a(s_ss_sss_index=stream_idx)
        self.__check_next_rand(rng, stream_idx)

        rng.advance_stream()
        stream_idx2 = [1, 0, 0]
        rng2 = mrg.MRG32k3a(s_ss_sss_index=stream_idx2)
        self.assertEqual(rng._current_state, rng2._current_state)
        self.assertTrue(np.array_equal(rng.stream_start, rng2.stream_start))
        self.assertTrue(np.array_equal(rng.substream_start, rng2.substream_start))
        self.assertTrue(np.array_equal(rng.subsubstream_start, rng2.subsubstream_start))
        self.assertTrue(np.array_equal(rng.stream_start, rng._current_state))
        self.assertTrue(np.array_equal(rng.substream_start, rng._current_state))
        self.assertTrue(np.array_equal(rng.subsubstream_start, rng._current_state))

        self.assertEqual(rng.s_ss_sss_index, stream_idx2)
        self.__check_next_rand(rng, stream_idx2)

    def test_advance_stream_10(self):
        stream_idx = [0, 1, 1]
        rng = mrg.MRG32k3a(s_ss_sss_index=stream_idx)
        self.__check_next_rand(rng, stream_idx)

        for _ in range(10):
            rng.advance_stream()
        stream_idx2 = [10, 0, 0]
        rng2 = mrg.MRG32k3a(s_ss_sss_index=stream_idx2)
        self.assertEqual(rng._current_state, rng2._current_state)
        self.assertTrue(np.array_equal(rng.stream_start, rng._current_state))
        self.assertTrue(np.array_equal(rng.substream_start, rng._current_state))
        self.assertTrue(np.array_equal(rng.subsubstream_start, rng._current_state))
        self.assertTrue(np.array_equal(rng.stream_start, rng2.stream_start))
        self.assertTrue(np.array_equal(rng.substream_start, rng2.substream_start))
        self.assertTrue(np.array_equal(rng.subsubstream_start, rng2.subsubstream_start))

        self.assertEqual(rng.s_ss_sss_index, stream_idx2)
        self.__check_next_rand(rng, stream_idx2)

    def test_advance_substream_1(self):
        stream_idx = [0, 0, 1]
        rng = mrg.MRG32k3a(s_ss_sss_index=stream_idx)
        self.__check_next_rand(rng, stream_idx)

        rng.advance_substream()
        stream_idx2 = [0, 1, 0]
        rng2 = mrg.MRG32k3a(s_ss_sss_index=stream_idx2)
        self.assertEqual(rng._current_state, rng2._current_state)
        self.assertTrue(np.array_equal(rng.stream_start, self.seed))
        self.assertTrue(np.array_equal(rng.substream_start, rng._current_state))
        self.assertTrue(np.array_equal(rng.subsubstream_start, rng._current_state))

        self.assertEqual(rng.s_ss_sss_index, stream_idx2)
        self.__check_next_rand(rng, stream_idx2)

    def test_advance_substream_10(self):
        stream_idx = [0, 0, 1]
        rng = mrg.MRG32k3a(s_ss_sss_index=stream_idx)
        self.__check_next_rand(rng, stream_idx)

        for _ in range(10):
            rng.advance_substream()
        stream_idx2 = [0, 10, 0]
        rng2 = mrg.MRG32k3a(s_ss_sss_index=stream_idx2)
        self.assertEqual(rng._current_state, rng2._current_state)
        self.assertTrue(np.array_equal(rng.stream_start, self.seed))
        self.assertTrue(np.array_equal(rng.substream_start, rng._current_state))
        self.assertTrue(np.array_equal(rng.subsubstream_start, rng._current_state))

        self.assertEqual(rng.s_ss_sss_index, stream_idx2)
        self.__check_next_rand(rng, stream_idx2)

    def test_advance_subsubstream_1(self):
        stream_idx = [0, 0, 0]
        rng = mrg.MRG32k3a()
        self.__check_next_rand(rng, stream_idx)

        rng.advance_subsubstream()
        stream_idx2 = [0, 0, 1]
        rng2 = mrg.MRG32k3a(s_ss_sss_index=stream_idx2)
        self.assertEqual(rng._current_state, rng2._current_state)
        self.assertTrue(np.array_equal(rng.stream_start, self.seed))
        self.assertTrue(np.array_equal(rng.substream_start, self.seed))
        self.assertTrue(np.array_equal(rng.subsubstream_start, rng._current_state))

        self.assertEqual(rng.s_ss_sss_index, stream_idx2)
        self.__check_next_rand(rng, stream_idx2)

    def test_advance_subsubstream_10(self):
        stream_idx = [0, 0, 0]
        rng = mrg.MRG32k3a(s_ss_sss_index=stream_idx)
        self.__check_next_rand(rng, stream_idx)

        for _ in range(10):
            rng.advance_subsubstream()
        stream_idx2 = [0, 0, 10]
        rng2 = mrg.MRG32k3a(s_ss_sss_index=stream_idx2)
        self.assertEqual(rng._current_state, rng2._current_state)
        self.assertTrue(np.array_equal(rng.stream_start, self.seed))
        self.assertTrue(np.array_equal(rng.substream_start, self.seed))
        self.assertTrue(np.array_equal(rng.subsubstream_start, rng._current_state))

        self.assertEqual(rng.s_ss_sss_index, stream_idx2)
        self.__check_next_rand(rng, stream_idx2)

    def test_reset_stream(self):
        stream_idx = [1, 1, 1]
        rng = mrg.MRG32k3a(s_ss_sss_index=stream_idx)
        self.__check_next_rand(rng, stream_idx)

        rng.reset_stream()
        self.assertTrue(np.array_equal(rng.stream_start, rng._current_state))
        self.assertTrue(np.array_equal(rng.substream_start, rng._current_state))
        self.assertTrue(np.array_equal(rng.subsubstream_start, rng._current_state))

        stream_idx2 = [1, 0, 0]
        self.assertEqual(rng.s_ss_sss_index, stream_idx2)
        rng2 = mrg.MRG32k3a(s_ss_sss_index=stream_idx2)
        self.assertEqual(rng._current_state, rng2._current_state)
        self.__check_next_rand(rng2, stream_idx2)

        rng3 = mrg.MRG32k3a()
        rng3.advance_stream()
        self.assertTrue(np.array_equal(rng.stream_start, rng3._current_state))
        self.__check_next_rand(rng3, stream_idx2)

        self.__check_next_rand(rng, stream_idx2)

    def test_reset_substream(self):
        stream_idx = [1, 1, 1]
        rng = mrg.MRG32k3a(s_ss_sss_index=stream_idx)
        self.__check_next_rand(rng, stream_idx)

        rng.reset_substream()
        stream_idx2 = [1, 1, 0]
        rng2 = mrg.MRG32k3a(s_ss_sss_index=stream_idx2)
        self.assertEqual(rng._current_state, rng2._current_state)
        self.__check_next_rand(rng2, stream_idx2)
        self.assertEqual(rng.s_ss_sss_index, stream_idx2)

        stream_idx3 = [1, 0, 0]
        rng3 = mrg.MRG32k3a(s_ss_sss_index=stream_idx3)
        self.assertTrue(np.array_equal(rng.stream_start, rng3._current_state))
        self.assertTrue(np.array_equal(rng.substream_start, rng._current_state))
        self.assertTrue(np.array_equal(rng.subsubstream_start, rng._current_state))

    def test_reset_subsubstream(self):
        stream_idx = [1, 1, 1]
        rng = mrg.MRG32k3a(s_ss_sss_index=stream_idx)
        self.__check_next_rand(rng, stream_idx)

        rng.reset_subsubstream()
        stream_idx2 = [1, 1, 1]
        rng2 = mrg.MRG32k3a(s_ss_sss_index=stream_idx2)
        self.assertEqual(rng._current_state, rng2._current_state)
        stream_idx3 = [1, 0, 0]
        rng3 = mrg.MRG32k3a(s_ss_sss_index=stream_idx3)
        stream_idx4 = [1, 1, 0]
        rng4 = mrg.MRG32k3a(s_ss_sss_index=stream_idx4)
        self.assertTrue(np.array_equal(rng.stream_start, rng3._current_state))
        self.assertTrue(np.array_equal(rng.substream_start, rng4._current_state))
        self.assertTrue(np.array_equal(rng.subsubstream_start, rng._current_state))
        self.assertEqual(rng.s_ss_sss_index, stream_idx2)
        self.__check_next_rand(rng, stream_idx2)
        self.__check_next_rand(rng2, stream_idx2)
        self.__check_next_rand(rng3, stream_idx3)
        self.__check_next_rand(rng4, stream_idx4)

    def test_init_fixed_s_ss_sss(self):
        stream_idx = [1, 1, 1]
        rng = mrg.MRG32k3a(s_ss_sss_index=stream_idx)
        rng2 = mrg.MRG32k3a()
        rng2.start_fixed_s_ss_sss(stream_idx)
        self.assertEqual(rng._current_state, rng2._current_state)
        self.assertTrue(np.array_equal(rng.stream_start, rng2.stream_start))
        self.assertTrue(np.array_equal(rng.substream_start, rng2.substream_start))
        self.assertTrue(np.array_equal(rng.subsubstream_start, rng2.subsubstream_start))
        self.assertEqual(rng.s_ss_sss_index, rng2.s_ss_sss_index)
        self.__check_next_rand(rng, stream_idx)
        self.__check_next_rand(rng2, stream_idx)

    def test_jump_fixed_s_ss_sss(self):
        stream_idx = [1, 1, 1]
        rng = mrg.MRG32k3a()
        rng.start_fixed_s_ss_sss(stream_idx)
        rng2 = mrg.MRG32k3a()
        rng2.advance_stream()
        self.__check_next_rand(rng2, [1, 0, 0])
        rng2.advance_substream()
        self.__check_next_rand(rng2, [1, 1, 0])
        rng2.advance_subsubstream()
        self.__check_next_rand(rng2, [1, 1, 1])
        rng2.reset_subsubstream()
        self.assertEqual(rng._current_state, rng2._current_state)
        self.assertTrue(np.array_equal(rng.stream_start, rng2.stream_start))
        self.assertTrue(np.array_equal(rng.substream_start, rng2.substream_start))
        self.assertTrue(np.array_equal(rng.subsubstream_start, rng2.subsubstream_start))
        self.assertEqual(rng.s_ss_sss_index, rng2.s_ss_sss_index)
        self.__check_next_rand(rng, stream_idx)


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


class TestVariates(unittest.TestCase):
    # TODO: come up with a better way of testing non-normal distributions
    # without hardcoding values for mean and std

    def setUp(self):
        self.rng = mrg.MRG32k3a()
        # Make sure we're getting a "fresh" RNG
        # This makes sure any small errors propagate to the tests
        for _ in range(100):
            self.rng.random()

    def test_normal(self):
        results = self.rng.normalvariate()
        self.assertTrue(isinstance(results, float))
        self.assertAlmostEqual(results, -1.407282773533036)

    def test_normal_dist(self):
        results = [self.rng.normalvariate() for _ in range(200000)]
        mean = np.mean(results)
        std = np.std(results)
        self.assertAlmostEqual(mean, 0, places=2)
        self.assertAlmostEqual(std, 1, places=2)

        results = [self.rng.normalvariate(1, 2) for _ in range(200000)]
        mean = np.mean(results)
        std = np.std(results)
        self.assertAlmostEqual(mean, 1, places=2)
        self.assertAlmostEqual(std, 2, places=2)

    def test_log_normal(self):
        result = self.rng.lognormalvariate(0.1, 5)
        self.assertTrue(isinstance(result, float))
        self.assertAlmostEqual(result, 0.17360153803999934)

        try:
            result = self.rng.lognormalvariate(-0.1, 5)
        except Exception as e:
            self.assertTrue(isinstance(e, ValueError))
        else:
            self.fail("Exception not raised")

        try:
            result = self.rng.lognormalvariate(0.1, -5)
        except Exception as e:
            self.assertTrue(isinstance(e, ValueError))
        else:
            self.fail("Exception not raised")

    def test_log_normal_dist(self):
        results = [self.rng.lognormalvariate(0.1, 5) for _ in range(500000)]
        mean = np.mean(results)
        std = np.std(results)
        self.assertAlmostEqual(mean, 1.1605134466960465, places=2)
        self.assertAlmostEqual(std, 1.5117748500697967, places=2)

    def test_mv_normal(self):
        averages = [0, 1, 2, 3]
        matrix = [
            [1, 0.5, 0.3, 0.2],
            [0.5, 1, 0.7, 0.4],
            [0.3, 0.7, 1, 0.6],
            [0.2, 0.4, 0.6, 1],
        ]
        result = self.rng.mvnormalvariate(mean_vec=averages, cov=matrix)
        self.assertTrue(isinstance(result, list))
        self.assertEqual(len(result), 4)
        self.assertTrue(all(isinstance(x, float) for x in result))
        self.assertAlmostEqual(result[0], -1.407282773533036)
        self.assertAlmostEqual(result[1], 0.3837072871268238)
        self.assertAlmostEqual(result[2], 3.3127997256814443)
        self.assertAlmostEqual(result[3], 4.086612958572276)

    def test_mv_normal_dist(self):
        averages = [0, 1, 2, 3]
        matrix = [
            [1, 0.5, 0.3, 0.2],
            [0.5, 1, 0.7, 0.4],
            [0.3, 0.7, 1, 0.6],
            [0.2, 0.4, 0.6, 1],
        ]
        results = [
            self.rng.mvnormalvariate(mean_vec=averages, cov=matrix)
            for _ in range(150000)
        ]
        means = np.mean(results, axis=0)
        stds = np.std(results, axis=0)
        self.assertAlmostEqual(means[0], 0, places=2)
        self.assertAlmostEqual(means[1], 1, places=2)
        self.assertAlmostEqual(means[2], 2, places=2)
        self.assertAlmostEqual(means[3], 3, places=2)
        self.assertAlmostEqual(stds[0], 1, places=2)
        self.assertAlmostEqual(stds[1], 1, places=2)
        self.assertAlmostEqual(stds[2], 1, places=2)
        self.assertAlmostEqual(stds[3], 1, places=2)

    def test_poisson(self):
        rng = mrg.MRG32k3a()
        result = rng.poissonvariate(-10)
        self.assertTrue(isinstance(result, int))
        self.assertEqual(result, 0)

        result = rng.poissonvariate(1)
        self.assertTrue(isinstance(result, int))
        self.assertEqual(result, 0)

        result = rng.poissonvariate(10)
        self.assertTrue(isinstance(result, int))
        self.assertEqual(result, 10)

        result = rng.poissonvariate(34)
        self.assertTrue(isinstance(result, int))
        self.assertEqual(result, 29)

        result = rng.poissonvariate(35)
        self.assertTrue(isinstance(result, int))
        self.assertEqual(result, 44)

        result = rng.poissonvariate(100)
        self.assertTrue(isinstance(result, int))
        self.assertEqual(result, 97)

    def test_poission_dist(self):
        rng = mrg.MRG32k3a()
        results = [rng.poissonvariate(10) for _ in range(100000)]
        mean = np.mean(results)
        std = np.std(results)
        self.assertAlmostEqual(mean, 10, places=2)
        self.assertAlmostEqual(std, 3.16, places=1)

        result = [rng.poissonvariate(100) for _ in range(100000)]
        mean = np.mean(result)
        std = np.std(result)
        self.assertAlmostEqual(mean, 100, delta=0.1)
        self.assertAlmostEqual(std, 10, places=1)

    def test_gumbel(self):
        rng = mrg.MRG32k3a()
        result = rng.gumbelvariate(0, 1)
        self.assertTrue(isinstance(result, float))
        self.assertAlmostEqual(result, -0.7243941789321531)

    def test_gumbel_dist(self):
        rng = mrg.MRG32k3a()
        results = [rng.gumbelvariate(2, 1) for _ in range(100000)]
        mean = np.mean(results)
        std = np.std(results)
        self.assertAlmostEqual(mean, 2.577, places=2)
        self.assertAlmostEqual(std, 1.28, places=2)

    def test_binomial(self):
        rng = mrg.MRG32k3a()
        result = rng.binomialvariate(1, 0.5)
        self.assertTrue(isinstance(result, int))
        self.assertEqual(result, 0)

        result = rng.binomialvariate(2, 0)
        self.assertTrue(isinstance(result, int))
        self.assertEqual(result, 0)

        result = rng.binomialvariate(3, 1)
        self.assertTrue(isinstance(result, int))
        self.assertEqual(result, 3)

    def test_binomial_dist(self):
        rng = mrg.MRG32k3a()
        results = [rng.binomialvariate(10, 0.5) for _ in range(100000)]
        mean = np.mean(results)
        std = np.std(results)
        self.assertAlmostEqual(mean, 5, delta=0.01)
        self.assertAlmostEqual(std, 1.58, delta=0.01)

        results = [rng.binomialvariate(100, 0.3) for _ in range(100000)]
        mean = np.mean(results)
        std = np.std(results)
        self.assertAlmostEqual(mean, 30, delta=0.03)
        self.assertAlmostEqual(std, 4.58, delta=0.03)


class TestVectors(unittest.TestCase):
    def test_int_random_vector(self):
        rng = mrg.MRG32k3a()
        result = rng.integer_random_vector_from_simplex(3, 10)
        self.assertTrue(isinstance(result, list))
        self.assertEqual(len(result), 3)
        self.assertEqual(sum(result), 10)
        self.assertTrue(all(x > 0 for x in result))
        self.assertTrue(all(isinstance(x, int) for x in result))
        self.assertEqual(result, [7, 2, 1])

        results = [rng.integer_random_vector_from_simplex(4, 9) for _ in range(100)]
        self.assertTrue(all(len(x) == 4 for x in results))
        self.assertTrue(all(sum(x) == 9 for x in results))
        self.assertTrue(all(x > 0 for x in result))

        try:
            rng.integer_random_vector_from_simplex(10, 5)
        except Exception as e:
            self.assertTrue(isinstance(e, ValueError))
        else:
            self.fail("Exception not raised")

        results = [
            rng.integer_random_vector_from_simplex(2, 1, True) for _ in range(100)
        ]
        self.assertTrue(all(len(x) == 2 for x in results))
        self.assertTrue(all(sum(x) == 1 for x in results))
        count_0 = sum([1 for x in results if x[0] == 0 or x[1] == 0])
        count_1 = sum([1 for x in results if x[0] == 1 or x[1] == 1])
        self.assertTrue(count_0 == 100)
        self.assertTrue(count_1 == 100)

    def test_cont_random_vector(self):
        rng = mrg.MRG32k3a()
        result = rng.continuous_random_vector_from_simplex(3, 10)
        self.assertTrue(isinstance(result, list))
        self.assertEqual(len(result), 3)
        self.assertLessEqual(sum(result), 10)
        self.assertTrue(all(isinstance(x, float) for x in result))
        self.assertAlmostEqual(result[0], 1.8217489353669294, places=2)
        self.assertAlmostEqual(result[1], 0.09341549813524419, places=2)
        self.assertAlmostEqual(result[2], 6.814724346032055, places=2)

        results = [rng.continuous_random_vector_from_simplex(4, 9) for _ in range(100)]
        self.assertTrue(all(len(x) == 4 for x in results))
        self.assertTrue(all(sum(x) < 9 for x in results))
        self.assertTrue(all(x > 0 for x in result))

        results = [
            rng.continuous_random_vector_from_simplex(2, 1, True) for _ in range(100)
        ]
        self.assertTrue(all(len(x) == 2 for x in results))
        for result in results:
            self.assertAlmostEqual(sum(result), 1, places=14)
        self.assertTrue(all(x > 0 for x in result))


class TestAxpxx(unittest.TestCase):
    # Old method of calculating A1p47, A2p47, A1p94, A2p94, A1p141, A2p141
    # No longer needed since we precomputed and stored them in mrg32k3a.py,
    # but kept here to verify the results are the same
    @staticmethod
    def _power_mod(a: np.ndarray, j: int, m: float) -> np.ndarray:
        b = np.eye(3, dtype=np.int64)
        while j > 0:
            if j & 1:
                b = (a @ b) % m
            a = (a @ a) % m
            j //= 2
        return b

    # These need to be object-types to avoid overflow errors
    # (Python's int type has arbitrary precision)
    A1p0 = np.array([[0, 1, 0], [0, 0, 1], [mrg.mrga13n, mrg.mrga12, 0]], dtype=object)
    A2p0 = np.array([[0, 1, 0], [0, 0, 1], [mrg.mrga23n, 0, mrg.mrga21]], dtype=object)

    def test_a1p47(self):
        a1p47 = self._power_mod(self.A1p0, 2**47, mrg.mrgm1)
        self.assertTrue(np.array_equal(a1p47, mrg.A1p47))

    def test_a2p47(self):
        a2p47 = self._power_mod(self.A2p0, 2**47, mrg.mrgm2)
        self.assertTrue(np.array_equal(a2p47, mrg.A2p47))

    def test_a1p94(self):
        a1p94 = self._power_mod(self.A1p0, 2**94, mrg.mrgm1)
        self.assertTrue(np.array_equal(a1p94, mrg.A1p94))

    def test_a2p94(self):
        a2p94 = self._power_mod(self.A2p0, 2**94, mrg.mrgm2)
        self.assertTrue(np.array_equal(a2p94, mrg.A2p94))

    def test_a1p141(self):
        a1p141 = self._power_mod(self.A1p0, 2**141, mrg.mrgm1)
        self.assertTrue(np.array_equal(a1p141, mrg.A1p141))

    def test_a2p141(self):
        a2p141 = self._power_mod(self.A2p0, 2**141, mrg.mrgm2)
        self.assertTrue(np.array_equal(a2p141, mrg.A2p141))


if __name__ == "__main__":
    unittest.main()
