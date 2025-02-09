#!/usr/bin/env python

import os
import sys
import unittest

import mrg32k3a.matmodops as mrg_mmo

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
Aneg = [[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]]
b = [1, 2, 3]
bneg = [-1, -2, -3]
m = 3

mrgm1 = 4294967087  # 2**32 - 209
mrgm2 = 4294944443  # 2**32 - 22853
mrga12 = 1403580  # 209*67
mrga13n = -810728  # 209*19*17
mrga21 = 527612  # 209*73
mrga23n = -1370589  # 209*19*67

A1p0 = [[0, 1, 0], [0, 0, 1], [mrga13n, mrga12, 0]]
A2p0 = [[0, 1, 0], [0, 0, 1], [mrga23n, 0, mrga21]]

A1p127 = [
    [2427906178, 3580155704, 949770784],
    [226153695, 1230515664, 3580155704],
    [1988835001, 986791581, 1230515664],
]
A2p127 = [
    [1464411153, 277697599, 1610723613],
    [32183930, 1464411153.0, 1022607788],
    [2824425944, 32183930.0, 2093834863],
]
A1p76 = [
    [82758667, 1871391091, 4127413238],
    [3672831523, 69195019, 1871391091],
    [3672091415, 3528743235, 69195019],
]
A2p76 = [
    [1511326704, 3759209742, 1610795712],
    [4292754251, 1511326704, 3889917532],
    [3859662829, 4292754251, 3708466080],
]


class TestMatModOps(unittest.TestCase):
    def test_A1p127(self):
        self.assertEqual(mrg_mmo.mat33_power_mod(A1p0, 2**127, mrgm1), A1p127)

    def test_A2p127(self):
        self.assertEqual(mrg_mmo.mat33_power_mod(A2p0, 2**127, mrgm2), A2p127)

    def test_A1p76(self):
        self.assertEqual(mrg_mmo.mat33_power_mod(A1p0, 2**76, mrgm1), A1p76)

    def test_A2p76(self):
        self.assertEqual(mrg_mmo.mat33_power_mod(A2p0, 2**76, mrgm2), A2p76)

    def test_A1p47(self):
        A1p47 = mrg_mmo.mat33_power_mod(A1p0, 2**47, mrgm1)
        self.assertEqual(mrg_mmo.mat33_power_mod(A1p0, 2**47, mrgm1), A1p47)

    def test_A2p47(self):
        A2p47 = mrg_mmo.mat33_power_mod(A2p0, 2**47, mrgm2)
        self.assertEqual(mrg_mmo.mat33_power_mod(A2p0, 2**47, mrgm2), A2p47)

    def test_A1p94(self):
        A1p94 = mrg_mmo.mat33_power_mod(A1p0, 2**94, mrgm1)
        self.assertEqual(mrg_mmo.mat33_power_mod(A1p0, 2**94, mrgm1), A1p94)

    def test_A2p94(self):
        A2p94 = mrg_mmo.mat33_power_mod(A2p0, 2**94, mrgm2)
        self.assertEqual(mrg_mmo.mat33_power_mod(A2p0, 2**94, mrgm2), A2p94)

    def test_A1p141(self):
        A1p141 = mrg_mmo.mat33_power_mod(A1p0, 2**141, mrgm1)
        self.assertEqual(mrg_mmo.mat33_power_mod(A1p0, 2**141, mrgm1), A1p141)

    def test_A2p141(self):
        A2p141 = mrg_mmo.mat33_power_mod(A2p0, 2**141, mrgm2)
        self.assertEqual(mrg_mmo.mat33_power_mod(A2p0, 2**141, mrgm2), A2p141)

    def test_mat33_mat31_mult(self):
        self.assertEqual(mrg_mmo.mat33_mat31_mult(A, b), [14, 32, 50])

    def test_mat33_mat33_mult(self):
        self.assertEqual(
            mrg_mmo.mat33_mat33_mult(A, A),
            [[30, 36, 42], [66, 81, 96], [102, 126, 150]],
        )

    def test_mat31_mod(self):
        self.assertEqual(mrg_mmo.mat31_mod(b, m), [1, 2, 0])

    def test_mat31_mod_neg(self):
        self.assertEqual(mrg_mmo.mat31_mod(bneg, m), [2, 1, 0])

    def test_mat33_mod(self):
        self.assertEqual(mrg_mmo.mat33_mod(A, m), [[1, 2, 0], [1, 2, 0], [1, 2, 0]])

    def test_mat33_mod_neg(self):
        self.assertEqual(mrg_mmo.mat33_mod(Aneg, m), [[2, 1, 0], [2, 1, 0], [2, 1, 0]])

    def test_mat33_mat33_mod(self):
        self.assertEqual(
            mrg_mmo.mat33_mat33_mod(A, A, m), [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        )

    def test_mat33_power_mod_power0(self):
        self.assertEqual(
            mrg_mmo.mat33_power_mod(A, 0, m), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )

    def test_mat33_power_mod_power1(self):
        self.assertEqual(
            mrg_mmo.mat33_power_mod(A, 1, m), [[1, 2, 0], [1, 2, 0], [1, 2, 0]]
        )

    def test_mat33_power_mod_power2(self):
        self.assertEqual(
            mrg_mmo.mat33_power_mod(A, 2, m), [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        )


if __name__ == "__main__":
    unittest.main()
