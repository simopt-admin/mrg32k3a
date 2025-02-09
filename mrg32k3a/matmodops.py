#!/usr/bin/env python
"""Useful matrix/modulus operations for mrg32k3a generator."""

import numpy as np


def mat33_mat31_mult(
    a: list[list[int]] | np.ndarray,
    b: list[int] | np.ndarray | tuple[int, int, int],
) -> list[int]:
    """Multiply a 3 x 3 matrix with a 3 x 1 matrix.

    Parameters
    ----------
    a : list[list[int]] | np.ndarray
        3 x 3 matrix.
    b : list[int] | np.ndarray | tuple[int, int, int]
        3 x 1 matrix.

    Returns
    -------
    list[int]
        3 x 1 matrix.

    """
    res = [0, 0, 0]
    r3 = range(3)
    for i in r3:
        res[i] = sum([a[i][j] * b[j] for j in r3])
    return res


def mat33_mat33_mult(
    a: list[list[int]] | np.ndarray,
    b: list[list[int]] | np.ndarray,
) -> list[list[int]]:
    """Multiply a 3 x 3 matrix with a 3 x 3 matrix.

    Parameters
    ----------
    a : list[list[int]] | np.ndarray
        3 x 3 matrix.
    b : list[list[int]] | np.ndarray
        3 x 3 matrix.

    Returns
    -------
    list[list[int]]
        3 x 3 matrix.

    """
    res = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    r3 = range(3)
    for i in r3:
        for j in r3:
            res[i][j] = sum([a[i][k] * b[k][j] for k in r3])
    return res


def mat31_mod(b: list[int], m: float) -> list[int]:
    """Compute moduli of a 3 x 1 matrix.

    Parameters
    ----------
    b : list[int]
        3 x 1 matrix.
    m : float
        Modulus.

    Returns
    -------
    list[int]
        3 x 1 matrix.

    """
    res = [0, 0, 0]
    for i in range(3):
        res[i] = int(b[i] % m)
    return res


def mat33_mod(a: list[list[int]], m: float) -> list[list[int]]:
    """Compute moduli of a 3 x 3 matrix.

    Parameters
    ----------
    a : list[list[int]]
        3 x 3 matrix.
    m : float
        Modulus.

    Returns
    -------
    list[list[int]]
        3 x 3 matrix.

    """
    res = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    r3 = range(3)
    for i in r3:
        for j in r3:
            res[i][j] = (a[i][j] % m) // 1
    return res


def mat33_mat33_mod(
    a: list[list[int]] | np.ndarray,
    b: list[list[int]] | np.ndarray,
    m: float,
) -> list[list[int]]:
    """Compute moduli of a 3 x 3 matrix x 3 x 3 matrix product.

    Parameters
    ----------
    a : list[list[int]] | np.ndarray
        3 x 3 matrix.
    b : list[list[int]] | np.ndarray
        3 x 3 matrix.
    m : float
        Modulus.

    Returns
    -------
    list[list[int]]
        3 x 3 matrix.

    """
    c = mat33_mat33_mult(a, b)
    res = mat33_mod(c, m)
    return res


def mat33_power_mod(
    a: list[list[int]] | np.ndarray, j: int, m: float
) -> list[list[int]]:
    """Compute moduli of a 3 x 3 matrix power.

    Use divide-and-conquer algorithm described in L'Ecuyer (1990).

    Parameters
    ----------
    a : list[list[int]] | np.ndarray
        3 x 3 matrix.
    j : int
        Exponent.
    m : float
        Modulus.

    Returns
    -------
    res : list[list[int]]
        3 x 3 matrix.

    """
    b = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    while j > 0:
        if j % 2 == 1:
            b = mat33_mat33_mod(a, b, m)
        a = mat33_mat33_mod(a, a, m)
        j = j // 2
    res = b
    return res
