#!/usr/bin/env python
"""Useful matrix/modulus operations for mrg32k3a generator."""


def mat33_mat31_mult(a: list[list[float]], b: list[float]) -> list[float]:
    """Multiply a 3 x 3 matrix with a 3 x 1 matrix.

    Parameters
    ----------
    a : list [list [float]]
        3 x 3 matrix.
    b : list [float]
        3 x 1 matrix.

    Returns
    -------
    list [float]
        3 x 1 matrix.

    """
    res = [0, 0, 0]
    r3 = range(3)
    for i in r3:
        res[i] = sum([a[i][j] * b[j] for j in r3])
    return res


def mat33_mat33_mult(
    a: list[list[float]], b: list[list[float]]
) -> list[list[float]]:
    """Multiply a 3 x 3 matrix with a 3 x 3 matrix.

    Parameters
    ----------
    a : list [list [float]]
        3 x 3 matrix.
    b : list [list [float]]
        3 x 3 matrix.

    Returns
    -------
    list [float]
        3 x 3 matrix.

    """
    res = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    r3 = range(3)
    for i in r3:
        for j in r3:
            res[i][j] = sum([a[i][k] * b[k][j] for k in r3])
    return res


def mat31_mod(b: list[float], m: float) -> list[float]:
    """Compute moduli of a 3 x 1 matrix.

    Parameters
    ----------
    b : list [float]
        3 x 1 matrix.
    m : float
        Modulus.

    Returns
    -------
    list [float]
        3 x 1 matrix.

    """
    res = [0, 0, 0]
    for i in range(3):
        res[i] = int(b[i] - int(b[i] / m) * m)
        # if negative, add back modulus m
        if res[i] < 0:
            res[i] += m
    return res


def mat33_mod(a: list[float], m: float) -> list[float]:
    """Compute moduli of a 3 x 3 matrix.

    Parameters
    ----------
    a : list [float]
        3 x 3 matrix.
    m : float
        Modulus.

    Returns
    -------
    list [float]
        3 x 3 matrix.

    """
    res = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    r3 = range(3)
    for i in r3:
        for j in r3:
            res[i][j] = int(a[i][j] - int(a[i][j] / m) * m)
            # if negative, add back modulus m
            if res[i][j] < 0:
                res[i][j] += m
    return res


def mat33_mat33_mod(
    a: list[list[float]], b: list[list[float]], m: float
) -> list[list[float]]:
    """Compute moduli of a 3 x 3 matrix x 3 x 3 matrix product.

    Parameters
    ----------
    a : list [list [float]]
        3 x 3 matrix.
    b : list [list [float]]
        3 x 3 matrix.
    m : float
        Modulus.

    Returns
    -------
    list [list [float]]
        3 x 3 matrix.

    """
    c = mat33_mat33_mult(a, b)
    res = mat33_mod(c, m)
    return res


def mat33_power_mod(
    a: list[list[float]], j: int, m: float
) -> list[list[float]]:
    """Compute moduli of a 3 x 3 matrix power.

    Use divide-and-conquer algorithm described in L'Ecuyer (1990).

    Parameters
    ----------
    a : list [list [float]]
        3 x 3 matrix.
    j : int
        Exponent.
    m : float
        Modulus.

    Returns
    -------
    res : list [list [float]]
        3 x 3 matrix.

    """
    b = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    while j > 0:
        if j % 2 == 1:
            b = mat33_mat33_mod(a, b, m)
        a = mat33_mat33_mod(a, a, m)
        j = int(j / 2)
    res = b
    return res
