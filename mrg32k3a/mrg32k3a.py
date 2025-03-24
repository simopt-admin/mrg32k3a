#!/usr/bin/env python
"""Provide a subclass of ``random.Random`` using mrg32k3a as the generator with stream/substream/subsubstream support."""

# Code largely adopted from PyMOSO repository (https://github.com/pymoso/PyMOSO).

import math
import random
from copy import deepcopy
import sys
from typing import Any, List, Optional, Tuple, Union

import numpy as np

# Type hint for the seed parameter.
if sys.version_info >= (3, 9):
    SeedType = Union[
        int, float, str, bytes, bytearray, Tuple[int, int, int, int, int, int], None
    ]
else:
    SeedType = Union[Any]

# Constants used in mrg32k3a and in substream generation.
# P. L'Ecuyer, ``Good Parameter Sets for Combined Multiple Recursive Random Number Generators'',
# Operations Research, 47, 1 (1999), 159--164.
# P. L'Ecuyer, R. Simard, E. J. Chen, and W. D. Kelton,
# ``An Objected-Oriented Random-Number Package with Many Long Streams and Substreams'',
# Operations Research, 50, 6 (2002), 1073--1075.

# Page 162, Table II
# J = 2, K = 3
mrgm1 = 4294967087
mrgm1_plus_1 = 4294967088  # mrgm1 + 1
mrgm1_div_mrgm1_plus_1 = 0.999999999767169  # mrgm1 / (mrgm1 + 1)
mrgm2 = 4294944443
mrga12 = 1403580
mrga13n = -810728
mrga21 = 527612
mrga23n = -1370589

# These need to be object-types to avoid overflow errors
# (Python's int type has arbitrary precision)
A1p47 = np.array(
    [
        [1362557480, 3230022138, 4278720212],
        [3427386258, 3848976950, 3230022138],
        [2109817045, 2441486578, 3848976950],
    ],
    dtype=object,
)
A2p47 = np.array(
    [
        [2920112852, 1965329198, 1177141043],
        [2135250851, 2920112852, 969184056],
        [296035385, 2135250851, 4267827987],
    ],
    dtype=object,
)
A1p94 = np.array(
    [
        [2873769531, 2081104178, 596284397],
        [4153800443, 1261269623, 2081104178],
        [3967600061, 1830023157, 1261269623],
    ],
    dtype=object,
)
A2p94 = np.array(
    [
        [1347291439, 2050427676, 736113023],
        [4102191254, 1347291439, 878627148],
        [1293500383, 4102191254, 745646810],
    ],
    dtype=object,
)
A1p141 = np.array(
    [
        [3230096243, 2131723358, 3262178024],
        [2882890127, 4088518247, 2131723358],
        [3991553306, 1282224087, 4088518247],
    ],
    dtype=object,
)
A2p141 = np.array(
    [
        [2196438580, 805386227, 4266375092],
        [4124675351, 2196438580, 2527961345],
        [94452540, 4124675351, 2825656399],
    ],
    dtype=object,
)

# Constants used in Beasley-Springer-Moro algorithm for approximating
# the inverse cdf of the standard normal distribution.
bsma = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
bsmb = [1, -8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
bsmc = [
    0.3374754822726147,
    0.9761690190917186,
    0.1607979714918209,
    0.0276438810333863,
    0.0038405729373609,
    0.0003951896511919,
    0.0000321767881768,
    0.0000002888167364,
    0.0000003960315187,
]


def _neg_log_log(x: float) -> float:
    return math.log(-math.log(x))


# Adapted to pure Python from the P. L'Ecuyer code referenced above.
def mrg32k3a(
    state: Tuple[int, int, int, int, int, int],
) -> Tuple[Tuple[int, int, int, int, int, int], float]:
    """Generate a random number between 0 and 1 from a given state.

    Parameters
    ----------
    state : Tuple [int, int, int, int, int, int]
        Current state of the generator.

    Returns
    -------
    Tuple [int, int, int, int, int, int]
        Next state of the generator.
    float
        Pseudo uniform random variate.

    """
    # Create the new state
    s0, s1, s2, s3, s4, s5 = state

    # Compute new values for the state
    new_s2 = (mrga12 * s1 + mrga13n * s0) % mrgm1
    new_s5 = (mrga21 * s5 + mrga23n * s3) % mrgm2

    # Calculate uniform random variate.
    diff = new_s2 - new_s5
    u = (diff % mrgm1) / mrgm1_plus_1 if diff != 0 else mrgm1_div_mrgm1_plus_1

    return (s1, s2, new_s2, s4, s5, new_s5), u


def bsm(u: float) -> float:
    """Approximate a quantile of the standard normal distribution via the Beasley-Springer-Moro algorithm.

    Parameters
    ----------
    u : float
        Probability value for the desired quantile (between 0 and 1).

    Returns
    -------
    float
        Corresponding quantile of the standard normal distribution.

    """

    def horner(x: float, coeffs: List[float]) -> float:
        result = 0.0
        for c in reversed(coeffs):
            result = result * x + c
        return result

    if not 0 < u < 1:
        raise ValueError("Argument must be in (0, 1).")
    y = u - 0.5
    # Approximate from the center (Beasly-Springer 1977).
    if abs(y) < 0.42:
        r = y * y
        return y * horner(r, bsma) / horner(r, bsmb)
    # Approximate from the tails (Moro 1995).
    if y < 0:
        r = u
        sign = -1
    else:
        r = 1 - u
        sign = 1
    s = _neg_log_log(r)
    return sign * horner(s, bsmc)


class MRG32k3a(random.Random):
    """Implements mrg32k3a as the generator for a ``random.Random`` object.

    Attributes
    ----------
    _current_state : tuple [int]
        Current state of mrg32k3a generator.
    ref_seed : tuple [int]
        Seed from which to start the generator.
        Streams/substreams/subsubstreams are referenced w.r.t. ``ref_seed``.
    s_ss_sss_index : List [int]
        Triplet of the indices of the current stream-substream-subsubstream.
    stream_start : List [int]
        State corresponding to the start of the current stream.
    substream_start: List [int]
        State corresponding to the start of the current substream.
    subsubstream_start: List [int]
        State corresponding to the start of the current subsubstream.

    See Also
    --------
    random.Random

    """

    def __init__(
        self,
        ref_seed: Tuple[int, int, int, int, int, int] = (
            12345,
            12345,
            12345,
            12345,
            12345,
            12345,
        ),
        s_ss_sss_index: Optional[List[int]] = None,
    ) -> None:
        """Initialize the MRG32k3a generator.

        Parameters
        ----------
        ref_seed : tuple [int, int, int, int, int, int], optional
            Seed from which to start the generator.
        s_ss_sss_index : List [int], optional
            Triplet of the indices of the stream-substream-subsubstream to start at.

        """
        self.version = 2
        self.generate = mrg32k3a
        self.ref_seed = ref_seed
        self._current_state = ref_seed
        self.gauss_next = None
        if s_ss_sss_index is None:
            s_ss_sss_index = [0, 0, 0]
        self.start_fixed_s_ss_sss(s_ss_sss_index)

    def __deepcopy__(self, memo: dict) -> "MRG32k3a":
        """Deepcopy the generator.

        Parameters
        ----------
        memo : dict
            Dictionary to store the copied objects.

        Returns
        -------
        MRG32k3a
            Deepcopy of the generator.

        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def seed(self, a: SeedType = None, version: int = 2) -> None:
        """Set the state (or seed) of the generator and update the generator state.

        Parameters
        ----------
        new_state : tuple [int]
            New state to which to advance the generator.

        """
        if isinstance(a, tuple):
            if len(a) != 6:
                raise ValueError("Seed must be a tuple of length 6.")
            if any(x < 0 for x in a):
                raise ValueError("Seed values must be non-negative.")
            self._current_state = a
        else:
            super().seed(a, version)

    def getstate(
        self,
    ) -> Tuple[Tuple[int, int, int, int, int, int], Tuple]:
        """Return the state of the generator.

        Returns
        -------
        tuple [int, int, int, int, int, int]
            Current state of the generator, ``_current_state``.
        tuple
            Ouptput of ``random.Random.getstate()``.

        See Also
        --------
        random.Random

        """
        return self.get_current_state(), super().getstate()

    def setstate(
        self,
        state: Tuple[
            Tuple[int, int, int, int, int, int],
            Tuple,
        ],
    ) -> None:
        """Set the internal state of the generator.

        Parameters
        ----------
        state : Tuple[Tuple[int, int, int, int, int, int], Tuple]
            ``state[0]`` is new state for the generator.
            ``state[1]`` is ``random.Random.getstate()``.

        See Also
        --------
        random.Random

        """
        self._current_state = state[0]
        super().setstate(state[1])

    def random(self) -> float:
        """Generate a standard uniform variate and advance the generator state.

        Returns
        -------
        float
            Pseudo uniform random variate.

        """
        self._current_state, u = self.generate(self._current_state)
        return float(u)

    def get_current_state(self) -> Tuple[int, int, int, int, int, int]:
        """Return the current state of the generator.

        Returns
        -------
        Tuple [int, int, int, int, int, int]
            Current state of the generator.

        """
        return self._current_state

    def normalvariate(self, mu: float = 0, sigma: float = 1) -> float:
        """Generate a normal random variate.

        Parameters
        ----------
        mu : float, optional
            Expected value of the normal distribution from which to
            generate.
        sigma : float, optional
            Standard deviation of the normal distribution from which to
            generate.

        Returns
        -------
        float
            A normal random variate from the specified distribution.

        """
        return mu + sigma * bsm(self.random())

    def lognormalvariate(self, lq: float, uq: float) -> float:
        """Generate a Lognormal random variate using 2.5% and 97.5% quantiles.

        Parameters
        ----------
        lq : float
            2.5% quantile of the lognormal distribution from which to
            generate.
        uq : float
            97.5% quantile of the lognormal distribution from which to
            generate.

        Returns
        -------
        float
            A lognormal random variate from the specified distribution.

        """
        if lq <= 0 or uq <= 0:
            raise ValueError("Quantiles must be greater than 0.")
        log_uq = math.log(uq)
        mu = (math.log(lq) + log_uq) / 2
        return math.exp(self.normalvariate(mu, (log_uq - mu) / 1.96))

    def mvnormalvariate(
        self,
        mean_vec: List[float],
        cov: Union[List[List[float]], np.ndarray],
        factorized: bool = False,
    ) -> List[float]:
        """Generate a normal random vector.

        Parameters
        ----------
        mean_vec : List [float]
            Location parameters of the multivariate normal distribution
            from which to generate.
        cov : List [List [float]]
            Covariance matrix of the multivariate normal distribution
            from which to generate.
        factorized : bool, default=False
            True if we do not need to calculate Cholesky decomposition,
            i.e., if Cholesky decomposition is given as ``cov``;
            False otherwise.

        Returns
        -------
        List [float]
            Multivariate normal random variate from the specified distribution.

        """
        if not isinstance(cov, np.ndarray):
            cov = np.array(cov)
        chol = np.linalg.cholesky(cov) if not factorized else cov
        observations = [self.normalvariate(0, 1) for _ in range(len(cov))]
        result_array = np.dot(chol, observations).transpose() + mean_vec
        return result_array.tolist()

    def poissonvariate(self, lmbda: float) -> int:
        """Generate a Poisson random variate.

        Parameters
        ----------
        lmbda : float
            Expected value of the Poisson distribution from which to
            generate.

        Returns
        -------
        float
            Poisson random variate from the specified distribution.

        """
        if lmbda >= 35:
            return max(
                math.ceil(lmbda + math.sqrt(lmbda) * self.normalvariate() - 0.5),
                0,
            )
        n = 0
        p = self.random()
        threshold = math.exp(-lmbda)
        while p >= threshold:
            p *= self.random()
            n += 1
        return n

    def gumbelvariate(self, mu: float, beta: float) -> float:
        """Generate a Gumbel random variate.

        Parameters
        ----------
        mu : float
            Location of the mode of the Gumbel distribution from which to
            generate.
        beta : float
            Scale parameter of the Gumbel distribution from which to
            generate; > 0.

        Returns
        -------
        float
            Gumbel random variate from the specified distribution.

        """
        return mu - beta * _neg_log_log(self.random())

    def binomialvariate(self, n: int = 1, p: float = 0.5) -> int:
        """Generate a Binomial(n, p) random variate.

        Parameters
        ----------
        n : int
            Number of i.i.d. Bernoulli trials; > 0.
        p : float
            Success probability of i.i.d. Bernoulli trials; in (0, 1).

        Returns
        -------
        int
            Binomial random variate from the specified distribution.

        """
        return sum(self.random() < p for _ in range(n))

    def integer_random_vector_from_simplex(
        self, n_elements: int, summation: int, with_zero: bool = False
    ) -> List[int]:
        """Generate a random vector with a specified number of non-negative integer elements that sum up to a specified number.

        Parameters
        ----------
        n_elements : float
            Number of elements in the requested vector.
        summation : int
            Number to which the integer elements of the vector must sum.
        with_zero: bool, optional
            True if zeros in the vector are permitted; False otherwise.

        Returns
        -------
        List [int]
            A non-negative integer vector of length n_elements that sum to n_elements.

        """
        if not with_zero and n_elements > summation:
            error_msg = "The sum cannot be greater than the number of positive integers requested."
            raise ValueError(error_msg)

        # Adjust the sum to account for the possibility of zeros.
        shift = 0 if not with_zero else n_elements
        adjusted_sum = summation + shift

        # Generate n_elements - 1 random integers in the range [1, adjusted_sum - 1].
        temp_x = sorted(self.sample(range(1, adjusted_sum), k=n_elements - 1))
        cut_points = [0, *temp_x, adjusted_sum]
        offset = int(with_zero)
        vec = [(cut_points[i + 1] - cut_points[i]) - offset for i in range(n_elements)]

        return vec

    def continuous_random_vector_from_simplex(
        self, n_elements: int, summation: float, exact_sum: bool = False
    ) -> List[float]:
        """Generate a random vector with a specified number of non-negative real-valued elements that sum up to (or less than or equal to) a specified number.

        Parameters
        ----------
        n_elements : float
            Number of elements in the requested vector.
        summation : float, optional
            Number to which the elements of the vector must sum.
        exact_sum : bool, optional
            True if the sum should be equal to summation;
            False if the sum should be less than or equal to summation.

        Returns
        -------
        List [float]
            Vector of ``n_elements`` non-negative real-valued numbers that
            sum up to (or less than or equal to) ``summation``.

        """
        if exact_sum:
            # Generate a vector of length n_elements of i.i.d. Exponential(1)
            # random variates. Normalize all values by the sum and multiply by
            # "summation".
            exp_rvs = np.array([self.expovariate(lambd=1) for _ in range(n_elements)])
            result_array = summation * exp_rvs / np.sum(exp_rvs)
            return result_array.tolist()
        # Follows Theorem 2.1 of "Non-Uniform Random Variate Generation" by DeVroye.
        # Chapter 11, page 568.
        # Generate a vector of length n_elements of i.i.d. Uniform(0, 1)
        # random variates. Sort it in ascending order, pre-append
        # "0", and post-append "summation".
        unif_rvs = np.sort(np.array([self.random() for _ in range(n_elements)]))
        diffs = np.diff(np.concatenate(([0], unif_rvs, [1])))
        # Construct a matrix of the vertices of the simplex in R^d in regular position.
        # Includes zero vector and d unit vectors in R^d.
        vertices = np.vstack((np.zeros(n_elements), np.eye(n_elements)))
        # Multiply each vertex by the corresponding term in diffs.
        # Then multiply each component by "summation" and sum the vectors
        # to get the convex combination of the vertices (scaled up to "summation").
        result_array = np.dot(summation * diffs, vertices)
        return result_array.tolist()

    @staticmethod
    def _advance_state(a: np.ndarray, b: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Efficiently advance state -> A*s % m for both state parts.

        Parameters
        ----------
        a : np.ndarray
            Matrix to multiply the state by.
        b : np.ndarray
            Matrix to multiply the state by.
        state : np.ndarray
            State to be advanced.

        Returns
        -------
        np.ndarray
            Advanced state.

        """
        new_state_a = np.dot(a, state[:3]) % mrgm1
        new_state_b = np.dot(b, state[3:]) % mrgm2
        return np.hstack((new_state_a, new_state_b))

    def advance_stream(self) -> None:
        """Advance the state of the generator to the start of the next stream.

        Streams are of length 2**141.
        """
        nstate = self._advance_state(A1p141, A2p141, self.stream_start)
        # Update state
        self._current_state = tuple(nstate)
        self.stream_start = nstate
        self.substream_start = nstate
        self.subsubstream_start = nstate
        # Increment the stream index.
        self.s_ss_sss_index[0] += 1
        # Reset index for substream and subsubstream.
        self.s_ss_sss_index[1] = 0
        self.s_ss_sss_index[2] = 0

    def advance_substream(self) -> None:
        """Advance the state of the generator to the start of the next substream.

        Substreams are of length 2**94.
        """
        nstate = self._advance_state(A1p94, A2p94, self.substream_start)
        # Update state
        self._current_state = tuple(nstate)
        self.substream_start = nstate
        self.subsubstream_start = nstate
        # Increment the substream index.
        self.s_ss_sss_index[1] += 1
        # Reset index for subsubstream.
        self.s_ss_sss_index[2] = 0

    def advance_subsubstream(self) -> None:
        """Advance the state of the generator to the start of the next subsubstream.

        Subsubstreams are of length 2**47.
        """
        nstate = self._advance_state(A1p47, A2p47, self.subsubstream_start)
        # Update state
        self._current_state = tuple(nstate)
        self.subsubstream_start = nstate
        # Increment the subsubstream index.
        self.s_ss_sss_index[2] += 1

    def reset_stream(self) -> None:
        """Reset the state of the generator to the start of the current stream."""
        nstate = self.stream_start
        # Update state
        self._current_state = tuple(nstate)
        self.substream_start = nstate
        self.subsubstream_start = nstate
        # Reset index for substream and subsubstream.
        self.s_ss_sss_index[1] = 0
        self.s_ss_sss_index[2] = 0

    def reset_substream(self) -> None:
        """Reset the state of the generator to the start of the current substream."""
        nstate = self.substream_start
        self._current_state = tuple(nstate)
        # Update state referencing.
        self.subsubstream_start = nstate
        # Reset index for subsubstream.
        self.s_ss_sss_index[2] = 0

    def reset_subsubstream(self) -> None:
        """Reset the state of the generator to the start of the current subsubstream."""
        self._current_state = tuple(self.subsubstream_start)

    def start_fixed_s_ss_sss(self, s_ss_sss_triplet: List[int]) -> None:
        """Set the rng to the start of a specified (stream, substream, subsubstream) triplet.

        Parameters
        ----------
        s_ss_sss_triplet : List [int]
            Triplet of the indices of the current stream-substream-subsubstream.
        """

        def power_mod(a: np.ndarray, j: int, m: float) -> np.ndarray:
            """Compute moduli of a 3 x 3 matrix power.

            Use divide-and-conquer algorithm described in L'Ecuyer (1990).

            Parameters
            ----------
            a : np.ndarray
                3 x 3 matrix.
            j : int
                Exponent.
            m : float
                Modulus.

            Returns
            -------
            np.ndarray
                3 x 3 matrix.
            """
            b = np.eye(3, dtype=object)
            while j > 0:
                if j & 1:
                    b = np.dot(a, b) % m
                a = np.dot(a, a) % m
                j //= 2
            return b

        def progress_state(
            a1: np.ndarray, a2: np.ndarray, stream: int, state: np.ndarray
        ) -> np.ndarray:
            """Efficiently advance state -> A*s % m for both state parts.

            Parameters
            ----------
            a1 : np.ndarray
                First matrix to multiply the state by.
            a2 : np.ndarray
                Second matrix to multiply the state by.
            stream : int
                Stream index to which to advance.
            state : np.ndarray
                State to be advanced.
            """
            a1pm = power_mod(a1, stream, mrgm1)
            a2pm = power_mod(a2, stream, mrgm2)
            return self._advance_state(a1pm, a2pm, state)

        # Grab the stream, substream, and subsubstream indices.
        stream, substream, subsubstream = s_ss_sss_triplet
        # Advance to start of specified stream.
        self.stream_start = progress_state(
            A1p141, A2p141, stream, np.array(self.ref_seed)
        )
        self.substream_start = progress_state(
            A1p94, A2p94, substream, self.stream_start
        )
        self.subsubstream_start = progress_state(
            A1p47, A2p47, subsubstream, self.substream_start
        )
        # Update state.
        self._current_state = tuple(self.subsubstream_start)
        # Update index referencing.
        self.s_ss_sss_index = s_ss_sss_triplet
