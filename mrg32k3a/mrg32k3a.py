#!/usr/bin/env python
"""Provide a subclass of ``random.Random`` using mrg32k3a as the generator with stream/substream/subsubstream support."""

# Code largely adopted from PyMOSO repository (https://github.com/pymoso/PyMOSO).

from __future__ import annotations

import random
from copy import deepcopy
from math import ceil, exp, log, sqrt

import numpy as np
from numpy.linalg import matrix_power
from numpy.polynomial.polynomial import polyval

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

A1p0 = np.array([[0, 1, 0], [0, 0, 1], [mrga13n, mrga12, 0]])
A2p0 = np.array([[0, 1, 0], [0, 0, 1], [mrga23n, 0, mrga21]])

# These need to be object-types to avoid overflow errors
# (Python's int type has arbitrary precision)
A1p47 = np.array(
    [
        [2150882049, 1012615007, 1753411989],
        [234971272, 3938477338, 966612171],
        [3006247121, 3687673689, 940826602],
    ],
    dtype=object,
)
A2p47 = np.array(
    [
        [1046183310, 837768296, 3615496901],
        [17393928, 3278539534, 2641844075],
        [1993644961, 3704895436, 3750989706],
    ],
    dtype=object,
)
A1p94 = np.array(
    [
        [755664978, 569748550, 3548871349],
        [1218252973, 522684588, 810477570],
        [1389789784, 3146723777, 3218949703],
    ],
    dtype=object,
)
A2p94 = np.array(
    [
        [1562155877, 1430955988, 3645089813],
        [3893748262, 3622354192, 1033072313],
        [369450389, 3504376307, 2126264688],
    ],
    dtype=object,
)
A1p141 = np.array(
    [
        [766528512, 1921679764, 2446008495],
        [2261886462, 1413988183, 1120803221],
        [3269079875, 1181992446, 144371898],
    ],
    dtype=object,
)
A2p141 = np.array(
    [
        [2180513949, 1961145626, 3911964994],
        [963935459, 2169350115, 2047463392],
        [2520335674, 2435164196, 3566463752],
    ],
    dtype=object,
)

# Constants used in Beasley-Springer-Moro algorithm for approximating
# the inverse cdf of the standard normal distribution.
bsma = np.array(
    [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637],
)
bsmb = np.array(
    [1, -8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833],
)
bsmc = np.array(
    [
        0.3374754822726147,
        0.9761690190917186,
        0.1607979714918209,
        0.0276438810333863,
        0.0038405729373609,
        0.0003951896511919,
        0.0000321767881768,
        0.0000002888167364,
        0.0000003960315187,
    ],
)


# Adapted to pure Python from the P. L'Ecuyer code referenced above.
def mrg32k3a(
    state: tuple[int, int, int, int, int, int],
) -> tuple[tuple[int, int, int, int, int, int], float]:
    """Generate a random number between 0 and 1 from a given state.

    Parameters
    ----------
    state : tuple [int, int, int, int, int, int]
        Current state of the generator.

    Returns
    -------
    tuple [int, int, int, int, int, int]
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
    if u <= 0 or u >= 1:
        raise ValueError("Argument must be in (0, 1).")
    y = u - 0.5
    if abs(y) < 0.42:
        # Approximate from the center (Beasly-Springer 1977).
        r = y * y
        return y * (polyval(r, bsma) / polyval(r, bsmb))
    else:
        # Approximate from the tails (Moro 1995).
        signum = -1 if y < 0 else 1
        r = u if y < 0 else 1 - u
        return signum * polyval(log(-log(r)), bsmc)


class MRG32k3a(random.Random):
    """Implements mrg32k3a as the generator for a ``random.Random`` object.

    Attributes
    ----------
    _current_state : tuple [int]
        Current state of mrg32k3a generator.
    ref_seed : tuple [int]
        Seed from which to start the generator.
        Streams/substreams/subsubstreams are referenced w.r.t. ``ref_seed``.
    s_ss_sss_index : list [int]
        Triplet of the indices of the current stream-substream-subsubstream.
    stream_start : list [int]
        State corresponding to the start of the current stream.
    substream_start: list [int]
        State corresponding to the start of the current substream.
    subsubstream_start: list [int]
        State corresponding to the start of the current subsubstream.

    See Also
    --------
    random.Random

    """

    def __init__(
        self,
        ref_seed: tuple[int, int, int, int, int, int] = (
            12345,
            12345,
            12345,
            12345,
            12345,
            12345,
        ),
        s_ss_sss_index: list[int] | None = None,
    ) -> None:
        """Initialize the MRG32k3a generator.

        Parameters
        ----------
        ref_seed : tuple [int, int, int, int, int, int], optional
            Seed from which to start the generator.
        s_ss_sss_index : list [int], optional
            Triplet of the indices of the stream-substream-subsubstream to start at.

        """
        self.version = 2
        self.generate = mrg32k3a
        self.ref_seed = ref_seed
        self.seed(ref_seed)
        self.gauss_next = None
        if s_ss_sss_index is None:
            s_ss_sss_index = [0, 0, 0]
        self.start_fixed_s_ss_sss(s_ss_sss_index)

    def __deepcopy__(self, memo: dict) -> MRG32k3a:
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

    def seed(self, new_state: tuple[int, int, int, int, int, int]) -> None:
        """Set the state (or seed) of the generator and update the generator state.

        Parameters
        ----------
        new_state : tuple [int]
            New state to which to advance the generator.

        """
        self._current_state = new_state
        # super().seed(new_state)

    def getstate(
        self,
    ) -> tuple[tuple[int, int, int, int, int, int], tuple]:
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
        state: tuple[
            tuple[int, int, int, int, int, int],
            tuple,
        ],
    ) -> None:
        """Set the internal state of the generator.

        Parameters
        ----------
        state : tuple[tuple[int, int, int, int, int, int], tuple]
            ``state[0]`` is new state for the generator.
            ``state[1]`` is ``random.Random.getstate()``.

        See Also
        --------
        random.Random

        """
        self.seed(state[0])
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

    def get_current_state(self) -> tuple[int, int, int, int, int, int]:
        """Return the current state of the generator.

        Returns
        -------
        tuple [int, int, int, int, int, int]
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
        mu = (log(lq) + log(uq)) / 2
        return exp(self.normalvariate(mu, (log(uq) - mu) / 1.96))

    def mvnormalvariate(
        self,
        mean_vec: list[float],
        cov: np.ndarray,
        factorized: bool = False,
    ) -> list[float]:
        """Generate a normal random vector.

        Parameters
        ----------
        mean_vec : list [float]
            Location parameters of the multivariate normal distribution
            from which to generate.
        cov : list [list [float]]
            Covariance matrix of the multivariate normal distribution
            from which to generate.
        factorized : bool, default=False
            True if we do not need to calculate Cholesky decomposition,
            i.e., if Cholesky decomposition is given as ``cov``;
            False otherwise.

        Returns
        -------
        list [float]
            Multivariate normal random variate from the specified distribution.

        """
        n_cols = len(cov)
        chol = np.linalg.cholesky(cov) if not factorized else cov
        observations = [self.normalvariate(0, 1) for _ in range(n_cols)]
        return chol.dot(observations).transpose() + mean_vec

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
        if lmbda < 35:
            n = 0
            p = self.random()
            threshold = exp(-lmbda)
            while p >= threshold:
                u = self.random()
                p = p * u
                n = n + 1
        else:
            z = self.normalvariate()
            n = max(ceil(lmbda + sqrt(lmbda) * z - 0.5), 0)
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
        u = self.random()
        q = mu - beta * np.log(-np.log(u))
        return q

    def binomialvariate(self, n: int, p: float) -> int:
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
        x = np.sum(self.choices(population=[0, 1], weights=[1 - p, p], k=n))
        return x

    def integer_random_vector_from_simplex(
        self, n_elements: int, summation: int, with_zero: bool = False
    ) -> list[int]:
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
        list [int]
            A non-negative integer vector of length n_elements that sum to n_elements.

        """
        if with_zero is False:
            if n_elements > summation:
                error_msg = "The sum cannot be greater than the number of positive integers requested."
                raise ValueError(error_msg)
            # Generate a vector of length n_elements by sampling without replacement from
            # the set {1, 2, 3, ..., n_elements-1}. Sort it in ascending order, pre-append
            # "0", and post-append "summation".
            temp_x = self.sample(population=range(1, summation), k=n_elements - 1)
            temp_x.sort()
            x = [0, *temp_x, summation]
            # Take differences between consecutive terms. Result will sum to summation.
            vec = [x[idx + 1] - x[idx] for idx in range(n_elements)]
        else:
            temp_vec = self.integer_random_vector_from_simplex(
                summation=summation + n_elements,
                n_elements=n_elements,
                with_zero=False,
            )
            vec = [tv - 1 for tv in temp_vec]
        return vec

    def continuous_random_vector_from_simplex(
        self, n_elements: int, summation: float, exact_sum: bool = False
    ) -> list[float]:
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
        list [float]
            Vector of ``n_elements`` non-negative real-valued numbers that
            sum up to (or less than or equal to) ``summation``.

        """
        if exact_sum is True:
            # Generate a vector of length n_elements of i.i.d. Exponential(1)
            # random variates. Normalize all values by the sum and multiply by
            # "summation".
            exp_rvs = [self.expovariate(lambd=1) for _ in range(n_elements)]
            exp_sum = np.sum(exp_rvs)
            vec = [summation * x / exp_sum for x in exp_rvs]
        else:  # Sum must equal summation.
            # Follows Theorem 2.1 of "Non-Uniform Random Variate Generation" by DeVroye.
            # Chapter 11, page 568.
            # Generate a vector of length n_elements of i.i.d. Uniform(0, 1)
            # random variates. Sort it in ascending order, pre-append
            # "0", and post-append "summation".
            unif_rvs = [self.random() for _ in range(n_elements)]
            unif_rvs.sort()
            x = [0, *unif_rvs, 1]
            # Take differences between consecutive terms. Result will sum to 1.
            diffs = np.array([x[idx + 1] - x[idx] for idx in range(n_elements + 1)])
            # Construct a matrix of the vertices of the simplex in R^d in regular position.
            # Includes zero vector and d unit vectors in R^d.
            vertices = np.concatenate(
                (np.zeros((1, n_elements)), np.identity(n=n_elements)), axis=0
            )
            # Multiply each vertex by the corresponding term in diffs.
            # Then multiply each component by "summation" and sum the vectors
            # to get the convex combination of the vertices (scaled up to "summation").
            vec = list(
                summation * np.sum(np.multiply(vertices, diffs[:, np.newaxis]), axis=0)
            )
        return vec

    def advance_stream(self) -> None:
        """Advance the state of the generator to the start of the next stream.

        Streams are of length 2**141.
        """
        state = self.stream_start
        # Split the state into 2 components of length 3.
        st1 = np.array(state[0:3])
        st2 = np.array(state[3:6])
        # Efficiently advance state -> A*s % m for both state parts.
        nst1 = (A1p141 @ st1) % mrgm1
        nst2 = (A2p141 @ st2) % mrgm2
        # Combine the 2 components into a single state.
        nstate = tuple(np.hstack((nst1, nst2)))
        self.seed(nstate)
        # Increment the stream index.
        self.s_ss_sss_index[0] += 1
        # Reset index for substream and subsubstream.
        self.s_ss_sss_index[1] = 0
        self.s_ss_sss_index[2] = 0
        # Update state referencing.
        self.stream_start = nstate
        self.substream_start = nstate
        self.subsubstream_start = nstate

    def advance_substream(self) -> None:
        """Advance the state of the generator to the start of the next substream.

        Substreams are of length 2**94.
        """
        state = self.substream_start
        # Split the state into 2 components of length 3.
        st1 = np.array(state[0:3])
        st2 = np.array(state[3:6])
        # Efficiently advance state -> A*s % m for both state parts.
        nst1 = (A1p94 @ st1) % mrgm1
        nst2 = (A2p94 @ st2) % mrgm2
        # Combine the 2 components into a single state.
        nstate = tuple(np.hstack((nst1, nst2)))
        self.seed(nstate)
        # Increment the substream index.
        self.s_ss_sss_index[1] += 1
        # Reset index for subsubstream.
        self.s_ss_sss_index[2] = 0
        # Update state referencing.
        self.substream_start = nstate
        self.subsubstream_start = nstate

    def advance_subsubstream(self) -> None:
        """Advance the state of the generator to the start of the next subsubstream.

        Subsubstreams are of length 2**47.
        """
        state = self.subsubstream_start
        # Split the state into 2 components of length 3.
        st1 = np.array(state[0:3])
        st2 = np.array(state[3:6])
        # Efficiently advance state -> A*s % m for both state parts.
        nst1 = (A1p47 @ st1) % mrgm1
        nst2 = (A2p47 @ st2) % mrgm2
        # Combine the 2 components into a single state.
        nstate = tuple(np.hstack((nst1, nst2)))
        self.seed(nstate)
        # Increment the subsubstream index.
        self.s_ss_sss_index[2] += 1
        # Update state referencing.
        self.subsubstream_start = nstate

    def reset_stream(self) -> None:
        """Reset the state of the generator to the start of the current stream."""
        nstate = self.stream_start
        self.seed(nstate)
        # Update state referencing.
        self.substream_start = nstate
        self.subsubstream_start = nstate
        # Reset index for substream and subsubstream.
        self.s_ss_sss_index[1] = 0
        self.s_ss_sss_index[2] = 0

    def reset_substream(self) -> None:
        """Reset the state of the generator to the start of the current substream."""
        nstate = self.substream_start
        self.seed(nstate)
        # Update state referencing.
        self.subsubstream_start = nstate
        # Reset index for subsubstream.
        self.s_ss_sss_index[2] = 0

    def reset_subsubstream(self) -> None:
        """Reset the state of the generator to the start of the current subsubstream."""
        self.seed(self.subsubstream_start)

    def start_fixed_s_ss_sss(self, s_ss_sss_triplet: list[int]) -> None:
        """Set the rng to the start of a specified (stream, substream, subsubstream) triplet.

        Parameters
        ----------
        s_ss_sss_triplet : list [int]
            Triplet of the indices of the current stream-substream-subsubstream.

        """
        # Grab the stream, substream, and subsubstream indices.
        stream, substream, subsubstream = s_ss_sss_triplet
        # Start from the reference seed.
        state = self.ref_seed
        # Split the reference seed into 2 components of length 3.
        st1 = np.array(state[0:3])
        st2 = np.array(state[3:6])
        # Advance to start of specified stream.
        # Efficiently advance state -> A*s % m for both state parts.
        power_mod_1 = matrix_power(A1p141, stream) % mrgm1
        power_mod_2 = matrix_power(A2p141, stream) % mrgm2
        st1 = (power_mod_1 @ st1) % mrgm1
        st2 = (power_mod_2 @ st2) % mrgm2
        self.stream_start = tuple(np.hstack((st1, st2)))
        # Advance to start of specified substream.
        # Efficiently advance state -> A*s % m for both state parts.
        power_mod_1 = matrix_power(A1p94, substream) % mrgm1
        power_mod_2 = matrix_power(A2p94, substream) % mrgm2
        st1 = (power_mod_1 @ st1) % mrgm1
        st2 = (power_mod_2 @ st2) % mrgm2
        self.substream_start = tuple(np.hstack((st1, st2)))
        # Advance to start of specified subsubstream.
        # Efficiently advance state -> A*s % m for both state parts.
        power_mod_1 = matrix_power(A1p47, subsubstream) % mrgm1
        power_mod_2 = matrix_power(A2p47, subsubstream) % mrgm2
        st1 = (power_mod_1 @ st1) % mrgm1
        st2 = (power_mod_2 @ st2) % mrgm2
        self.subsubstream_start = tuple(np.hstack((st1, st2)))
        self.seed(self.subsubstream_start)
        # Update index referencing.
        self.s_ss_sss_index = s_ss_sss_triplet
