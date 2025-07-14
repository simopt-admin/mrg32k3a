import math
import random
from typing import List, Union

import mrg32k3a_core
import numpy as np
from mrg32k3a_core import bsm


def _neg_log_log(x: float) -> float:
    return math.log(-math.log(x))


class MRG32k3a(random.Random):
    def __init__(
        self,
        seed=(12345, 12345, 12345, 12345, 12345, 12345),
        s_ss_sss_index=None,
    ) -> None:
        self.rng = mrg32k3a_core.Mrg32k3a(seed, s_ss_sss_index)

    @property
    def _current_state(self):
        return self.rng._current_state

    @_current_state.setter
    def _current_state(self, value):
        self.rng.set_state(value)

    def get_current_state(self):
        return self._current_state

    def random(self):
        return self.rng.random()

    def getstate(self):
        return self.rng.get_current_state()

    def setstate(self, state):
        self.rng.set_state(state)

    @property
    def stream_start(self):
        return self.rng.stream_start

    @property
    def substream_start(self):
        return self.rng.substream_start

    @property
    def subsubstream_start(self):
        return self.rng.subsubstream_start

    @property
    def s_ss_sss_index(self):
        return self.rng.s_ss_sss_index

    def advance_stream(self) -> None:
        self.rng.advance_stream()

    def advance_substream(self) -> None:
        self.rng.advance_substream()

    def advance_subsubstream(self) -> None:
        self.rng.advance_subsubstream()

    def reset_stream(self) -> None:
        self.rng.reset_stream()

    def reset_substream(self) -> None:
        self.rng.reset_substream()

    def reset_subsubstream(self) -> None:
        self.rng.reset_subsubstream()

    def start_fixed_s_ss_sss(self, s_ss_sss_triplet: List[int]) -> None:
        self.rng.start_fixed_s_ss_sss(s_ss_sss_triplet)

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
