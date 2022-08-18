# mrg32k3a

This package provides a Python implementation of the mrg32k3a pseudo-random number generator of L'Ecuyer (1999) and L'Ecuyer et al. (2002). It extends the implementation used in [PyMOSO](https://github.com/pymoso/PyMOSO#the-pymosoprngmrg32k3a-module) to handle streams, substreams, *and* subsubstreams. The generator's period of ~2<sup>191</sup> is split into ~2<sup>50</sup> streams of length 2<sup>141</sup>, each containing 2<sup>47</sup> substreams of length 2<sup>94</sup>, each containing 2<sup>47</sup> subsubstreams of length 2<sup>47</sup>.

### Details
The `mrg32k3a` module includes the `MRG32k3a` class and several useful functions for controlling the generators.
* The `MRG32k3a` class is a subclass of Python's `random.Random` class and therefore inherits easy-to-use methods for generating random variates. E.g., if `rng` is an instance of the `MRG32k3a` class, the command `rng.normalvariate(mu=2, sigma=5)` generates a normal random variate with mean 2 and standard deviation 5. Normal random variates are generated via inversion using the Beasley-Springer-Moro algorithm.
* The `MRG32k3a` class expands the suite of functions for random-variate generation available in `random.Random` to include `lognormalvariate`, `mvnormalvariate`, `poissonvariate`, `gumbelvariate`, `binomialvariate`. Additionally, the methods `integer_random_vector_from_simplex` and `continuous_random_vector_from_simplex` generate discrete and continuous vectors from a symmetric non-negative simplex.
* The `advance_stream`, `advance_substream`, and `advance_subsubstream` functions advance the generator to the start of the next stream, substream, or subsubstream, respectively.
They make use of techniques for efficiently "jumping ahead," as outlined by L'Ecuyer (1990).
* The `reset_stream`, `reset_substream`, and `reset_subsubstream` functions reset the generator to the start of the current stream, substream, or subsubstream, respectively.

The `matmodops` module includes basic matrix/modulus operations used by the `mrg32k3a` module.

### Installation

The `mrg32k3a` package is available to download through the Python Packaging Index (PyPI) and can be installed from the terminal with the following command:

    python -m pip install mrg32k3a

### Basic Example

After installing `mrg32k3a`, the package's main class (`MRG32k3a`) can be imported from the Python console (or in code):

    from mrg32k3a.mrg32k3a import MRG32k3a

One can instantiate a random number generator set at a given stream, substream, and subsubstream triplet or seed. For example, the command

    rng = MRG32k3a(s_ss_sss_index=[1, 2, 3])

creates a object of the `MRG32k3a` class called `rng` that it initialized at the start of subsubstream 3 of substream 2 of stream 1. If the argument `s_ss_sss_index` is not provided, the random number generator is initialized at stream-substream-subsubstream 0-0-0. (We adopt the Python convention of indexing from 0.) Alternatively, the command

    rng = MRG32k3a(ref_seed=(12345, 12345, 12345, 12345, 12345, 12345))

initializes the random number generator at the state described by the length-6 tuple (12345, 12345, 12345, 12345, 12345, 12345). Streams, substreams, and subsubstreams are indexed using `ref_seed` as a point of reference.

After instantiating a random number generator, its methods can be invoked to generate (scalar or vector) random variates from a particular probability distribution. For example,

    x = rng.normalvariate(mu=2, sigma=5)

returns a normally distributed random variate `x` with mean 2 and standard deviation 5.

Similarly,

    x = rng.poissonvariate(lmdba=50)

returns a Poisson distributed random variate `x` with rate parameter (mean) 50.

Finally,

    v = rng.integer_random_vector_from_simplex(n_elements=3, summation=10, with_zero=False))

returns a random length-3 vector `v` of positive integers summing to 10. The vector `v` is uniformly distributed over the set of such vectors.

### Documentation
Full documentation for the `mrg32k3a` source code can be found [here](https://mrg32k3a.readthedocs.io/en/latest/).

### References
* Cooper, Kyle and Susan R. Hunter (2020). [PyMOSO: Software for multi-objective simulation optimization with R-PERLE and R-MinRLE.](https://pubsonline.informs.org/doi/10.1287/ijoc.2019.0902) *INFORMS Journal on Computing* 32(4): 1101-1108.
* L'Ecuyer, Pierre (1990). [Random numbers for simulation.](https://dl.acm.org/doi/10.1145/84537.84555) *Communications of the ACM* 33(10):85-97.
* L'Ecuyer, Pierre (1999). [Good parameters and implementations for combined multiple recursive random number generators.](https://pubsonline.informs.org/doi/pdf/10.1287/opre.47.1.159) *Operations Research* 47(1):159-164.
* L'Ecuyer, Pierre, Richard Simard, E Jack Chen, and W. David Kelton (2002). [An object-oriented random number package with many long streams and substreams.](https://pubsonline.informs.org/doi/10.1287/opre.50.6.1073.358) *Operations Research* 50(6):1073-1075.
