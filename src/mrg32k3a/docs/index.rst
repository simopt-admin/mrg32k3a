.. mrg32k3a documentation master file, created by
   sphinx-quickstart on Mon Jul 18 15:03:20 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to mrg32k3a's documentation!
====================================

This package provides a Python implementation of the mrg32k3a pseudo-random number generator of L'Ecuyer (1999) and L'Ecuyer et al. (2002). It extends the implementation used in `PyMOSO <https://github.com/pymoso/PyMOSO#the-pymosoprngmrg32k3a-module>`_ to handle streams, substreams, *and* subsubstreams. The period of :math:`~2^{191}` is split into :math:`~2^{50}` streams of length :math:`2^{141}`, each containing :math:`2^{47}` substreams of length :math:`2^{94}`, each containing :math:`2^{47}` subsubstreams of length :math:`2^{47}`.

Details
-------
The `mrg32k3a <https://mrg32k3a.readthedocs.io/en/latest/mrg32k3a.html#module-mrg32k3a.mrg32k3a>`_ module includes the ``MRG32k3a`` class and several useful functions for controlling the generators.

* The ``MRG32k3a`` class is a subclass of Python's ``random.Random`` class and therefore inherits easy-to-use methods for generating random variates. E.g., if ``rng`` is an instance of the ``MRG32k3a`` class, the command ``rng.normalvariate(mu=2, sigma=5)`` generates a normal random variate with mean 2 and standard deviation 5.

* The ``advance_stream``, ``advance_substream``, and ``advance_subsubstream`` functions advance the generator to the start of the next stream, substream, or subsubstream, respectively. They make use of techniques for efficiently "jumping ahead," as outlined by L'Ecuyer (1990).

* The ``reset_stream``, ``reset_substream``, and ``reset_subsubstream`` functions reset the generator to the start of the current stream, substream, or subsubstream, respectively.

The `matmodops <https://mrg32k3a.readthedocs.io/en/latest/mrg32k3a.html#module-mrg32k3a.matmodops>`_ module includes basic matrix/modulus operations used by the ``mrg32k3a`` module.

References
----------
* L'Ecuyer, Pierre (1990). `"Random numbers for simulation." <https://dl.acm.org/doi/10.1145/84537.84555>`_ *Communications of the ACM* 33(10):85-97.

* L'Ecuyer, Pierre (1999). `"Good parameters and implementations for combined multiple recursive random number generators." <https://pubsonline.informs.org/doi/pdf/10.1287/opre.47.1.159>`_ *Operations Research* 47(1):159-164.

* L'Ecuyer, Pierre, Richard Simard, E Jack Chen, and W. David Kelton (2002). `"An object-oriented random number package with many long streams and substreams." <https://pubsonline.informs.org/doi/10.1287/opre.50.6.1073.358>`_ *Operations Research* 50(6):1073-1075.


.. toctree::
   :maxdepth: 2
   :caption: Contents:
