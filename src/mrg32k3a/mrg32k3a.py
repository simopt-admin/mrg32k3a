import os

# Import core components from Python implementation
# Import constants and utility functions that might be used elsewhere
from .python import (
    A1p47,
    A1p94,
    A1p141,
    A2p47,
    A2p94,
    A2p141,
    MRG32k3a,
    bsm,
    mrga12,
    mrga13n,
    mrga21,
    mrga23n,
    mrgm1,
    mrgm1_div_mrgm1_plus_1,
    mrgm1_plus_1,
    mrgm2,
)

# Override with Rust implementation if requested
if os.environ.get("MRG32K3A_BACKEND") == "rust":
    from .rust import MRG32k3a, bsm

__all__ = [
    "A1p47",
    "A1p94",
    "A1p141",
    "A2p47",
    "A2p94",
    "A2p141",
    "MRG32k3a",
    "bsm",
    "mrga12",
    "mrga13n",
    "mrga21",
    "mrga23n",
    "mrgm1",
    "mrgm1_div_mrgm1_plus_1",
    "mrgm1_plus_1",
    "mrgm2",
]
