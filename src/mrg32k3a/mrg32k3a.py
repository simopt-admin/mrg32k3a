import os

from .python import *

if os.environ.get("MRG32K3A_BACKEND") == "rust":
    from .rust import *
