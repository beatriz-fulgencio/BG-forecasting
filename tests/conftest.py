"""Shared pytest configuration: put the repo root and RUN/ on sys.path.

Lets tests import both the ``benchmark`` package and the ``RUN/`` analysis
scripts (which are run as top-level scripts, not an installed package).
"""

import os
import sys
from pathlib import Path

# Cap BLAS/OpenMP threads before any test imports numpy/scipy (conftest is loaded
# by pytest before test modules). Mirrors the runner's own guard and avoids the
# multi-threaded-BLAS import deadlock seen when scipy + statsmodels load together.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "RUN"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
