"""
Temporal-irregularity features of a patient's glucose signal.

These quantify how *erratic / unpredictable* a CGM series is over time, an axis
distinct from spread (``std``) and from train->test distributional shift
(``shift_score``). Two patients can share the same variance yet differ sharply in
short-term predictability. Two complementary measures are provided:

- ``sample_entropy`` (SampEn): signal irregularity / unpredictability. Standard in
  the glucose-variability literature. Higher = more irregular.
- ``autocorr_lag1``: lag-1 autocorrelation. Temporal persistence / smoothness.
  Near 1 = smooth and persistent; near 0 = jumpy / erratic.

Both are computed from the ordered test CGM series (the same ``glucose`` column
used for ``shift_score``), NOT from windowed prediction targets.

Data note: preprocessing marks missing glucose with ``-1``. All inputs are
filtered to physiologically valid readings (see :data:`GLUCOSE_MIN` /
:data:`GLUCOSE_MAX`, reused from :mod:`distribution_shift`). Dropping invalid
readings collapses gaps for the entropy calculation (standard practice); for the
lag-1 autocorrelation we instead keep positions and use only consecutive valid
pairs, so gaps do not create false adjacencies.
"""

from typing import Dict

import numpy as np
import pandas as pd  # type: ignore

from .distribution_shift import GLUCOSE_MIN, GLUCOSE_MAX

# Cap the series length used for SampEn. The template-matching is O(N^2); with
# ~2500 CGM samples per patient this keeps each patient well under a second while
# preserving the signal's irregularity structure. Uses the most recent samples.
_SAMPEN_MAX_SAMPLES = 2000


def _valid_mask(values: np.ndarray) -> np.ndarray:
    """Boolean mask of physiologically valid readings (drops the ``-1`` sentinel)."""
    return (~np.isnan(values)) & (values >= GLUCOSE_MIN) & (values <= GLUCOSE_MAX)


def lag1_autocorrelation(glucose) -> float:
    """
    Lag-1 autocorrelation of a glucose series.

    Uses only consecutive positions where BOTH ``t`` and ``t+1`` are valid, so
    the ``-1`` missing-value sentinel (and out-of-range readings) do not create
    false adjacencies across gaps.

    Returns:
        Pearson correlation between successive valid readings, or ``np.nan`` if
        there are fewer than two usable pairs (or zero variance).
    """
    values = np.asarray(glucose, dtype=float)
    if values.size < 2:
        return float("nan")

    valid = _valid_mask(values)
    # Pairs where both endpoints are valid.
    pair = valid[:-1] & valid[1:]
    x = values[:-1][pair]
    y = values[1:][pair]

    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def sample_entropy(glucose, m: int = 2, r: float = None) -> float:
    """
    Sample entropy (SampEn) of a glucose series.

    SampEn measures the negative log conditional probability that two subseries
    similar for ``m`` points remain similar at the next point. Higher = more
    irregular / less predictable.

    Args:
        glucose: 1D array-like of glucose values (mg/dL). Invalid readings
            (``-1`` sentinel, out of range, NaN) are dropped; this collapses gaps
            (standard practice for SampEn).
        m: Embedding dimension (template length). Default 2.
        r: Tolerance for matches. Defaults to ``0.2 * std`` of the cleaned series.

    Returns:
        The SampEn value, or ``np.nan`` if the series is too short or no matches
        occur at length ``m`` (undefined ratio).
    """
    values = np.asarray(glucose, dtype=float)
    values = values[_valid_mask(values)]

    # Use the most recent samples to bound the O(N^2) template matching.
    if values.size > _SAMPEN_MAX_SAMPLES:
        values = values[-_SAMPEN_MAX_SAMPLES:]

    n = values.size
    if n <= m + 1:
        return float("nan")

    if r is None:
        sd = np.std(values)
        if sd == 0:
            # Perfectly constant signal -> maximally regular.
            return 0.0
        r = 0.2 * sd

    def _count_matches(template_len: int) -> int:
        """Count template pairs (i < j) within Chebyshev distance r."""
        # Build all templates of the given length as rows.
        num = n - template_len + 1
        templates = np.array([values[i:i + template_len] for i in range(num)])
        count = 0
        # Compare each template against all later templates (i < j), vectorized
        # over j to keep this to a single Python loop over i.
        for i in range(num - 1):
            dist = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
            count += int(np.sum(dist <= r))
        return count

    b = _count_matches(m)      # matches of length m
    a = _count_matches(m + 1)  # matches of length m + 1

    if b == 0 or a == 0:
        # Undefined (log of 0 or division by 0); not enough recurrence.
        return float("nan")

    return float(-np.log(a / b))


def compute_signal_features(glucose, m: int = 2, r: float = None) -> Dict:
    """
    Compute the temporal-irregularity feature set for one glucose series.

    Returns:
        Dict with ``sample_entropy`` and ``autocorr_lag1`` (each may be
        ``np.nan`` when undefined for the series).
    """
    return {
        "sample_entropy": sample_entropy(glucose, m=m, r=r),
        "autocorr_lag1": lag1_autocorrelation(glucose),
    }


def compute_patient_signal_features(patient_data: Dict) -> pd.DataFrame:
    """
    Compute per-patient temporal-irregularity features from the test glucose.

    Mirrors :func:`distribution_shift.compute_patient_shift_scores`.

    Args:
        patient_data: Mapping ``{patient_id: {'train': df, 'test': df}}`` where
            each DataFrame has a ``'glucose'`` column (the structure returned by
            ``PopulationAnalysis.load_and_preprocess_glucose_data``). Features are
            computed from the ``test`` series.

    Returns:
        DataFrame with columns ``patient_id, sample_entropy, autocorr_lag1``,
        sorted by ``patient_id``.
    """
    rows = []
    for patient_id in sorted(patient_data.keys()):
        test_df = patient_data[patient_id]["test"]
        feats = compute_signal_features(test_df["glucose"].values)
        rows.append({"patient_id": patient_id, **feats})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Sanity check: SampEn(sine) << SampEn(noise); autocorr(smooth) ~ 1.
    rng = np.random.default_rng(0)
    t = np.linspace(0, 40 * np.pi, 1500)
    sine = 120 + 40 * np.sin(t)
    noise = 120 + 40 * rng.standard_normal(1500)
    ramp = np.linspace(80, 200, 1500)

    print(f"SampEn(sine)  = {sample_entropy(sine):.3f}")
    print(f"SampEn(noise) = {sample_entropy(noise):.3f}")
    print(f"autocorr(ramp)     = {lag1_autocorrelation(ramp):.3f}")
    print(f"autocorr(shuffled) = {lag1_autocorrelation(rng.permutation(ramp)):.3f}")
