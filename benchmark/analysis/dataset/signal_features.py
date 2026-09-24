"""
Temporal-irregularity features of a patient's glucose signal.

These quantify how *erratic / unpredictable* a CGM series is over time, an axis
distinct from spread (``std``) and from train->test distributional shift
(``shift_score``). Two patients can share the same variance yet differ sharply in
short-term predictability. Two complementary measures are provided:

Reference implementations : Sample entropy: ``antropy.sample_entropy`` (BSD-3-Clause),
  https://github.com/raphaelvallat/antropy, and ``nolds.sampen`` (MIT),
  https://github.com/CSchoel/nolds.

"""

from typing import Dict

import numpy as np
import pandas as pd  # type: ignore
from statsmodels.tsa.stattools import acf  # type: ignore

from .distribution_shift import GLUCOSE_MIN, GLUCOSE_MAX

# Cap the series length used for SampEn. The template-matching is O(N^2); with
# ~2500 CGM samples per patient this keeps each patient well under a second while
# preserving the signal's irregularity structure. Uses the most recent samples.
_SAMPEN_MAX_SAMPLES = 2000


def _valid_mask(values: np.ndarray):
    """Boolean mask of physiologically valid readings (drops the ``-1`` sentinel)."""
    return np.isfinite(values) & (values >= GLUCOSE_MIN) & (values <= GLUCOSE_MAX)


def _as_1d_float(glucose):
    """Convert a public signal input to the one-dimensional series we require."""
    values = np.asarray(glucose, dtype=float)
    if values.ndim != 1:
        raise ValueError("glucose must be a one-dimensional series")
    return values


def lag1_autocorrelation(glucose):
    """
    Lag-1 autocorrelation of a glucose series, from ``statsmodels``.

    Returns:
        The lag-1 autocorrelation, or ``np.nan`` if fewer than two readings are
        observed, the observed readings are constant, or no two observed
        readings are adjacent.
    """
    values = _as_1d_float(glucose)
    observed = _valid_mask(values)

    if np.count_nonzero(observed) < 2 or np.std(values[observed]) == 0:
        return float("nan")

    # The one departure from ``acf``: with no adjacent observed pair it returns
    # 0.0, which reads as "no persistence" when the truth is "no information".
    if not np.any(observed[:-1] & observed[1:]):
        return float("nan")

    series = np.where(observed, values, np.nan)
    return float(acf(series, nlags=1, fft=False, missing="conservative", adjusted=True)[1])


def _sample_entropy_details(glucose, m: int = 2, r: float = None):
    """Return SampEn plus the validity and template counts behind it."""
    if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or m < 1:
        raise ValueError("m must be a positive integer")
    if r is not None:
        try:
            r = float(r)
        except (TypeError, ValueError) as error:
            raise ValueError("r must be a finite, non-negative number") from error
        if not np.isfinite(r) or r < 0:
            raise ValueError("r must be a finite, non-negative number")

    raw = _as_1d_float(glucose)
    valid = _valid_mask(raw)
    n_valid = int(valid.sum())

    # Cap by *valid observations* but retain every raw position after the cap
    # boundary. This preserves missing-data gaps rather than closing them.
    if n_valid > _SAMPEN_MAX_SAMPLES:
        start = np.flatnonzero(valid)[-_SAMPEN_MAX_SAMPLES]
        values, valid = raw[start:], valid[start:]
    else:
        values = raw
    n_used = int(valid.sum())
    result: Dict[str, float | int] = {
        "sample_entropy": float("nan"),
        "sample_entropy_valid_n": n_valid,
        "sample_entropy_used_n": n_used,
        "sample_entropy_template_n": 0,
    }

    if values.size <= m + 1 or n_used <= m + 1:
        return result

    observed_values = values[valid]
    if r is None:
        sd = np.std(observed_values)
        if sd == 0:
            # A constant observed signal is maximally regular. Gaps do not
            # change that interpretation, provided it has usable templates.
            r = 0.0
        else:
            r = 0.2 * sd

    # Use the same candidate starts for m and m+1, as required by the SampEn
    # definition, and require every point of the longer template to be valid.
    # Thus no template can bridge a CGM dropout.
    starts = np.array([
        i for i in range(values.size - m)
        if np.all(valid[i:i + m + 1])
    ], dtype=int)
    result["sample_entropy_template_n"] = int(starts.size)
    if starts.size < 2:
        return result

    def _count_matches(template_len: int):
        templates = np.array([values[i:i + template_len] for i in starts])
        count = 0
        for i in range(len(templates) - 1):
            dist = np.max(np.abs(templates[i + 1:] - templates[i]), axis=1)
            count += int(np.sum(dist <= r))
        return count

    b = _count_matches(m)
    a = _count_matches(m + 1)
    if b and a:
        result["sample_entropy"] = float(-np.log(a / b))
    return result


def sample_entropy(glucose, m: int = 2, r: float = None):
    """
    Sample entropy (SampEn) of a glucose series.
    """
    return float(_sample_entropy_details(glucose, m=m, r=r)["sample_entropy"])


def compute_signal_features(glucose, m: int = 2, r: float = None):
    """
    Compute the temporal-irregularity feature set for one glucose series.

    Returns:
        Dict with the two feature values plus valid/template/pair counts. The
        counts let callers distinguish an undefined signal from one estimated
        from sparse data.
    """
    values = _as_1d_float(glucose)
    valid = _valid_mask(values)
    entropy = _sample_entropy_details(values, m=m, r=r)
    return {
        **entropy,
        "autocorr_lag1": lag1_autocorrelation(values),
        "autocorr_valid_n": int(valid.sum()),
        "autocorr_adjacent_pair_n": int(np.sum(valid[:-1] & valid[1:])),
    }


def compute_patient_signal_features(patient_data: Dict):
    """
    Compute per-patient temporal-irregularity features from the test glucose.

    Mirrors :func:`distribution_shift.compute_patient_shift_scores`.

    Args:
        patient_data: Mapping ``{patient_id: {'train': df, 'test': df}}`` where
            each DataFrame has a ``'glucose'`` column (the structure returned by
            ``PopulationAnalysis.load_and_preprocess_glucose_data``). Features are
            computed from the ``test`` series.

    Returns:
        DataFrame with feature values and their validity/template counts,
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
