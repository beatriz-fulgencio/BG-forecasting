"""The glucose bounds shared by the training, evaluation and analysis paths.

Kept in a module of its own, with no imports, so that a lightweight analysis can
reuse the policy without pulling in the training stack: the pair used to live in
:mod:`benchmark.experiments.configured`, which imports torch.

The two ranges answer different questions and are not interchangeable. The
plausibility bounds reject garbage (NaN, the ``-1`` missing-value sentinel, a
mis-scaled unit) while staying agnostic about where a reading came from. The
sensor range is a property of one instrument, and belongs only where the
argument is about what a CGM can physically report.
"""

# Broad measurement plausibility bounds in the units used by the clinical
# metrics. These are deliberately wider than the usual treatment range: a real
# hypo below 40 or a DKA reading above 400 is physiologically possible, and a
# filter meant to catch corrupt values should not discard it.
GLUCOSE_PLAUSIBLE_RANGE_MG_DL = (20.0, 600.0)

# The OhioT1DM CGM reports only within this interval and saturates at both ends,
# so every target is already censored to it. Predictions are NOT clipped to it:
# every metric is computed from the raw prediction vector, and the error grids
# exclude and report whatever falls outside their own domain. This range is kept
# only to count how far outside the measurement range a model's output strayed,
# which is how a model that needs constraining makes itself known.
CGM_SENSOR_RANGE_MG_DL = (40.0, 400.0)
