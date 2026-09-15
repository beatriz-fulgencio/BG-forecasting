import numpy as np
import pytest

from benchmark.evaluation.evaluator import BGEvaluator
from benchmark.experiments.configured import _validate_evaluation_arrays


def test_evaluator_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        BGEvaluator().compute_metrics(np.array([1.0, 2.0]), np.array([1.0]))


def test_evaluator_rejects_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        BGEvaluator().compute_metrics(np.array([1.0, np.nan]), np.array([1.0, 2.0]))


def test_evaluator_requires_mg_dl_units():
    with pytest.raises(ValueError, match="require mg/dL"):
        BGEvaluator().compute_metrics(np.array([100.0]), np.array([100.0]), units="standardized")


def test_implausible_targets_are_fatal():
    """A target outside the range means broken data, not a poor model."""
    with pytest.raises(ValueError, match="targets outside the plausible"):
        _validate_evaluation_arrays(np.array([100.0, 601.0]), np.array([100.0, 100.0]))


def test_implausible_predictions_are_reported_not_fatal():
    diagnostics = _validate_evaluation_arrays(
        np.array([100.0, 120.0, 140.0, 160.0]),
        np.array([100.0, 5.0, 140.0, 900.0]),
    )
    assert diagnostics["n_implausible_predictions"] == 2
    assert diagnostics["n_points"] == 4
    assert diagnostics["implausible_prediction_rate"] == 0.5
    assert diagnostics["prediction_min_mg_dl"] == 5.0
    assert diagnostics["prediction_max_mg_dl"] == 900.0


def test_plausible_arrays_report_no_implausible_predictions():
    diagnostics = _validate_evaluation_arrays(
        np.array([100.0, 200.0]), np.array([110.0, 190.0])
    )
    assert diagnostics["n_implausible_predictions"] == 0
    assert diagnostics["implausible_prediction_rate"] == 0.0
    assert diagnostics["target_min_mg_dl"] == 100.0
    assert diagnostics["target_max_mg_dl"] == 200.0


def test_non_finite_predictions_remain_fatal():
    """A NaN makes every downstream metric undefined, so it is a failed run."""
    with pytest.raises(ValueError, match="finite"):
        _validate_evaluation_arrays(np.array([100.0]), np.array([np.nan]))


def test_shape_mismatch_is_fatal():
    with pytest.raises(ValueError, match="shape mismatch"):
        _validate_evaluation_arrays(np.array([100.0, 110.0]), np.array([100.0]))


def test_grid_metrics_clip_while_point_metrics_keep_raw_values():
    """Clipping must reach the error grids only, never MAE/RMSE."""
    y_true = np.array([150.0, 150.0])
    y_pred = np.array([150.0, 700.0])
    result = BGEvaluator().compute_metrics(
        y_true, y_pred, metrics=["mae", "clarke_ega"], grid_clip_range=(40.0, 400.0)
    )
    # MAE sees the raw 700: (|150-150| + |150-700|) / 2
    assert result["mae"] == pytest.approx(275.0)
    # Clarke sees 400 and still classifies the pair instead of raising.
    assert sum(result["clarke_zones"].values()) == pytest.approx(100.0)


def test_grid_metrics_reject_out_of_range_values_without_clipping():
    with pytest.raises(ValueError, match="physiological range"):
        BGEvaluator().compute_metrics(
            np.array([150.0]), np.array([700.0]), metrics=["clarke_ega"]
        )


def test_clarke_and_parkes_agree_once_predictions_are_clipped():
    """Clipping gives both grids the same domain, so neither reports OOR."""
    result = BGEvaluator().compute_metrics(
        np.array([300.0, 300.0]),
        np.array([300.0, 560.0]),
        metrics=["clarke_ega", "parkes_ega"],
        grid_clip_range=(40.0, 400.0),
    )
    assert sum(result["clarke_zones"].values()) == pytest.approx(100.0)
    assert sum(result["parkes_zones"].values()) == pytest.approx(100.0)
    assert "OOR" not in result["parkes_zones"] or result["parkes_zones"]["OOR"] == 0


def test_invalid_clip_range_is_rejected():
    with pytest.raises(ValueError, match="grid_clip_range"):
        BGEvaluator().compute_metrics(
            np.array([150.0]), np.array([150.0]), grid_clip_range=(400.0, 40.0)
        )


def test_diagnostics_count_predictions_clipped_for_grids():
    diagnostics = _validate_evaluation_arrays(
        np.array([100.0, 100.0, 100.0, 100.0]),
        np.array([30.0, 100.0, 450.0, 200.0]),
    )
    # 30 and 450 fall outside the CGM range but inside the plausible range.
    assert diagnostics["n_predictions_clipped_for_grids"] == 2
    assert diagnostics["clipped_prediction_rate"] == 0.5
    assert diagnostics["n_implausible_predictions"] == 0


def test_clipping_absorbs_round_trip_error_at_the_sensor_ceiling():
    """A target pinned at 400 mg/dL returns from the inverse transform slightly
    above 400, so the grids must clip targets too, not only predictions."""
    y_true = np.array([400.0000092345508])
    y_pred = np.array([390.0])
    with pytest.raises(ValueError, match="physiological range"):
        BGEvaluator().compute_metrics(y_true, y_pred, metrics=["clarke_ega"])
    result = BGEvaluator().compute_metrics(
        y_true, y_pred, metrics=["clarke_ega"], grid_clip_range=(40.0, 400.0)
    )
    assert sum(result["clarke_zones"].values()) == pytest.approx(100.0)


def test_tir_difference_is_prediction_minus_reference():
    """Sign convention: positive means the model overestimates time in range."""
    reference = np.array([300.0] * 10)   # 0% in range, 100% above
    prediction = np.array([120.0] * 10)  # 100% in range
    tir = BGEvaluator().compute_metrics(reference, prediction, metrics=["tir"])["tir"]
    assert tir["time_in_range"] == pytest.approx(100.0)
    assert tir["time_above_range"] == pytest.approx(-100.0)
    assert tir["time_below_range"] == pytest.approx(0.0)


def test_tir_difference_reverses_when_the_model_underestimates():
    reference = np.array([120.0] * 10)   # 100% in range
    prediction = np.array([50.0] * 10)   # 100% below range
    tir = BGEvaluator().compute_metrics(reference, prediction, metrics=["tir"])["tir"]
    assert tir["time_in_range"] == pytest.approx(-100.0)
    assert tir["time_below_range"] == pytest.approx(100.0)


def test_tir_difference_is_zero_when_bands_match():
    values = np.array([60.0, 120.0, 200.0])
    tir = BGEvaluator().compute_metrics(values, values.copy(), metrics=["tir"])["tir"]
    assert all(v == pytest.approx(0.0) for v in tir.values())
