"""
Standardized evaluation metrics for blood glucose forecasting.

This module implements domain-specific metrics including:
- Standard regression metrics (MAE, RMSE, MAPE)
- Clarke Error Grid Analysis (EGA)
- Parkes Error Grid Analysis (PEGA)
- Time in Range (TIR) metrics
- Continuous Glucose Error Grid Analysis (CG-EGA) TODO
- Clinical Trend Concurrence Analysis (CTCA) TODO

"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings

try:
    import pandas as pd # type: ignore
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Import the new EGA implementations
try:
    from .clarke_ega import ClarkeEGA
    CLARKE_EGA_AVAILABLE = True
except ImportError:
    CLARKE_EGA_AVAILABLE = False
    warnings.warn("Clarke EGA implementation not available")

try:
    from .parkes_ega import ParkesEGA
    PARKES_EGA_AVAILABLE = True
except ImportError:
    PARKES_EGA_AVAILABLE = False
    warnings.warn("Parkes EGA implementation not available")


class MissingDataWarning(UserWarning):
    """Raised when a metric silently drops missing values from its average."""


def _report_missing(metric: str, missing: int, total: int) -> None:
    if missing:
        warnings.warn(
            f"{metric}: {missing} of {total} values were missing and excluded "
            f"from the average ({100 * missing / total:.2f}%). The value is "
            f"computed over the remaining {total - missing}.",
            MissingDataWarning,
            stacklevel=3,
        )


def _missing_pairs(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int]:
    """Count positions where either array is missing, and the total."""
    pairwise = np.isnan(np.asarray(y_true, dtype=float)) | \
        np.isnan(np.asarray(y_pred, dtype=float))
    return int(np.count_nonzero(pairwise)), int(pairwise.size)


class OutOfDomainWarning(MissingDataWarning):
    """Raised when an error grid drops pairs that fall outside its domain."""


GRID_DOMAIN_ROUNDING_TOLERANCE = 0.001
"""How far outside a grid's domain a value may sit and still be float noise.

A sensor reading pinned at the measurement ceiling returns from the inverse
transform a fraction of a mg/dL above it -- 400.0000092345508 for a true 400 --
and that is a round-trip artefact, not an out-of-domain value. It is snapped
onto the bound rather than excluded, which is what a prediction of 700 gets.
Same value and same reasoning as ``RUN/run_zone_d_analysis.py``.
"""


def _snap_into_domain(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Pull values sitting a rounding error outside ``[low, high]`` onto the bound."""
    snapped = np.array(values, dtype=float, copy=True)
    below = (snapped < low) & (snapped >= low - GRID_DOMAIN_ROUNDING_TOLERANCE)
    above = (snapped > high) & (snapped <= high + GRID_DOMAIN_ROUNDING_TOLERANCE)
    snapped[below] = low
    snapped[above] = high
    return snapped


def _grid_inputs(y_true: np.ndarray, y_pred: np.ndarray, low: float, high: float
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare a pair of arrays for an error grid defined on ``[low, high]``.

    Returns the two arrays snapped onto the domain where they were only a
    rounding error outside it, plus a mask of the positions both grids can
    actually classify. Non-finite values are excluded alongside the genuinely
    out-of-domain ones: the Clarke range check passes a NaN -- every comparison
    against it is False -- and the zone rules then fall through to their else
    branch, so a missing pair would otherwise be counted as a benign Zone B.
    """
    reference = _snap_into_domain(y_true, low, high)
    prediction = _snap_into_domain(y_pred, low, high)
    inside = (np.isfinite(reference) & np.isfinite(prediction)
              & (reference >= low) & (reference <= high)
              & (prediction >= low) & (prediction <= high))
    return reference, prediction, inside


def _clarke_domain() -> Tuple[float, float]:
    """The domain ``ClarkeEGA.analyze`` accepts, read from the grid itself."""
    if CLARKE_EGA_AVAILABLE:
        ega = ClarkeEGA()
        return float(ega.min_range), float(ega.max_range)
    return 0.0, 400.0


def clarke_grid_series(y_true: np.ndarray, y_pred: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Restrict a pair of arrays to the Clarke grid's domain for plotting.

    Returns the pairs the grid can classify plus how many were dropped, so a
    figure shows the same points the Clarke *metric* is computed from and can
    say what it left out. ``ClarkeEGA`` rejects anything outside its domain, and
    its drawing helpers classify before they draw: handed raw output they raise,
    which renders the zone background and nothing else -- no points, no
    boundaries, no labels -- or takes down the caller when it is not guarded.
    """
    low, high = _clarke_domain()
    reference, prediction, inside = _grid_inputs(y_true, y_pred, low, high)
    return reference[inside], prediction[inside], int(np.count_nonzero(~inside))


def _report_outside_domain(metric: str, outside: int, total: int,
                           low: float, high: float) -> None:
    """
    Warn that pairs outside a grid's domain were excluded from its percentages.

    Excluded rather than clipped onto the boundary. Clipping keeps every pair
    classified, but it can only move a point to a better zone -- a prediction of
    500 against a reference of 380 becomes Zone A -- so it silently flatters the
    result; and excluded rather than fatal, because one out-of-domain pair
    should not discard the zone assignment of every other point in the run.
    """
    if outside:
        warnings.warn(
            f"{metric}: {outside} of {total} pairs fall outside the grid domain "
            f"[{low:g}, {high:g}] mg/dL or are missing, and were excluded. The "
            f"zone percentages are over the remaining {total - outside}.",
            OutOfDomainWarning,
            stacklevel=3,
        )


class BGMetrics:
    """
    Comprehensive metrics calculator for blood glucose forecasting evaluation.
    
    This class provides all standard and domain-specific metrics used in blood glucose prediction research.
    """
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Error.
        
        Args:
            y_true: True glucose values
            y_pred: Predicted glucose values
            
        Returns:
            MAE value
        """
        _report_missing("MAE", *_missing_pairs(y_true, y_pred))
        return np.nanmean(np.abs(y_true - y_pred)) # Mean Absolute Error = (1/n) * Σ|y_true - y_pred|
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Root Mean Square Error.
        
        Args:
            y_true: True glucose values
            y_pred: Predicted glucose values
            
        Returns:
            RMSE value
        """
        _report_missing("RMSE", *_missing_pairs(y_true, y_pred))
        return np.sqrt(np.nanmean((y_true - y_pred) ** 2)) # Root Mean Squared Error = sqrt((1/n) * Σ(y_true - y_pred)^2)
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Percentage Error.
        
        Args:
            y_true: True glucose values
            y_pred: Predicted glucose values
            
        Returns:
            MAPE value as percentage
        """
        _report_missing("MAPE", *_missing_pairs(y_true, y_pred))
        mask = (y_true != 0) & ~np.isnan(y_true)
        if not np.any(mask):
            warnings.warn("All true values are zero, MAPE is undefined")
            return float('inf')
        
        return np.nanmean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 # Mean Absolute Percentage Error = (1/n) * Σ(|y_true - y_pred| / y_true) * 100
    
    @staticmethod
    def mard(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Relative Difference - standard in CGM literature.
        
        Args:
            y_true: True glucose values (mg/dL)
            y_pred: Predicted glucose values (mg/dL)
            
        Returns:
            MARD value as percentage
        """
        _report_missing("MARD", *_missing_pairs(y_true, y_pred))
        
        # avoid error division by zero
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        n_zero = int(np.count_nonzero(y_true == 0))
        if n_zero:
            warnings.warn(
                f"MARD: {n_zero} of {y_true.size} true values were zero and "
                f"excluded from the average; a zero glucose reading is not a "
                f"valid measurement.",
                MissingDataWarning,
                stacklevel=2,
            )

        mask = (y_true != 0) & ~np.isnan(y_true)
        if not np.any(mask):
            warnings.warn("All true values are zero or missing, MARD is undefined")
            return float('inf')

        return np.nanmean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100 # Mean Absolute Relative Difference = (1/n) * Σ(|y_true - y_pred| / y_true) * 100

    @staticmethod
    def time_in_range(glucose_values: np.ndarray, 
                     lower_bound: float = 70, 
                     upper_bound: float = 180) -> float:
        """
        Time in Range (TIR) -> percentage of readings within target range.
        
        Args:
            glucose_values: Glucose readings (mg/dL)
            lower_bound: Lower bound of target range (default: 70 mg/dL)
            upper_bound: Upper bound of target range (default: 180 mg/dL)
            
        Returns:
            TIR as percentage
        """
        # Missing readings are excluded from the denominator, matching the
        # nan-skipping reductions used by the error metrics above; counting
        # them would put every band below its true share and stop the three
        # from summing to 100%.
        readings = np.asarray(glucose_values, dtype=float)
        _report_missing("TIR", int(np.count_nonzero(np.isnan(readings))), int(readings.size))
        readings = readings[~np.isnan(readings)]
        if readings.size == 0:
            return float('nan')
        in_range = (readings >= lower_bound) & (readings <= upper_bound)
        return np.mean(in_range) * 100
    
    @staticmethod
    def time_below_range(glucose_values: np.ndarray, threshold: float = 70) -> float:
        """
        Time Below Range (TBR) - percentage of readings below threshold.
        
        Args:
            glucose_values: Glucose readings (mg/dL)
            threshold: Threshold for hypoglycemia (default: 70 mg/dL)
            
        Returns:
            TBR as percentage
        """
        # Missing readings are excluded from the denominator, matching the
        # nan-skipping reductions used by the error metrics above; counting
        # them would put every band below its true share and stop the three
        # from summing to 100%.
        readings = np.asarray(glucose_values, dtype=float)
        _report_missing("TBR", int(np.count_nonzero(np.isnan(readings))), int(readings.size))
        readings = readings[~np.isnan(readings)]
        if readings.size == 0:
            return float('nan')
        below_range = readings < threshold
        return np.mean(below_range) * 100
    
    @staticmethod
    def time_above_range(glucose_values: np.ndarray, threshold: float = 180) -> float:
        """
        Time Above Range (TAR) - percentage of readings above threshold.
        
        Args:
            glucose_values: Glucose readings (mg/dL)
            threshold: Threshold for hyperglycemia (default: 180 mg/dL)
            
        Returns:
            TAR as percentage
        """
        # Missing readings are excluded from the denominator, matching the
        # nan-skipping reductions used by the error metrics above; counting
        # them would put every band below its true share and stop the three
        # from summing to 100%.
        readings = np.asarray(glucose_values, dtype=float)
        _report_missing("TAR", int(np.count_nonzero(np.isnan(readings))), int(readings.size))
        readings = readings[~np.isnan(readings)]
        if readings.size == 0:
            return float('nan')
        above_range = readings > threshold
        return np.mean(above_range) * 100
    
    @staticmethod
    def comparing_time_in_range(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compare glucose-range occupancy between predictions and reference.

        The argument order matches every other metric in this class. Taking the
        reference first and the prediction second previously inverted the sign
        of every value returned here.

        Args:
            y_true: True glucose readings (mg/dL)
            y_pred: Predicted glucose readings (mg/dL)

        Returns:
            Prediction minus reference, in percentage points, for each band. A
            positive ``time_in_range`` means the model places more readings
            inside 70-180 than the reference does, i.e. it overestimates time
            in range.
        """
        return {
            'time_in_range': BGMetrics.time_in_range(y_pred) - BGMetrics.time_in_range(y_true),
            'time_below_range': BGMetrics.time_below_range(y_pred) - BGMetrics.time_below_range(y_true),
            'time_above_range': BGMetrics.time_above_range(y_pred) - BGMetrics.time_above_range(y_true)
        }
    
    @staticmethod
    def clarke_error_grid_analysis(y_true: np.ndarray, 
                                  y_pred: np.ndarray) -> Dict[str, float]:
        """
        Clarke Error Grid Analysis for glucose prediction evaluation.
        
        Uses the new ClarkeEGA implementation for more accurate zone classification.
        
        Zones:
        - Zone A: Clinically acceptable (within 20% or ±15 mg/dL for values <75)
        - Zone B: Benign errors (would not lead to inappropriate treatment)
        - Zone C: Overcorrection errors (unnecessary treatment)
        - Zone D: Dangerous failures to detect (missed treatment)
        - Zone E: Erroneous treatment (opposite treatment than needed)
        
        Args:
            y_true: True glucose values (mg/dL)
            y_pred: Predicted glucose values (mg/dL)
            
        Returns:
            Dictionary with percentage of points in each zone
        """
        if CLARKE_EGA_AVAILABLE:
            # Use the new implementation
            ega = ClarkeEGA()

            # The grid is published on a bounded domain and ``analyze`` rejects
            # anything outside it, so a single out-of-domain pair would raise
            # and take the whole evaluation with it. Those pairs are excluded
            # and reported; see ``_report_outside_domain`` for why they are not
            # clipped onto the boundary instead.
            reference, prediction, inside = _grid_inputs(
                y_true, y_pred, ega.min_range, ega.max_range
            )
            _report_outside_domain("Clarke EGA", int(np.count_nonzero(~inside)),
                                   int(inside.size), ega.min_range, ega.max_range)
            if not np.any(inside):
                return {letter: float('nan') for letter in ['A', 'B', 'C', 'D', 'E']}

            results = ega.analyze(reference[inside], prediction[inside])
            
            # Convert to the expected format
            zones = {}
            for i, zone_letter in enumerate(['A', 'B', 'C', 'D', 'E']):
                zones[zone_letter] = results['percentage'][i]
            
            return zones
        else:
           raise NotImplementedError("Clarke Error Grid Analysis not available")

    @staticmethod
    def parkes_error_grid_analysis(y_true: np.ndarray, 
                                  y_pred: np.ndarray) -> Dict[str, float]:
        """
        Parkes Error Grid Analysis for Type 1 diabetes patients.
        
        Uses the new ParkesEGA implementation for more accurate zone classification.
        
        Based on 'Technical Aspects of the Parkes Error Grid':
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3876371/
        
        Zones:
        - Zone A: Clinically accurate (≤20% or ±15 mg/dL for values <75)
        - Zone B: Benign errors with little/no clinical effect
        - Zone C: Overcorrection errors (could lead to unnecessary treatment)
        - Zone D: Dangerous failure to detect/treat
        - Zone E: Erroneous treatment (opposite treatment than needed)
        
        Args:
            y_true: True glucose values (mg/dL)
            y_pred: Predicted glucose values (mg/dL)
            
        Returns:
            Dictionary with percentage of points in each zone
        """
        if PARKES_EGA_AVAILABLE:
            # Use the new implementation
            ega = ParkesEGA(units_mg_dl=True)
            results = ega.analyze(y_true, y_pred)
            
            # Parkes needs no domain guard -- it classifies any value and
            # reports what it cannot place in its own 'OOR' bucket. That bucket
            # is not part of the returned A-E shape every consumer expects, but
            # a non-empty one means the percentages below do not sum to 100, so
            # it is reported rather than dropped silently.
            out_of_range = int(results['counts']['OOR'])
            if out_of_range:
                warnings.warn(
                    f"Parkes EGA: {out_of_range} of {int(results['total_points'])} "
                    f"pairs could not be placed in any zone and are counted as "
                    f"out-of-range; the A-E percentages do not sum to 100.",
                    OutOfDomainWarning,
                    stacklevel=2,
                )

            # Convert to the expected format (exclude 'OOR' zone)
            zones = {}
            for zone_letter in ['A', 'B', 'C', 'D', 'E']:
                zones[zone_letter] = results['percentages'][zone_letter]
            
            return zones
        else:
           raise NotImplementedError("Parkes Error Grid Analysis not available")
   
    
    @staticmethod
    def calculate_comprehensive_metrics(y_true: np.ndarray,
                                      y_pred: np.ndarray
                                      ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Calculate all available metrics for blood glucose prediction evaluation.
        
        Args:
            y_true: True glucose values (mg/dL)
            y_pred: Predicted glucose values (mg/dL)

        Every metric is computed from the same prediction vector. Nothing is
        clipped: an earlier version clipped both arrays onto the CGM range for
        the error grids alone, which meant the point-error and grid columns of
        one results row described two different predictors.
            
        Returns:
            Dictionary containing all calculated metrics
        """
        metrics = {}
        
        # ADAPTED SECTION due to the switch to nan-skipping reductions: record
        # how many values were dropped, so a run with missing predictions is
        # recoverable from the written metrics and not only from a warning that
        # may have been filtered or lost to a log.
        n_missing, n_samples = _missing_pairs(y_true, y_pred)
        metrics['n_samples'] = n_samples
        metrics['n_missing'] = n_missing

        # Basic regression metrics
        metrics['rmse'] = BGMetrics.rmse(y_true, y_pred)
        metrics['mae'] = BGMetrics.mae(y_true, y_pred)
        metrics['mape'] = BGMetrics.mape(y_true, y_pred)
        metrics['mard'] = BGMetrics.mard(y_true, y_pred)
        
        # Time in range metrics
        metrics['time_in_range'] = BGMetrics.time_in_range(y_pred)
        metrics['time_below_range'] = BGMetrics.time_below_range(y_pred)
        metrics['time_above_range'] = BGMetrics.time_above_range(y_pred)
        
        # Error grid analyses. Pairs outside a grid's own domain are excluded
        # from its percentages and counted here, so a run whose model left the
        # measurement range is visible in the written metrics and not only in a
        # warning that may have been filtered or lost to a log.
        grid_low, grid_high = _clarke_domain()
        _, _, inside_grid = _grid_inputs(y_true, y_pred, grid_low, grid_high)
        metrics['n_outside_grid_domain'] = int(np.count_nonzero(~inside_grid))
        metrics['clarke_zones'] = BGMetrics.clarke_error_grid_analysis(y_true, y_pred)
        metrics['parkes_zones'] = BGMetrics.parkes_error_grid_analysis(y_true, y_pred)
        
        return metrics

