"""The drawn Clarke zone boundaries must follow the classifier.

The shaded background is produced by ``analyze`` itself, so any boundary line
that drifts off a real zone edge shows up directly in the published figure.
These tests read the segments back out of the rendered axes rather than
restating the coordinates, so editing the plotting code cannot quietly diverge
from what is asserted here.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from benchmark.evaluation.clarke_ega.clarke_ega import ClarkeEGA

PROBE_MARGIN = 2.0


@pytest.fixture(scope="module")
def clarke():
    return ClarkeEGA()


def _zone(clarke, y_ref, y_test):
    return clarke.analyze(np.array([float(y_ref)]), np.array([float(y_test)]))["zones"][0]


def _crosses_a_zone_edge(clarke, y_ref, y_test):
    """True when the classifier's zone changes across the line at this point."""
    return _zone(clarke, y_ref, y_test - PROBE_MARGIN) != _zone(clarke, y_ref, y_test + PROBE_MARGIN)


def _probeable(clarke, y_ref, y_test):
    """The classifier rejects values off the grid, so only probe inside it."""
    limit = clarke._max_range
    return (
        0 <= y_ref <= limit
        and PROBE_MARGIN <= y_test <= limit - PROBE_MARGIN
    )


def _drawn_boundary_segments(clarke):
    """Solid black zone boundaries as actually drawn by ``plot``."""
    figure = clarke.plot(
        np.array([150.0]), np.array([150.0]), color_zones=False, color_points=False
    )
    axes = figure.axes[0]
    segments = []
    for line in axes.lines:
        if line.get_linestyle() != "-" or line.get_linewidth() != 1.5:
            continue  # data points and the dotted perfect-agreement line
        x, y = line.get_xdata(), line.get_ydata()
        if len(x) == 2:
            segments.append(((x[0], y[0]), (x[1], y[1])))
    matplotlib.pyplot.close(figure)
    return segments


def test_every_sloped_boundary_lies_on_a_classifier_edge(clarke):
    """A sloped segment's geometry depends on arithmetic, so verify each one."""
    segments = _drawn_boundary_segments(clarke)
    sloped = [
        ((x0, y0), (x1, y1))
        for (x0, y0), (x1, y1) in segments
        if x0 != x1 and y0 != y1
    ]
    assert len(sloped) >= 4, f"expected the sloped boundaries, found {len(sloped)}"

    for (x0, y0), (x1, y1) in sloped:
        checked = 0
        for fraction in (0.2, 0.4, 0.6, 0.8):
            y_ref = x0 + fraction * (x1 - x0)
            y_test = y0 + fraction * (y1 - y0)
            if not _probeable(clarke, y_ref, y_test):
                continue
            checked += 1
            assert _crosses_a_zone_edge(clarke, y_ref, y_test), (
                f"segment ({x0:.1f},{y0:.1f})->({x1:.1f},{y1:.1f}): "
                f"drawn point ({y_ref:.1f},{y_test:.1f}) is not on a zone edge"
            )
        assert checked >= 2, (
            f"segment ({x0:.1f},{y0:.1f})->({x1:.1f},{y1:.1f}) was never probed"
        )


def test_boundaries_span_the_published_grid(clarke):
    """Clarke is defined on 0-400; the drawn grid must not exceed it."""
    assert clarke._max_range == 400
    for (x0, y0), (x1, y1) in _drawn_boundary_segments(clarke):
        assert max(x0, x1) <= clarke._max_range
        assert max(y0, y1) <= clarke._max_range


@pytest.mark.parametrize("y_ref", [80, 150, 250, 390])
def test_lower_a_edge_follows_the_point_eight_rule(clarke, y_ref):
    assert _crosses_a_zone_edge(clarke, y_ref, 0.8 * y_ref)


@pytest.mark.parametrize("y_ref", [70, 120, 200, 285])
def test_upper_c_edge_follows_the_plus_110_rule(clarke, y_ref):
    assert _crosses_a_zone_edge(clarke, y_ref, y_ref + 110)


@pytest.mark.parametrize("y_ref", [80, 150, 250, 330])
def test_upper_a_edge_follows_the_one_point_two_rule(clarke, y_ref):
    assert _crosses_a_zone_edge(clarke, y_ref, 1.2 * y_ref)


def test_values_beyond_the_published_grid_are_rejected(clarke):
    """Callers clip to the sensor range rather than extending the grid."""
    with pytest.raises(ValueError, match="physiological range"):
        clarke.analyze(np.array([150.0]), np.array([401.0]))
