import math

from least_squares import (
    editor_rows_to_xy,
    evaluate_fit,
    initial_manual_parameters,
    least_squares,
    slider_parameter_ranges,
)


def test_least_squares_exact_line():
    beta0, beta1 = least_squares([0, 1, 2], [1, 3, 5])
    assert math.isclose(beta0, 1.0)
    assert math.isclose(beta1, 2.0)


def test_initial_manual_line_stays_near_data():
    x = [0, 1, 2, 3, 4, 5]
    y = [1.405, 3.180, 2.578, 6.047, 6.479, 6.781]
    beta0, beta1 = initial_manual_parameters(x, y)
    fit = evaluate_fit(x, y, beta0, beta1)
    y_span = max(y) - min(y)
    assert min(fit.predictions) >= min(y) - 0.30 * y_span - 1e-9
    assert max(fit.predictions) <= max(y) + 0.30 * y_span + 1e-9


def test_slider_ranges_include_mc_and_initial():
    x = [0, 1, 2, 3, 4, 5]
    y = [1.405, 3.180, 2.578, 6.047, 6.479, 6.781]
    b0_mc, b1_mc = least_squares(x, y)
    b0_start, b1_start = initial_manual_parameters(x, y)
    b0_range, b1_range = slider_parameter_ranges(x, y)
    assert b0_range[0] <= b0_mc <= b0_range[1]
    assert b0_range[0] <= b0_start <= b0_range[1]
    assert b1_range[0] <= b1_mc <= b1_range[1]
    assert b1_range[0] <= b1_start <= b1_range[1]


def test_editor_rows_dynamic_and_incomplete():
    rows = [
        {"X": 0.0, "Y": 1.0},
        {"X": 1.0, "Y": 3.0},
        {"X": 2.0, "Y": None},
        {"X": None, "Y": None},
    ]
    x, y, incomplete = editor_rows_to_xy(rows)
    assert x == (0.0, 1.0)
    assert y == (1.0, 3.0)
    assert incomplete == 1
